"""ResNet-18 BONSAI model combining VIB and localized dynamic junctions."""

from __future__ import annotations

import math

from torch import Tensor, nn

from src.algorithms.baselines import build_resnet18
from src.models.dynamic_resnet import ResidualJunction
from src.models.adapters import BottleneckAdapter
from src.models.vib_layers import VIBLinear


class BonsaiResNet18(nn.Module):
    """ResNet-18 feature extractor with a stochastic bottleneck and adapters."""

    def __init__(
        self,
        num_classes: int,
        junction_growth_ratio: float = 0.04,
        plateau_patience: int = 5,
        plateau_min_improvement: float = 0.01,
        beta: float = 0.001,
        task_adapter_rank: int = 8,
    ) -> None:
        super().__init__()
        self.backbone = build_resnet18(num_classes=num_classes)
        self.backbone.fc = nn.Identity()
        self.feature_dim = 512
        self.beta = beta
        if task_adapter_rank < 0:
            raise ValueError("task_adapter_rank must be nonnegative")
        self.task_adapter_rank = task_adapter_rank
        self.junction_growth_ratio = junction_growth_ratio
        self.plateau_patience = plateau_patience
        self.plateau_min_improvement = plateau_min_improvement
        self.vib = VIBLinear(self.feature_dim, self.feature_dim)
        nn.init.constant_(self.vib.logvar_layer.weight, 0.0)
        nn.init.constant_(self.vib.logvar_layer.bias, -4.0)
        self.junctions = nn.ModuleList()
        self.task_adapters = nn.ModuleList()
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.initial_parameter_count = self.total_parameters
        self.best_validation_loss = math.inf
        self.plateau_epochs = 0

    @property
    def kl_loss(self) -> Tensor:
        return self.vib.kl_loss

    @property
    def total_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def parameter_overhead(self) -> float:
        return (self.total_parameters - self.initial_parameter_count) / self.initial_parameter_count

    def _junction_hidden_dim(self) -> int:
        target = self.initial_parameter_count * self.junction_growth_ratio
        estimate = max(1, round((target - self.feature_dim) / (2 * self.feature_dim + 1)))
        candidates = range(max(1, estimate - 3), estimate + 4)
        return min(
            candidates,
            key=lambda hidden: abs(2 * self.feature_dim * hidden + hidden + self.feature_dim - target),
        )

    def expand(self) -> ResidualJunction:
        device = next(self.parameters()).device
        junction = ResidualJunction(self.feature_dim, self._junction_hidden_dim()).to(device)
        self.junctions.append(junction)
        return junction

    def add_task_path(self) -> nn.Module | None:
        """Allocate a small private residual adapter for one new task."""

        if self.task_adapter_rank == 0:
            self.task_adapters.append(nn.Identity())
            return None
        device = next(self.parameters()).device
        adapter = BottleneckAdapter(self.feature_dim, self.task_adapter_rank).to(device)
        self.task_adapters.append(adapter)
        return adapter

    def task_parameters(self, task_id: int) -> list[nn.Parameter]:
        """Return shared parameters plus the current task's private adapter."""

        if not 0 <= task_id < len(self.task_adapters):
            raise IndexError(f"task path {task_id} has not been allocated")
        current_adapter = {id(parameter) for parameter in self.task_adapters[task_id].parameters()}
        adapter_parameter_ids = {
            id(parameter)
            for adapter in self.task_adapters
            for parameter in adapter.parameters()
        }
        return [
            parameter
            for parameter in self.parameters()
            if id(parameter) in current_adapter or id(parameter) not in adapter_parameter_ids
        ]

    def record_validation_loss(self, validation_loss: float) -> bool:
        if not math.isfinite(validation_loss) or validation_loss < 0.0:
            raise ValueError("validation_loss must be a finite nonnegative number")
        if math.isinf(self.best_validation_loss):
            self.best_validation_loss = validation_loss
            return False
        if validation_loss < self.best_validation_loss * (1.0 - self.plateau_min_improvement):
            self.best_validation_loss = validation_loss
            self.plateau_epochs = 0
            return False
        self.plateau_epochs += 1
        if self.plateau_epochs < self.plateau_patience:
            return False
        self.expand()
        self.best_validation_loss = validation_loss
        self.plateau_epochs = 0
        return True

    def forward_features(self, inputs: Tensor, task_id: int | None = None) -> Tensor:
        features = self.backbone(inputs)
        features = self.vib(features)
        for junction in self.junctions:
            features = junction(features)
        if task_id is not None:
            if not 0 <= task_id < len(self.task_adapters):
                raise IndexError(f"task path {task_id} has not been allocated")
            features = self.task_adapters[task_id](features)
        return features

    def forward(self, inputs: Tensor, task_id: int | None = None) -> Tensor:
        return self.classifier(self.forward_features(inputs, task_id=task_id))
