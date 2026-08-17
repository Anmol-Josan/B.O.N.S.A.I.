"""Reference continual-learning baselines on a common ResNet-18 backbone."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn


def build_resnet18(num_classes: int, input_channels: int = 3) -> nn.Module:
    """Construct an uninitialized torchvision ResNet-18 with a new classifier."""

    try:
        from torchvision.models import resnet18
    except ImportError as error:  # pragma: no cover - depends on optional runtime
        raise ImportError("torchvision is required for ResNet-18 baselines") from error
    backbone = resnet18(weights=None)
    if input_channels != 3:
        backbone.conv1 = nn.Conv2d(
            input_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    return backbone


class ResNetBaseline(nn.Module):
    """Common wrapper used to compare methods with identical initial capacity."""

    def __init__(self, num_classes: int, input_channels: int = 3) -> None:
        super().__init__()
        self.backbone = build_resnet18(num_classes, input_channels)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.backbone(inputs)

    @property
    def total_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class EWC(ResNetBaseline):
    """Elastic Weight Consolidation using empirical diagonal Fisher estimates."""

    def __init__(self, num_classes: int, input_channels: int = 3, penalty_weight: float = 1.0) -> None:
        super().__init__(num_classes, input_channels)
        self.penalty_weight = penalty_weight
        self.fisher: dict[str, Tensor] = {}
        self.optimal_params: dict[str, Tensor] = {}

    def consolidate(self, loader, max_batches: int | None = None) -> None:
        self.train()
        fisher = {
            name: torch.zeros_like(parameter)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        batches = 0
        for inputs, labels in loader:
            self.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(self(inputs), labels)
            loss.backward()
            for name, parameter in self.named_parameters():
                if parameter.grad is not None:
                    fisher[name].add_(parameter.grad.detach().square())
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break
        if batches == 0:
            raise ValueError("cannot consolidate EWC with an empty loader")
        self.fisher = {name: value / batches for name, value in fisher.items()}
        self.optimal_params = {
            name: parameter.detach().clone() for name, parameter in self.named_parameters()
        }

    def ewc_penalty(self) -> Tensor:
        if not self.fisher:
            return torch.zeros((), device=next(self.parameters()).device)
        return self.penalty_weight * sum(
            (self.fisher[name] * (parameter - self.optimal_params[name]).square()).sum()
            for name, parameter in self.named_parameters()
            if name in self.fisher
        )


class SI(ResNetBaseline):
    """Synaptic Intelligence with an online importance accumulator."""

    def __init__(self, num_classes: int, input_channels: int = 3, penalty_weight: float = 1.0) -> None:
        super().__init__(num_classes, input_channels)
        self.penalty_weight = penalty_weight
        self.importance: dict[str, Tensor] = {}
        self.reference_params: dict[str, Tensor] = {
            name: parameter.detach().clone() for name, parameter in self.named_parameters()
        }

    def update_synaptic_importance(self, previous_params: Mapping[str, Tensor]) -> None:
        """Accumulate a stable magnitude proxy for the task's parameter changes."""

        for name, parameter in self.named_parameters():
            previous = previous_params.get(name, parameter.detach())
            contribution = (parameter.detach() - previous.detach()).abs()
            self.importance[name] = self.importance.get(name, torch.zeros_like(parameter)) + contribution

    def consolidate_task(self) -> None:
        self.reference_params = {
            name: parameter.detach().clone() for name, parameter in self.named_parameters()
        }

    def si_penalty(self) -> Tensor:
        if not self.importance:
            return torch.zeros((), device=next(self.parameters()).device)
        return self.penalty_weight * sum(
            (self.importance[name] * (parameter - self.reference_params[name]).square()).sum()
            for name, parameter in self.named_parameters()
            if name in self.importance
        )


class PackNet(ResNetBaseline):
    """Magnitude-based pruning and freezing baseline."""

    def __init__(self, num_classes: int, input_channels: int = 3) -> None:
        super().__init__(num_classes, input_channels)
        self.masks: dict[str, Tensor] = {}

    def prune_by_magnitude(self, fraction_to_freeze: float = 0.5) -> dict[str, Tensor]:
        if not 0.0 <= fraction_to_freeze <= 1.0:
            raise ValueError("fraction_to_freeze must be in [0, 1]")
        parameters = {name: parameter for name, parameter in self.named_parameters()}
        threshold = torch.quantile(
            torch.cat([parameter.detach().abs().flatten() for parameter in parameters.values()]),
            fraction_to_freeze,
        )
        self.masks = {name: parameter.detach().abs().ge(threshold) for name, parameter in parameters.items()}
        return self.masks

    @property
    def frozen_parameter_count(self) -> int:
        return sum(mask.sum().item() for mask in self.masks.values())

    def apply_gradient_masks(self) -> None:
        for name, parameter in self.named_parameters():
            if parameter.grad is not None and name in self.masks:
                parameter.grad.mul_(~self.masks[name].to(parameter.grad.device))


class PNN(nn.Module):
    """Progressive Neural Network with a frozen ResNet-18 column per task."""

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        task_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        if task_classes is not None and task_classes < 1:
            raise ValueError("task_classes must be positive when provided")
        self.task_classes = task_classes
        output_classes = task_classes if task_classes is not None else num_classes
        self.columns = nn.ModuleList([build_resnet18(output_classes, input_channels)])

    @property
    def backbone(self) -> nn.Module:
        return self.columns[0]

    @property
    def total_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def add_task_column(self) -> nn.Module:
        output_classes = self.task_classes if self.task_classes is not None else self.num_classes
        column = build_resnet18(output_classes, self.input_channels)
        self.columns.append(column)
        return column

    def forward(self, inputs: Tensor, task_id: int | None = None) -> Tensor:
        index = len(self.columns) - 1 if task_id is None else task_id
        return self.columns[index](inputs)
