"""ResNet-18 BONSAI model combining VIB and localized dynamic junctions."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.algorithms.baselines import build_resnet18
from src.models.dynamic_resnet import ResidualJunction
from src.models.adapters import BottleneckAdapter, ConvBottleneckAdapter
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
        classes_per_task: int | None = None,
        route_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = build_resnet18(num_classes=num_classes)
        self.backbone.fc = nn.Identity()
        self.feature_dim = 512
        self.beta = beta
        if task_adapter_rank < 0:
            raise ValueError("task_adapter_rank must be nonnegative")
        if classes_per_task is not None and classes_per_task < 1:
            raise ValueError("classes_per_task must be positive when provided")
        if route_hidden_dim < 0:
            raise ValueError("route_hidden_dim must be nonnegative")
        self.task_adapter_rank = task_adapter_rank
        self.classes_per_task = classes_per_task
        self.route_hidden_dim = route_hidden_dim
        self.junction_growth_ratio = junction_growth_ratio
        self.plateau_patience = plateau_patience
        self.plateau_min_improvement = plateau_min_improvement
        self.vib = VIBLinear(self.feature_dim, self.feature_dim)
        nn.init.constant_(self.vib.logvar_layer.weight, 0.0)
        nn.init.constant_(self.vib.logvar_layer.bias, -4.0)
        self.junctions = nn.ModuleList()
        self.task_adapters = nn.ModuleList()
        self.task_stage_adapters = nn.ModuleList()
        self.task_heads = nn.ModuleList()
        self.route_compatibility_heads = nn.ModuleList()
        self.num_tasks = (
            num_classes // classes_per_task
            if classes_per_task is not None and num_classes % classes_per_task == 0
            else 0
        )
        self.route_head = (
            nn.Linear(self.feature_dim, self.num_tasks) if self.num_tasks > 0 else None
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._route_prototypes: list[Tensor] = []
        self._route_valid: list[Tensor] = []
        self._route_scales: list[Tensor] = []
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
            self.task_stage_adapters.append(nn.ModuleList([nn.Identity() for _ in range(4)]))
            adapter = None
        else:
            device = next(self.parameters()).device
            adapter = BottleneckAdapter(self.feature_dim, self.task_adapter_rank).to(device)
            self.task_adapters.append(adapter)
            stage_channels = (64, 128, 256, 512)
            self.task_stage_adapters.append(
                nn.ModuleList(
                    [
                        ConvBottleneckAdapter(channels, self.task_adapter_rank).to(device)
                        for channels in stage_channels
                    ]
                )
            )
        if self.classes_per_task is not None:
            device = next(self.parameters()).device
            self.task_heads.append(
                nn.Linear(self.feature_dim, self.classes_per_task).to(device)
            )
        device = next(self.parameters()).device
        if self.route_hidden_dim == 0:
            compatibility_head: nn.Module = nn.Linear(self.feature_dim, 1)
        else:
            compatibility_head = nn.Sequential(
                nn.Linear(self.feature_dim, self.route_hidden_dim),
                nn.GELU(),
                nn.Linear(self.route_hidden_dim, 1),
            )
        self.route_compatibility_heads.append(compatibility_head.to(device))
        return adapter

    def task_parameters(self, task_id: int) -> list[nn.Parameter]:
        """Return shared parameters plus the current task's private adapter."""

        if not 0 <= task_id < len(self.task_adapters):
            raise IndexError(f"task path {task_id} has not been allocated")
        current_adapter = {id(parameter) for parameter in self.task_adapters[task_id].parameters()}
        current_adapter.update(
            id(parameter) for parameter in self.task_stage_adapters[task_id].parameters()
        )
        if self.classes_per_task is not None:
            current_adapter.update(id(parameter) for parameter in self.task_heads[task_id].parameters())
        current_adapter.update(
            id(parameter) for parameter in self.route_compatibility_heads[task_id].parameters()
        )
        adapter_parameter_ids = {
            id(parameter)
            for adapter in self.task_adapters
            for parameter in adapter.parameters()
        }
        adapter_parameter_ids.update(
            id(parameter)
            for stage_adapters in self.task_stage_adapters
            for parameter in stage_adapters.parameters()
        )
        adapter_parameter_ids.update(
            id(parameter)
            for head in self.task_heads
            for parameter in head.parameters()
        )
        adapter_parameter_ids.update(
            id(parameter)
            for head in self.route_compatibility_heads
            for parameter in head.parameters()
        )
        return [
            parameter
            for parameter in self.parameters()
            if id(parameter) in current_adapter or id(parameter) not in adapter_parameter_ids
        ]

    def task_parameter_groups(
        self,
        task_id: int,
        learning_rate: float,
        shared_learning_rate_scale: float = 1.0,
    ) -> list[dict[str, object]]:
        """Separate shared-scaffold and current-path optimization rates."""

        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if shared_learning_rate_scale < 0.0:
            raise ValueError("shared_learning_rate_scale must be nonnegative")
        if not 0 <= task_id < len(self.task_adapters):
            raise IndexError(f"task path {task_id} has not been allocated")
        current_ids = {id(parameter) for parameter in self.task_adapters[task_id].parameters()}
        current_ids.update(
            id(parameter) for parameter in self.task_stage_adapters[task_id].parameters()
        )
        if self.classes_per_task is not None:
            current_ids.update(id(parameter) for parameter in self.task_heads[task_id].parameters())
        current_ids.update(
            id(parameter) for parameter in self.route_compatibility_heads[task_id].parameters()
        )
        task_parameters = self.task_parameters(task_id)
        current = [parameter for parameter in task_parameters if id(parameter) in current_ids]
        shared = [parameter for parameter in task_parameters if id(parameter) not in current_ids]
        groups: list[dict[str, object]] = []
        if shared and shared_learning_rate_scale > 0.0:
            groups.append({"params": shared, "lr": learning_rate * shared_learning_rate_scale})
        if current:
            groups.append({"params": current, "lr": learning_rate})
        if not groups:
            raise ValueError("task path has no trainable parameters")
        return groups

    def task_logits(self, inputs: Tensor, task_id: int) -> Tensor:
        """Return logits from a task-local head when one is configured."""

        if not 0 <= task_id < len(self.task_adapters):
            raise IndexError(f"task path {task_id} has not been allocated")
        features = self.forward_features(inputs, task_id=task_id)
        if self.classes_per_task is None:
            return self.classifier(features)
        return self.task_heads[task_id](features)

    def route_logits(self, inputs: Tensor) -> Tensor:
        """Return learned task-gate logits from shared bottleneck features."""

        if self.route_head is None:
            raise RuntimeError("route head requires classes_per_task at construction")
        return self.route_head(self.forward_features(inputs))

    def route_logits_from_features(self, features: Tensor) -> Tensor:
        """Apply the task gate to precomputed shared features."""

        if self.route_head is None:
            raise RuntimeError("route head requires classes_per_task at construction")
        return self.route_head(features)

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

    def start_task(self) -> None:
        """Reset the growth tracker when a new task begins."""

        self.best_validation_loss = math.inf
        self.plateau_epochs = 0

    @torch.no_grad()
    def register_task_route(
        self,
        task_id: int,
        inputs: Tensor,
        global_labels: Tensor,
        classes_per_task: int,
    ) -> None:
        """Store class-conditional bottleneck prototypes for task-free routing."""

        if not 0 <= task_id < len(self.task_adapters):
            raise IndexError(f"task path {task_id} has not been allocated")
        if classes_per_task < 1:
            raise ValueError("classes_per_task must be positive")
        if inputs.shape[0] != global_labels.shape[0]:
            raise ValueError("inputs and global_labels must have the same batch length")
        was_training = self.training
        self.eval()
        features = self.forward_features(inputs, task_id=task_id)
        local_labels = global_labels.long() - task_id * classes_per_task
        if ((local_labels < 0) | (local_labels >= classes_per_task)).any():
            raise ValueError("global_labels contain classes outside the task's expected range")
        prototypes = torch.zeros(
            classes_per_task, self.feature_dim, device=features.device, dtype=features.dtype
        )
        valid = torch.zeros(classes_per_task, dtype=torch.bool, device=features.device)
        for local_class in range(classes_per_task):
            selected = features[local_labels == local_class]
            if selected.numel() == 0:
                continue
            prototypes[local_class].copy_(selected.mean(dim=0))
            valid[local_class] = True
        if not valid.any():
            raise ValueError(f"task {task_id} has no labels in its expected class range")
        assigned = prototypes[local_labels.clamp(0, classes_per_task - 1)]
        scale = (features - assigned).square().mean().sqrt().clamp_min(1e-3)
        if task_id == len(self._route_prototypes):
            self._route_prototypes.append(prototypes.detach().clone())
            self._route_valid.append(valid.detach().clone())
            self._route_scales.append(scale.detach().clone())
        elif task_id < len(self._route_prototypes):
            if self._route_prototypes[task_id].shape != prototypes.shape:
                raise ValueError("route prototype shape changed for an existing task")
            self._route_prototypes[task_id] = prototypes.detach().clone()
            self._route_valid[task_id] = valid.detach().clone()
            self._route_scales[task_id] = scale.detach().clone()
        else:
            raise ValueError("task routes must be registered sequentially")
        if was_training:
            self.train()

    @torch.no_grad()
    def predict_task_free(
        self,
        inputs: Tensor,
        classes_per_task: int,
        route_strategy: str = "prototype",
        prototype_weight: float = 1.0,
        route_head_weight: float = 1.0,
        global_route_weight: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Predict global classes without receiving a task ID.

        Each allocated task path produces local logits for its private head. The
        route score is computed only over that task's class range, preventing a
        task from winning merely because it assigns low probability to classes
        it could never own.
        """

        if route_strategy not in {
            "entropy",
            "prototype",
            "hybrid",
            "learned",
            "scaffold",
            "compatibility",
        }:
            raise ValueError(
                "route_strategy must be entropy, prototype, hybrid, learned, scaffold, or compatibility"
            )
        if classes_per_task < 1:
            raise ValueError("classes_per_task must be positive")
        if prototype_weight < 0.0:
            raise ValueError("prototype_weight must be nonnegative")
        if route_head_weight < 0.0:
            raise ValueError("route_head_weight must be nonnegative")
        if global_route_weight < 0.0:
            raise ValueError("global_route_weight must be nonnegative")
        if not self.task_adapters:
            raise RuntimeError("no task paths have been allocated")
        if self.classes_per_task is not None and self.classes_per_task != classes_per_task:
            raise ValueError("classes_per_task does not match the model's task heads")
        was_training = self.training
        self.eval()
        logits = torch.stack(
            [self.task_logits(inputs, task_id=task_id) for task_id in range(len(self.task_adapters))],
            dim=1,
        )
        if logits.shape[-1] != classes_per_task:
            local_logits = torch.stack(
                [
                    logits[:, task_id, task_id * classes_per_task : (task_id + 1) * classes_per_task]
                    for task_id in range(len(self.task_adapters))
                ],
                dim=1,
            )
        else:
            local_logits = logits
        probabilities = local_logits.softmax(dim=-1)
        entropies = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        use_prototypes = (
            route_strategy in {"prototype", "hybrid"}
            and len(self._route_prototypes) == len(self.task_adapters)
            and all(valid.any() for valid in self._route_valid)
        )
        needs_shared_features = use_prototypes or route_strategy in {"learned", "scaffold"}
        shared_features = self.forward_features(inputs) if needs_shared_features else None
        global_log_prob = None
        global_task_log_mass = None
        if shared_features is not None and self.classifier.out_features >= len(self.task_adapters) * classes_per_task:
            global_log_prob = self.classifier(shared_features).log_softmax(dim=-1)
            global_task_log_mass = torch.stack(
                [
                    torch.logsumexp(
                        global_log_prob[:, task_id * classes_per_task : (task_id + 1) * classes_per_task],
                        dim=-1,
                    )
                    for task_id in range(len(self.task_adapters))
                ],
                dim=1,
            )
        if use_prototypes:
            routed_features = torch.stack(
                [
                    self.forward_features(inputs, task_id=task_id)
                    for task_id in range(len(self.task_adapters))
                ],
                dim=1,
            )
            prototypes = torch.stack(
                [prototype.to(device=inputs.device) for prototype in self._route_prototypes], dim=0
            )
            valid = torch.stack(
                [task_valid.to(device=inputs.device) for task_valid in self._route_valid], dim=0
            )
            distances = (
                routed_features.unsqueeze(2) - prototypes.unsqueeze(0)
            ).square().mean(dim=-1)
            distances = distances.masked_fill(~valid.unsqueeze(0), float("inf"))
            distances = distances.min(dim=2).values / torch.stack(
                [scale.to(device=inputs.device) for scale in self._route_scales], dim=0
            ).clamp_min(1e-3).unsqueeze(0)
            if route_strategy == "prototype":
                scores = distances
            else:
                entropy_scale = torch.log(
                    torch.tensor(float(classes_per_task), device=inputs.device)
                ).clamp_min(1e-8)
                scores = distances + prototype_weight * entropies / entropy_scale
                if self.route_head is not None and self.route_head.out_features >= len(
                    self.task_adapters
                ):
                    route_log_prob = self.route_logits_from_features(shared_features)[:, : len(
                        self.task_adapters
                    )].log_softmax(dim=-1)
                    scores = scores - route_head_weight * route_log_prob
                if global_task_log_mass is not None:
                    scores = scores - global_route_weight * global_task_log_mass
        elif route_strategy == "compatibility":
            compatibility_logits = torch.stack(
                [
                    self.route_compatibility_heads[task_id](
                        self.forward_features(inputs, task_id=task_id)
                    ).squeeze(-1)
                    for task_id in range(len(self.task_adapters))
                ],
                dim=1,
            )
            scores = -compatibility_logits
        elif route_strategy in {"learned", "scaffold"}:
            if route_strategy == "learned" and self.route_head is None:
                scores = entropies
            else:
                scores = torch.zeros_like(entropies)
                if route_strategy == "learned":
                    if self.route_head.out_features < len(self.task_adapters):
                        raise ValueError("route head size does not match allocated task paths")
                    scores = scores - route_head_weight * self.route_logits_from_features(
                        shared_features
                    )[:, : len(self.task_adapters)].log_softmax(dim=-1)
                if global_task_log_mass is not None:
                    scores = scores - global_route_weight * global_task_log_mass
                else:
                    scores = entropies
        else:
            scores = entropies
        selected_tasks = scores.argmin(dim=1)
        batch_indices = torch.arange(inputs.shape[0], device=inputs.device)
        if route_strategy == "scaffold" and global_log_prob is not None:
            selected_global_logits = self.classifier(shared_features)
            local_predictions = torch.stack(
                [
                    selected_global_logits[
                        :, task_id * classes_per_task : (task_id + 1) * classes_per_task
                    ]
                    for task_id in range(len(self.task_adapters))
                ],
                dim=1,
            )
            selected_local = local_predictions[batch_indices, selected_tasks].argmax(dim=1)
        else:
            selected_local = local_logits[batch_indices, selected_tasks].argmax(dim=1)
        global_predictions = selected_tasks * classes_per_task + selected_local
        if was_training:
            self.train()
        return global_predictions, selected_tasks, entropies

    def forward_features(self, inputs: Tensor, task_id: int | None = None) -> Tensor:
        backbone = self.backbone
        features = backbone.conv1(inputs)
        features = backbone.bn1(features)
        features = backbone.relu(features)
        features = backbone.maxpool(features)
        for stage_index, stage in enumerate(
            (backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4)
        ):
            features = stage(features)
            if task_id is not None:
                if not 0 <= task_id < len(self.task_stage_adapters):
                    raise IndexError(f"task path {task_id} has not been allocated")
                features = self.task_stage_adapters[task_id][stage_index](features)
        features = backbone.avgpool(features)
        features = torch.flatten(features, 1)
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
