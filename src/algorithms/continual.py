"""Fast toy continual-learning loop used to validate BONSAI invariants."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.models.adapters import BottleneckAdapter, IdentityAdapter
from src.models.vib_layers import VIBLinear


class ToyContinualLearner(nn.Module):
    """Shared VIB encoder with low-rank task-local classifier pathways.

    Each task receives a zero-initialized bottleneck adapter and a local head.
    The adapter is a cheap private residual path; the encoder can optionally be
    updated at a lower learning rate for representation sharing.  At inference,
    routes can be selected by entropy, latent prototypes, or their calibrated
    hybrid.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        classes_per_task: int,
        num_tasks: int,
        beta: float = 0.001,
        adapter_rank: int = 0,
        encoder_learning_rate_scale: float = 1.0,
        lazy_task_paths: bool = False,
        replay_per_task: int = 0,
        replay_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, classes_per_task, num_tasks) < 1:
            raise ValueError("model dimensions must be positive")
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.beta = beta
        if adapter_rank < 0:
            raise ValueError("adapter_rank must be nonnegative")
        if encoder_learning_rate_scale < 0.0:
            raise ValueError("encoder_learning_rate_scale must be nonnegative")
        if replay_per_task < 0:
            raise ValueError("replay_per_task must be nonnegative")
        if replay_weight < 0.0:
            raise ValueError("replay_weight must be nonnegative")
        self.adapter_rank = adapter_rank
        self.encoder_learning_rate_scale = encoder_learning_rate_scale
        self.lazy_task_paths = lazy_task_paths
        self.replay_per_task = replay_per_task
        self.replay_weight = replay_weight
        self.replay_memory: list[tuple[int, Tensor, Tensor]] = []
        self.encoder = VIBLinear(input_dim, hidden_dim)
        nn.init.constant_(self.encoder.logvar_layer.weight, 0.0)
        nn.init.constant_(self.encoder.logvar_layer.bias, -4.0)
        path_count = 1 if lazy_task_paths else num_tasks
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, classes_per_task) for _ in range(path_count)]
        )
        self.adapters = nn.ModuleList()
        for _ in range(path_count):
            self.adapters.append(self._new_adapter())
        self.register_buffer(
            "route_prototypes", torch.zeros(num_tasks, classes_per_task, hidden_dim)
        )
        self.register_buffer(
            "route_prototype_valid", torch.zeros(num_tasks, classes_per_task, dtype=torch.bool)
        )
        self.register_buffer("route_scales", torch.ones(num_tasks))

    def _new_adapter(self) -> nn.Module:
        return (
            BottleneckAdapter(self.encoder.mu_layer.out_features, self.adapter_rank)
            if self.adapter_rank > 0
            else IdentityAdapter()
        )

    def allocate_task_path(self, task_id: int) -> None:
        """Lazily allocate the small private path for a new task."""

        if not self.lazy_task_paths:
            if task_id >= len(self.heads):
                raise IndexError(f"task_id {task_id} is outside allocated paths")
            return
        if task_id < len(self.heads):
            return
        if task_id != len(self.heads):
            raise ValueError("lazy task paths must be allocated sequentially")
        self.heads.append(nn.Linear(self.encoder.mu_layer.out_features, self.classes_per_task))
        self.adapters.append(self._new_adapter())

    @property
    def total_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward_task(self, task_id: int, inputs: Tensor) -> Tensor:
        if not 0 <= task_id < self.num_tasks:
            raise IndexError(f"task_id {task_id} is outside [0, {self.num_tasks})")
        if task_id >= len(self.heads):
            raise RuntimeError(f"task path {task_id} has not been allocated")
        features = self.adapters[task_id](torch.tanh(self.encoder(inputs)))
        return self.heads[task_id](features)

    def task_features(self, task_id: int, inputs: Tensor) -> Tensor:
        """Return deterministic routed features for route calibration."""

        if not 0 <= task_id < self.num_tasks:
            raise IndexError(f"task_id {task_id} is outside [0, {self.num_tasks})")
        return self.adapters[task_id](torch.tanh(self.encoder(inputs)))

    @torch.no_grad()
    def register_task_route(
        self, task_id: int, dataset: Dataset[tuple[Tensor, Tensor]]
    ) -> None:
        """Store class-conditional latent prototypes for task-ID-free routing."""

        if not 0 <= task_id < self.num_tasks:
            raise IndexError(f"task_id {task_id} is outside [0, {self.num_tasks})")
        was_training = self.training
        self.eval()
        inputs = torch.stack([dataset[index][0] for index in range(len(dataset))])
        labels = torch.stack([dataset[index][1] for index in range(len(dataset))])
        features = self.task_features(task_id, inputs)
        for local_class in range(self.classes_per_task):
            class_features = features[labels == local_class]
            if class_features.numel() == 0:
                continue
            prototype = class_features.mean(dim=0)
            self.route_prototypes[task_id, local_class].copy_(prototype)
            self.route_prototype_valid[task_id, local_class] = True
        valid = self.route_prototype_valid[task_id]
        if valid.any():
            assigned = self.route_prototypes[task_id, labels]
            self.route_scales[task_id] = (
                (features - assigned).square().mean().sqrt().clamp_min(1e-3)
            )
        if was_training:
            self.train()

    def loss_on_task(self, task_id: int, dataset: Dataset[tuple[Tensor, Tensor]]) -> Tensor:
        self.train()
        inputs = torch.stack([dataset[index][0] for index in range(len(dataset))])
        labels = torch.stack([dataset[index][1] for index in range(len(dataset))])
        logits = self.forward_task(task_id, inputs)
        return nn.functional.cross_entropy(logits, labels) + self.beta * self.encoder.kl_loss

    @torch.no_grad()
    def store_replay(
        self, task_id: int, dataset: Dataset[tuple[Tensor, Tensor]], max_examples: int | None = None
    ) -> None:
        """Store a small class-balanced input rehearsal buffer for one task."""

        limit = self.replay_per_task if max_examples is None else max_examples
        if limit <= 0:
            return
        inputs = torch.stack([dataset[index][0] for index in range(len(dataset))])
        labels = torch.stack([dataset[index][1] for index in range(len(dataset))])
        selected_indices: list[Tensor] = []
        per_class = max(1, limit // self.classes_per_task)
        for local_class in range(self.classes_per_task):
            class_indices = (labels == local_class).nonzero(as_tuple=False).flatten()
            selected_indices.append(class_indices[:per_class])
        indices = torch.cat(selected_indices)[:limit]
        self.replay_memory = [
            (old_task, old_inputs, old_labels)
            for old_task, old_inputs, old_labels in self.replay_memory
            if old_task != task_id
        ]
        self.replay_memory.append((task_id, inputs[indices].clone(), labels[indices].clone()))

    def train_task(
        self,
        task_id: int,
        dataset: Dataset[tuple[Tensor, Tensor]],
        epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.2,
        update_encoder: bool | None = None,
        encoder_learning_rate_scale: float | None = None,
        replay_weight: float | None = None,
    ) -> None:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if update_encoder is None:
            update_encoder = task_id == 0
        adapter_parameters = list(self.adapters[task_id].parameters())
        head_parameters = list(self.heads[task_id].parameters())
        if encoder_learning_rate_scale is None:
            encoder_learning_rate_scale = self.encoder_learning_rate_scale
        if encoder_learning_rate_scale < 0.0:
            raise ValueError("encoder_learning_rate_scale must be nonnegative")
        if replay_weight is None:
            replay_weight = self.replay_weight
        if replay_weight < 0.0:
            raise ValueError("replay_weight must be nonnegative")
        parameter_groups = [
            {"params": adapter_parameters + head_parameters, "lr": learning_rate}
        ]
        if update_encoder:
            parameter_groups.append(
                {
                    "params": list(self.encoder.parameters()),
                    "lr": learning_rate * encoder_learning_rate_scale,
                }
            )
        optimizer = torch.optim.SGD(parameter_groups, lr=learning_rate)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.train()
        for _ in range(epochs):
            for inputs, labels in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = self.forward_task(task_id, inputs)
                loss = nn.functional.cross_entropy(logits, labels)
                if update_encoder:
                    loss = loss + self.beta * self.encoder.kl_loss
                if replay_weight > 0.0 and self.replay_memory:
                    replay_loss = torch.zeros((), device=inputs.device)
                    for old_task, replay_inputs, replay_labels in self.replay_memory:
                        replay_logits = self.forward_task(
                            old_task, replay_inputs.to(device=inputs.device)
                        )
                        replay_loss = replay_loss + nn.functional.cross_entropy(
                            replay_logits, replay_labels.to(device=inputs.device)
                        )
                    loss = loss + replay_weight * replay_loss / len(self.replay_memory)
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def accuracy(self, task_id: int, dataset: Dataset[tuple[Tensor, Tensor]]) -> float:
        was_training = self.training
        self.eval()
        inputs = torch.stack([dataset[index][0] for index in range(len(dataset))])
        labels = torch.stack([dataset[index][1] for index in range(len(dataset))])
        predictions = self.forward_task(task_id, inputs).argmax(dim=1)
        result = (predictions == labels).float().mean().item()
        if was_training:
            self.train()
        return result

    @torch.no_grad()
    def predict_with_entropy(
        self,
        inputs: Tensor,
        route_strategy: str = "entropy",
        prototype_weight: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Select a task pathway without a task ID.

        ``entropy`` is the original selector.  ``prototype`` uses stored
        class-conditional bottleneck prototypes.  ``hybrid`` adds normalized
        predictive entropy to the prototype distance, reducing confidence
        calibration failures between independently trained task heads.
        """

        if route_strategy not in {"entropy", "prototype", "hybrid"}:
            raise ValueError("route_strategy must be 'entropy', 'prototype', or 'hybrid'")
        if prototype_weight < 0.0:
            raise ValueError("prototype_weight must be nonnegative")

        was_training = self.training
        self.eval()
        logits = torch.stack([self.forward_task(task, inputs) for task in range(self.num_tasks)], dim=1)
        probabilities = logits.softmax(dim=-1)
        entropies = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        if route_strategy == "entropy" or not self.route_prototype_valid.any():
            selected_tasks = entropies.argmin(dim=1)
        else:
            shared_features = torch.tanh(self.encoder(inputs))
            routed_features = torch.stack(
                [adapter(shared_features) for adapter in self.adapters], dim=1
            )
            distances = (routed_features.unsqueeze(2) - self.route_prototypes.unsqueeze(0)).square().mean(dim=-1)
            invalid = ~self.route_prototype_valid.unsqueeze(0)
            distances = distances.masked_fill(invalid, float("inf"))
            distances = distances.min(dim=2).values / self.route_scales.clamp_min(1e-3).unsqueeze(0)
            if route_strategy == "prototype":
                scores = distances
            else:
                normalized_entropy = entropies / torch.log(
                    torch.tensor(float(self.classes_per_task), device=entropies.device)
                ).clamp_min(1e-8)
                scores = distances + prototype_weight * normalized_entropy
            selected_tasks = scores.argmin(dim=1)
        selected_logits = logits[torch.arange(inputs.shape[0]), selected_tasks]
        local_predictions = selected_logits.argmax(dim=1)
        global_predictions = selected_tasks * self.classes_per_task + local_predictions
        if was_training:
            self.train()
        return global_predictions, selected_tasks, entropies
