"""Fast toy continual-learning loop used to validate BONSAI invariants."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.models.vib_layers import VIBLinear


class ToyContinualLearner(nn.Module):
    """Shared VIB encoder with one small classifier pathway per task.

    Task 0 trains the encoder and its head. Later tasks train only their newly
    allocated head in the toy protocol, which isolates retention assertions and
    mirrors a localized subgraph update.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        classes_per_task: int,
        num_tasks: int,
        beta: float = 0.001,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, classes_per_task, num_tasks) < 1:
            raise ValueError("model dimensions must be positive")
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.beta = beta
        self.encoder = VIBLinear(input_dim, hidden_dim)
        nn.init.constant_(self.encoder.logvar_layer.weight, 0.0)
        nn.init.constant_(self.encoder.logvar_layer.bias, -4.0)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, classes_per_task) for _ in range(num_tasks)]
        )

    @property
    def total_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward_task(self, task_id: int, inputs: Tensor) -> Tensor:
        if not 0 <= task_id < self.num_tasks:
            raise IndexError(f"task_id {task_id} is outside [0, {self.num_tasks})")
        features = torch.tanh(self.encoder(inputs))
        return self.heads[task_id](features)

    def loss_on_task(self, task_id: int, dataset: Dataset[tuple[Tensor, Tensor]]) -> Tensor:
        self.train()
        inputs = torch.stack([dataset[index][0] for index in range(len(dataset))])
        labels = torch.stack([dataset[index][1] for index in range(len(dataset))])
        logits = self.forward_task(task_id, inputs)
        return nn.functional.cross_entropy(logits, labels) + self.beta * self.encoder.kl_loss

    def train_task(
        self,
        task_id: int,
        dataset: Dataset[tuple[Tensor, Tensor]],
        epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.2,
        update_encoder: bool | None = None,
    ) -> None:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if update_encoder is None:
            update_encoder = task_id == 0
        if update_encoder:
            parameters = list(self.encoder.parameters()) + list(self.heads[task_id].parameters())
        else:
            parameters = list(self.heads[task_id].parameters())
        optimizer = torch.optim.SGD(parameters, lr=learning_rate)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.train()
        for _ in range(epochs):
            for inputs, labels in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = self.forward_task(task_id, inputs)
                loss = nn.functional.cross_entropy(logits, labels)
                if update_encoder:
                    loss = loss + self.beta * self.encoder.kl_loss
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
    def predict_with_entropy(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Select the lowest-entropy task pathway without a task ID."""

        was_training = self.training
        self.eval()
        logits = torch.stack([self.forward_task(task, inputs) for task in range(self.num_tasks)], dim=1)
        probabilities = logits.softmax(dim=-1)
        entropies = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        selected_tasks = entropies.argmin(dim=1)
        selected_logits = logits[torch.arange(inputs.shape[0]), selected_tasks]
        local_predictions = selected_logits.argmax(dim=1)
        global_predictions = selected_tasks * self.classes_per_task + local_predictions
        if was_training:
            self.train()
        return global_predictions, selected_tasks, entropies
