"""Composition of the modular BONSAI architecture."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from src.bonsai.geometry import LowRankRiemannianMetric
from src.bonsai.model import BONSAIModel, BONSAIModelOutput
from src.bonsai.repository import TaskRepository
from src.bonsai.router import RouteResult, TaskRouter
from src.bonsai.sheaf import SparseTaskSheaf


class BONSAISystem(nn.Module):
    """VIB -> repository retrieval -> geometry/sheaf -> shared adapter model."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        vib_beta: float = 1e-3,
        adapter_rank: int = 2,
        metric_rank: int = 2,
        sheaf_stalk_dim: int = 4,
        repository_kwargs: dict | None = None,
        router_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.model = BONSAIModel(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            vib_beta=vib_beta,
            adapter_rank=adapter_rank,
        )
        self.metric = LowRankRiemannianMetric(latent_dim=latent_dim, rank=metric_rank)
        self.sheaf = SparseTaskSheaf(latent_dim=latent_dim, stalk_dim=sheaf_stalk_dim)
        self.repository = TaskRepository(latent_dim=latent_dim, **(repository_kwargs or {}))
        self.router = TaskRouter(
            self.repository, self.metric, self.sheaf, **(router_kwargs or {})
        )
        self.initial_parameter_count = self.total_parameters

    @property
    def total_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def parameter_overhead(self) -> float:
        return (self.total_parameters - self.initial_parameter_count) / self.initial_parameter_count

    @property
    def task_ids(self) -> tuple[int, ...]:
        return self.repository.task_ids

    def add_task(self, task_id: int, latent_samples: Tensor | None = None) -> None:
        self.model.add_task(task_id)
        if latent_samples is not None:
            self.router.add_task(task_id, latent_samples)

    def register_task(self, task_id: int, latent_samples: Tensor) -> None:
        if task_id not in self.model.adapter.task_ids:
            self.model.add_task(task_id)
        if task_id in self.repository.records:
            self.router.update_task(task_id, latent_samples)
        else:
            self.router.add_task(task_id, latent_samples)

    def forward(
        self, inputs: Tensor, task_id: int | None = None, sample: bool | None = None
    ) -> BONSAIModelOutput:
        return self.model(inputs, task_id=task_id, sample=sample)

    @torch.no_grad()
    def route(self, inputs: Tensor, context: Tensor | None = None) -> RouteResult:
        was_training = self.model.training
        self.model.eval()
        latents = self.model.deterministic_features(inputs)
        context_latents = None
        if context is not None:
            context_latents = self.model.deterministic_features(context)
        result = self.router.route(latents, context=context_latents)
        self.model.train(was_training)
        return result

    def save_repository(self, path: str | Path) -> None:
        self.repository.save(path)
