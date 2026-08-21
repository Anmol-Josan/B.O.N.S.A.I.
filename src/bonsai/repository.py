"""Task repository with cached OT/TDA descriptors."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.bonsai.hierarchy import TaskHierarchy
from src.bonsai.ot import SlicedWasserstein, TaskDistributionDescriptor
from src.bonsai.tda import PersistenceDescriptor, ZeroDimensionalPersistence


@dataclass
class TaskRecord:
    """All repository-level information cached for one task."""

    task_id: int
    prototype: Tensor
    coarse_embedding: Tensor
    distribution: TaskDistributionDescriptor
    topology: PersistenceDescriptor
    sample_count: int


class TaskRepository:
    """Incrementally updated task prototypes, distributions, and topology."""

    def __init__(
        self,
        latent_dim: int,
        coarse_dim: int | None = None,
        ot_projections: int = 8,
        ot_samples: int = 32,
        tda_bins: int = 8,
        tda_samples: int = 32,
        hierarchy_branching: int = 4,
        hierarchy_leaf_capacity: int = 8,
        seed: int = 17,
    ) -> None:
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        coarse_dim = latent_dim if coarse_dim is None else coarse_dim
        if not 1 <= coarse_dim <= latent_dim:
            raise ValueError("coarse_dim must be in [1, latent_dim]")
        self.latent_dim = latent_dim
        self.coarse_dim = coarse_dim
        generator = torch.Generator().manual_seed(seed)
        projection = torch.randn(latent_dim, coarse_dim, generator=generator)
        self.coarse_projection = torch.linalg.qr(projection, mode="reduced").Q
        self.ot = SlicedWasserstein(
            latent_dim, projections=ot_projections, representative_samples=ot_samples, seed=seed
        )
        self.tda = ZeroDimensionalPersistence(bins=tda_bins, representative_samples=tda_samples)
        self.hierarchy = TaskHierarchy(
            branching=hierarchy_branching, leaf_capacity=hierarchy_leaf_capacity
        )
        self.records: dict[int, TaskRecord] = {}
        self.version = 0

    @property
    def task_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.records))

    @property
    def task_count(self) -> int:
        return len(self.records)

    def _validate_latents(self, latents: Tensor) -> Tensor:
        if latents.ndim != 2 or latents.shape[1] != self.latent_dim:
            raise ValueError(f"latents must have shape [N, {self.latent_dim}]")
        if latents.shape[0] < 1:
            raise ValueError("a task needs at least one latent representation")
        if not torch.isfinite(latents).all():
            raise ValueError("latents must be finite")
        return latents.float()

    def coarse_embedding(self, latents: Tensor) -> Tensor:
        latents = self._validate_latents(latents)
        projection = self.coarse_projection.to(device=latents.device, dtype=latents.dtype)
        return (latents @ projection).mean(dim=0)

    def add_task(self, task_id: int, latents: Tensor) -> TaskRecord:
        if task_id in self.records:
            raise KeyError(f"task {task_id} is already in the repository")
        latents = self._validate_latents(latents)
        prototype = latents.mean(dim=0)
        coarse = self.coarse_embedding(latents).detach().cpu()
        record = TaskRecord(
            task_id=task_id,
            prototype=prototype.detach().cpu(),
            coarse_embedding=coarse,
            distribution=self.ot.build(latents.detach().cpu()),
            topology=self.tda.build(latents.detach().cpu()),
            sample_count=int(latents.shape[0]),
        )
        self.records[task_id] = record
        self.hierarchy.insert(task_id, coarse)
        self.version += 1
        return record

    def update_task(self, task_id: int, latents: Tensor) -> TaskRecord:
        if task_id not in self.records:
            raise KeyError(f"task {task_id} is not in the repository")
        latents = self._validate_latents(latents)
        coarse = self.coarse_embedding(latents).detach().cpu()
        record = TaskRecord(
            task_id=task_id,
            prototype=latents.mean(dim=0).detach().cpu(),
            coarse_embedding=coarse,
            distribution=self.ot.build(latents.detach().cpu()),
            topology=self.tda.build(latents.detach().cpu()),
            sample_count=int(latents.shape[0]),
        )
        self.records[task_id] = record
        self.hierarchy.update(task_id, coarse)
        self.version += 1
        return record

    def get(self, task_id: int) -> TaskRecord:
        try:
            return self.records[task_id]
        except KeyError as error:
            raise KeyError(f"task {task_id} is not in the repository") from error

    def prototypes(self, task_ids: list[int] | tuple[int, ...] | None = None) -> Tensor:
        ids = self.task_ids if task_ids is None else tuple(task_ids)
        if not ids:
            return torch.empty((0, self.latent_dim))
        return torch.stack([self.get(task_id).prototype for task_id in ids])

    def query_coarse(self, latents: Tensor) -> Tensor:
        if latents.ndim != 2 or latents.shape[1] != self.latent_dim:
            raise ValueError(f"latents must have shape [N, {self.latent_dim}]")
        projection = self.coarse_projection.to(device=latents.device, dtype=latents.dtype)
        return latents @ projection

    def task_similarity(
        self,
        left_task_id: int,
        right_task_id: int,
        ot_temperature: float = 0.5,
        tda_temperature: float = 0.5,
        mode: str = "full",
    ) -> float:
        """Return a cached relation score used by relation-gated CL.

        ``mode`` is intentionally exposed for causal ablations.  ``full``
        uses both cached descriptors; ``no_ot`` and ``no_tda`` remove one
        descriptor; ``euclidean`` replaces both with prototype distance; and
        ``uniform`` disables relation gating altogether.
        """

        if ot_temperature <= 0.0 or tda_temperature <= 0.0:
            raise ValueError("similarity temperatures must be positive")
        if mode not in {"full", "no_ot", "no_tda", "euclidean", "uniform"}:
            raise ValueError(f"unknown task-similarity mode: {mode}")
        left = self.get(left_task_id)
        right = self.get(right_task_id)
        if left_task_id == right_task_id:
            return 1.0
        if mode == "uniform":
            return 1.0
        if mode == "euclidean":
            prototype_distance = float((left.prototype - right.prototype).norm())
            return float(torch.exp(torch.tensor(-prototype_distance / ot_temperature)))
        ot_distance = (
            float(self.ot.distance(left.distribution, right.distribution).detach())
            if mode != "no_ot"
            else 0.0
        )
        tda_distance = (
            float(self.tda.distance(left.topology, right.topology).detach())
            if mode != "no_tda"
            else 0.0
        )
        return float(
            torch.exp(
                torch.tensor(
                    -ot_distance / ot_temperature - tda_distance / tda_temperature
                )
            )
        )

    def state_dict(self) -> dict[str, Any]:
        """Return a tensor-only serialization payload."""

        return {
            "latent_dim": self.latent_dim,
            "coarse_dim": self.coarse_dim,
            "coarse_projection": self.coarse_projection,
            "ot_projections": self.ot.projections,
            "ot_samples": self.ot.representative_samples,
            "ot_directions": self.ot.directions,
            "tda_bins": self.tda.bins,
            "tda_samples": self.tda.representative_samples,
            "hierarchy_branching": self.hierarchy.branching,
            "hierarchy_leaf_capacity": self.hierarchy.leaf_capacity,
            "records": self.records,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "TaskRepository":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        repository = cls(
            latent_dim=int(payload["latent_dim"]),
            coarse_dim=int(payload["coarse_dim"]),
            ot_projections=int(payload["ot_projections"]),
            ot_samples=int(payload["ot_samples"]),
            tda_bins=int(payload["tda_bins"]),
            tda_samples=int(payload["tda_samples"]),
            hierarchy_branching=int(payload["hierarchy_branching"]),
            hierarchy_leaf_capacity=int(payload["hierarchy_leaf_capacity"]),
        )
        repository.coarse_projection = payload["coarse_projection"]
        repository.ot.directions = payload["ot_directions"]
        repository.records = payload["records"]
        repository.hierarchy = TaskHierarchy(
            branching=int(payload["hierarchy_branching"]),
            leaf_capacity=int(payload["hierarchy_leaf_capacity"]),
        )
        for task_id in sorted(repository.records):
            repository.hierarchy.insert(task_id, repository.records[task_id].coarse_embedding)
        repository.version = len(repository.records)
        return repository
