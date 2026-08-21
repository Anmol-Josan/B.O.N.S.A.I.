"""Stable local Riemannian metric and tangent-space routing losses."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class LowRankRiemannianMetric(nn.Module):
    """Shared diagonal metric with bounded positive low-rank task updates.

    The metric is

    ``g_k = diag(g_0) + U diag(delta_k) U^T``.

    ``g_0`` is bounded elementwise and ``delta_k`` is nonnegative, so every
    metric is positive definite. This is intentionally a local tangent-space
    approximation: ``Log_p(z)`` is represented by ``z - p`` in the learned
    Euclidean coordinate chart, avoiding unsupported global geodesic claims.
    """

    def __init__(
        self,
        latent_dim: int,
        rank: int = 2,
        min_eigenvalue: float = 0.25,
        max_base_eigenvalue: float = 2.0,
        max_update: float = 0.5,
    ) -> None:
        super().__init__()
        if min(latent_dim, rank) < 1:
            raise ValueError("latent_dim and rank must be positive")
        if rank > latent_dim:
            raise ValueError("rank cannot exceed latent_dim")
        if min_eigenvalue <= 0.0 or max_base_eigenvalue <= min_eigenvalue:
            raise ValueError("metric eigenvalue bounds are invalid")
        if max_update < 0.0:
            raise ValueError("max_update must be nonnegative")
        self.latent_dim = latent_dim
        self.rank = rank
        self.min_eigenvalue = min_eigenvalue
        self.max_base_eigenvalue = max_base_eigenvalue
        self.max_update = max_update
        initial_fraction = 0.5
        initial_logit = math.log(initial_fraction / (1.0 - initial_fraction))
        self.base_raw = nn.Parameter(torch.full((latent_dim,), initial_logit))
        basis = torch.randn(latent_dim, rank) / latent_dim**0.5
        self.shared_basis = nn.Parameter(basis)
        self.task_coefficients = nn.ParameterDict()

    @property
    def task_ids(self) -> tuple[int, ...]:
        return tuple(sorted(int(key) for key in self.task_coefficients.keys()))

    def _key(self, task_id: int) -> str:
        return str(int(task_id))

    def add_task(self, task_id: int, initial_coefficients: Tensor | None = None) -> None:
        key = self._key(task_id)
        if key in self.task_coefficients:
            raise KeyError(f"metric for task {task_id} already exists")
        if initial_coefficients is None:
            coefficients = torch.full((self.rank,), -4.0)
        else:
            if initial_coefficients.shape != (self.rank,):
                raise ValueError("initial_coefficients has the wrong shape")
            coefficients = initial_coefficients.detach().float().clone()
        self.task_coefficients[key] = nn.Parameter(coefficients)

    def _base_diagonal(self) -> Tensor:
        return self.min_eigenvalue + (
            self.max_base_eigenvalue - self.min_eigenvalue
        ) * torch.sigmoid(self.base_raw)

    def _normalized_basis(self) -> Tensor:
        return self.shared_basis / self.shared_basis.norm(dim=0, keepdim=True).clamp_min(1e-8)

    def matrix(self, task_id: int) -> Tensor:
        key = self._key(task_id)
        if key not in self.task_coefficients:
            raise KeyError(f"metric for task {task_id} is not registered")
        basis = self._normalized_basis()
        updates = self.max_update * torch.sigmoid(self.task_coefficients[key])
        return torch.diag(self._base_diagonal()) + (basis * updates.unsqueeze(0)) @ basis.T

    def log_map(self, point: Tensor, prototype: Tensor) -> Tensor:
        """Local chart approximation to ``Log_prototype(point)``."""

        if point.shape[-1] != self.latent_dim or prototype.shape[-1] != self.latent_dim:
            raise ValueError("point and prototype have incompatible dimensions")
        return point - prototype

    def distance_squared(self, point: Tensor, prototype: Tensor, task_id: int) -> Tensor:
        tangent = self.log_map(point, prototype)
        metric = self.matrix(task_id).to(device=tangent.device, dtype=tangent.dtype)
        return torch.einsum("...d,de,...e->...", tangent, metric, tangent).clamp_min(0.0)

    def distance(self, point: Tensor, prototype: Tensor, task_id: int) -> Tensor:
        return self.distance_squared(point, prototype, task_id).sqrt()

    def condition_number(self, task_id: int) -> Tensor:
        eigenvalues = torch.linalg.eigvalsh(self.matrix(task_id))
        return eigenvalues[-1] / eigenvalues[0].clamp_min(1e-8)

    def all_condition_numbers(self) -> dict[int, float]:
        return {task_id: float(self.condition_number(task_id).detach()) for task_id in self.task_ids}

    def geometry_loss(
        self,
        points: Tensor,
        task_id: int,
        prototype: Tensor,
        other_prototypes: Tensor,
        margin: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return within-task, inter-task margin, and combined geometry loss."""

        if margin < 0.0:
            raise ValueError("margin must be nonnegative")
        within = self.distance_squared(points, prototype, task_id).mean()
        if other_prototypes.numel() == 0:
            separation = points.new_zeros(())
        else:
            prototype_distances = self.distance(prototype, other_prototypes, task_id)
            separation = torch.relu(margin - prototype_distances).square().mean()
        return within, separation, within + separation
