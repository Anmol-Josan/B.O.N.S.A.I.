"""Cached sliced-Wasserstein task distribution descriptors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class TaskDistributionDescriptor:
    """Compact empirical distribution descriptor."""

    quantiles: Tensor
    prototype: Tensor
    sample_count: int

    @property
    def projection_count(self) -> int:
        return int(self.quantiles.shape[0])

    @property
    def representative_count(self) -> int:
        return int(self.quantiles.shape[1])


class SlicedWasserstein:
    """CPU-friendly sliced-Wasserstein-1 scorer with fixed projections."""

    def __init__(
        self,
        latent_dim: int,
        projections: int = 8,
        representative_samples: int = 32,
        seed: int = 17,
    ) -> None:
        if min(latent_dim, projections, representative_samples) < 1:
            raise ValueError("OT dimensions must be positive")
        self.latent_dim = latent_dim
        self.projections = projections
        self.representative_samples = representative_samples
        generator = torch.Generator().manual_seed(seed)
        directions = torch.randn(projections, latent_dim, generator=generator)
        self.directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def _validate(self, points: Tensor) -> Tensor:
        if points.ndim != 2 or points.shape[1] != self.latent_dim:
            raise ValueError(f"points must have shape [N, {self.latent_dim}]")
        if points.shape[0] < 1:
            raise ValueError("at least one point is required")
        if not torch.isfinite(points).all():
            raise ValueError("points must be finite")
        return points.float()

    def _representative_points(self, points: Tensor) -> Tensor:
        count = min(self.representative_samples, points.shape[0])
        if count == points.shape[0]:
            return points
        indices = torch.linspace(0, points.shape[0] - 1, count, device=points.device)
        return points[indices.round().long()]

    def build(self, points: Tensor) -> TaskDistributionDescriptor:
        points = self._validate(points)
        representatives = self._representative_points(points)
        directions = self.directions.to(device=points.device, dtype=points.dtype)
        quantiles = torch.sort(representatives @ directions.T, dim=0).values.T
        return TaskDistributionDescriptor(
            quantiles=quantiles, prototype=points.mean(dim=0), sample_count=int(points.shape[0])
        )

    @staticmethod
    def _resample_quantiles(values: Tensor, count: int) -> Tensor:
        if values.shape[1] == count:
            return values
        return F.interpolate(
            values.unsqueeze(0), size=count, mode="linear", align_corners=True
        ).squeeze(0)

    def distance(
        self, left: TaskDistributionDescriptor, right: TaskDistributionDescriptor
    ) -> Tensor:
        if left.projection_count != self.projections or right.projection_count != self.projections:
            raise ValueError("descriptor projection count does not match this scorer")
        count = max(left.representative_count, right.representative_count)
        left_values = self._resample_quantiles(left.quantiles, count)
        right_values = self._resample_quantiles(right.quantiles, count)
        return (left_values - right_values).abs().mean()

    def prototype_distance(self, points: Tensor, descriptor: TaskDistributionDescriptor) -> Tensor:
        """Single-example-safe fallback: use prototype distance only."""

        points = self._validate(points)
        if points.shape[0] != 1:
            raise ValueError("prototype_distance expects exactly one query point")
        return (points[0] - descriptor.prototype).norm()
