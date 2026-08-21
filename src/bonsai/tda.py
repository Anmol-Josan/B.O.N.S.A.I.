"""Repository-level persistence descriptors.

This implementation computes exact zero-dimensional Vietoris--Rips
persistence for a subsampled point cloud: finite death times are the edge
lengths of the Euclidean minimum spanning tree. It is a genuine persistence
invariant and avoids repeatedly building expensive higher-dimensional
complexes on the local machine. The limitation is explicit: it does not claim
to capture H1/H2 holes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PersistenceDescriptor:
    """Compact H0 persistence summary for one task."""

    deaths: Tensor
    histogram: Tensor
    statistics: Tensor
    sample_count: int

    @property
    def vector(self) -> Tensor:
        return torch.cat((self.histogram, self.statistics))


class ZeroDimensionalPersistence:
    """Build and compare cached H0 persistence summaries."""

    def __init__(self, bins: int = 8, representative_samples: int = 32) -> None:
        if min(bins, representative_samples) < 1:
            raise ValueError("TDA dimensions must be positive")
        self.bins = bins
        self.representative_samples = representative_samples

    def _validate(self, points: Tensor) -> Tensor:
        if points.ndim != 2 or points.shape[0] < 1:
            raise ValueError("points must have shape [N, D] with N >= 1")
        if not torch.isfinite(points).all():
            raise ValueError("points must be finite")
        return points.float()

    def _representative_points(self, points: Tensor) -> Tensor:
        count = min(self.representative_samples, points.shape[0])
        if count == points.shape[0]:
            return points
        indices = torch.linspace(0, points.shape[0] - 1, count, device=points.device)
        return points[indices.round().long()]

    @staticmethod
    def _mst_death_times(points: Tensor) -> Tensor:
        count = points.shape[0]
        if count == 1:
            return points.new_zeros(0)
        distances = torch.cdist(points, points)
        used = torch.zeros(count, dtype=torch.bool, device=points.device)
        best = torch.full((count,), float("inf"), dtype=points.dtype, device=points.device)
        best[0] = 0.0
        deaths: list[Tensor] = []
        for _ in range(count):
            masked = best.masked_fill(used, float("inf"))
            index = int(masked.argmin().item())
            used[index] = True
            if index != 0:
                deaths.append(best[index])
            best = torch.minimum(best, distances[index])
        return torch.stack(deaths) if deaths else points.new_zeros(0)

    def build(self, points: Tensor) -> PersistenceDescriptor:
        points = self._validate(points)
        representatives = self._representative_points(points)
        deaths = self._mst_death_times(representatives).sort().values
        maximum = deaths.max() if deaths.numel() else points.new_tensor(0.0)
        scale = maximum.clamp_min(1e-6)
        if deaths.numel():
            histogram = torch.histc(deaths, bins=self.bins, min=0.0, max=float(scale.item()))
            histogram = histogram / histogram.sum().clamp_min(1.0)
            quantiles = torch.quantile(
                deaths, torch.tensor([0.25, 0.5, 0.75], device=deaths.device)
            )
            statistics = torch.stack(
                (
                    deaths.mean(),
                    deaths.std(unbiased=False),
                    maximum,
                    deaths.sum(),
                    quantiles[0],
                    quantiles[1],
                    quantiles[2],
                )
            )
        else:
            histogram = points.new_zeros(self.bins)
            statistics = points.new_zeros(7)
        return PersistenceDescriptor(
            deaths=deaths,
            histogram=histogram,
            statistics=statistics,
            sample_count=int(points.shape[0]),
        )

    def distance(self, left: PersistenceDescriptor, right: PersistenceDescriptor) -> Tensor:
        if left.histogram.shape != right.histogram.shape:
            raise ValueError("TDA descriptors have incompatible histogram sizes")
        return (left.histogram - right.histogram).abs().mean() + 0.25 * (
            left.statistics - right.statistics
        ).abs().mean()
