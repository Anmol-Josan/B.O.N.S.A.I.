"""Sparse task graph and sheaf-style compatibility energy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SheafEdge:
    left: int
    right: int


class SparseTaskSheaf(nn.Module):
    """Small vector-space stalks with endpoint restriction maps.

    New tasks connect to only their nearest existing tasks in the coarse task
    space. Adding a task therefore creates a bounded number of maps instead of
    rebuilding a dense all-task compatibility matrix.
    """

    def __init__(self, latent_dim: int, stalk_dim: int = 4, max_edges_per_task: int = 3) -> None:
        super().__init__()
        if min(latent_dim, stalk_dim, max_edges_per_task) < 1:
            raise ValueError("sheaf dimensions must be positive")
        self.latent_dim = latent_dim
        self.stalk_dim = stalk_dim
        self.max_edges_per_task = max_edges_per_task
        self.shared_restriction = nn.Parameter(torch.randn(stalk_dim, latent_dim) / latent_dim**0.5)
        self.edges: list[SheafEdge] = []
        # A shared stalk map is reused on every edge. Each sparse edge keeps
        # only two scalar gates, allowing mild endpoint-specific compatibility
        # without allocating a dense map for every pair of tasks.
        self.left_scales = nn.ParameterDict()
        self.right_scales = nn.ParameterDict()

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def _key(self, left: int, right: int) -> str:
        return f"{min(left, right)}__{max(left, right)}"

    def add_edge(self, left: int, right: int) -> SheafEdge:
        if left == right:
            raise ValueError("self-edges are not meaningful for compatibility")
        edge = SheafEdge(min(int(left), int(right)), max(int(left), int(right)))
        key = self._key(edge.left, edge.right)
        if edge in self.edges:
            return edge
        self.edges.append(edge)
        self.left_scales[key] = nn.Parameter(torch.ones(1))
        self.right_scales[key] = nn.Parameter(torch.ones(1))
        return edge

    def add_task(
        self,
        task_id: int,
        embedding: Tensor,
        existing_embeddings: dict[int, Tensor],
        max_edges: int | None = None,
    ) -> list[SheafEdge]:
        """Connect a new task to nearest existing task embeddings."""

        if embedding.ndim != 1 or not torch.isfinite(embedding).all():
            raise ValueError("task embedding must be a finite vector")
        limit = self.max_edges_per_task if max_edges is None else max_edges
        if limit < 1:
            return []
        distances = sorted(
            (
                float((other_embedding - embedding).square().sum()),
                int(other_id),
            )
            for other_id, other_embedding in existing_embeddings.items()
            if int(other_id) != int(task_id)
        )
        return [self.add_edge(task_id, other_id) for _, other_id in distances[:limit]]

    def _maps_for(self, edge: SheafEdge) -> tuple[Tensor, Tensor]:
        key = self._key(edge.left, edge.right)
        shared = self.shared_restriction
        return self.left_scales[key] * shared, self.right_scales[key] * shared

    def energy(self, node_values: dict[int, Tensor]) -> Tensor:
        """Compute sum of squared restriction disagreement over present edges."""

        if not self.edges:
            device = next(self.parameters()).device
            return torch.zeros((), device=device)
        terms: list[Tensor] = []
        for edge in self.edges:
            if edge.left not in node_values or edge.right not in node_values:
                continue
            left_map, right_map = self._maps_for(edge)
            left = node_values[edge.left]
            right = node_values[edge.right]
            if left.shape[-1] != self.latent_dim or right.shape[-1] != self.latent_dim:
                raise ValueError("node value has incompatible latent dimension")
            difference = left @ left_map.T - right @ right_map.T
            terms.append(difference.square().mean())
        if not terms:
            return next(self.parameters()).new_zeros(())
        return torch.stack(terms).mean()

    def local_compatibility(
        self, task_id: int, query: Tensor, prototypes: dict[int, Tensor]
    ) -> Tensor:
        """Compatibility score of a query with a task's neighboring stalks."""

        terms: list[Tensor] = []
        for edge in self.edges:
            if task_id not in (edge.left, edge.right):
                continue
            neighbor = edge.right if edge.left == task_id else edge.left
            if neighbor not in prototypes:
                continue
            left_map, right_map = self._maps_for(edge)
            if edge.left == task_id:
                difference = query @ left_map.T - prototypes[neighbor] @ right_map.T
            else:
                difference = prototypes[neighbor] @ left_map.T - query @ right_map.T
            terms.append(difference.square().sum(dim=-1))
        if not terms:
            return query.new_zeros(query.shape[:-1])
        return torch.stack(terms).mean(dim=0)
