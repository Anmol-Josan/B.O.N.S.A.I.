"""Hierarchical retrieval, OT/TDA scoring, and local Riemannian routing."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
from torch import Tensor

from src.bonsai.geometry import LowRankRiemannianMetric
from src.bonsai.repository import TaskRepository
from src.bonsai.sheaf import SparseTaskSheaf


@dataclass
class RouteResult:
    selected_task_ids: Tensor
    candidate_task_ids: list[int]
    candidate_comparisons: int
    hierarchy_depth: int
    retrieval_latency_ms: float
    total_latency_ms: float
    scores: Tensor
    geometric_scores: Tensor


class TaskRouter:
    """Route query latents through the requested pipeline."""

    def __init__(
        self,
        repository: TaskRepository,
        metric: LowRankRiemannianMetric,
        sheaf: SparseTaskSheaf,
        top_k: int = 4,
        beam_width: int = 2,
        ot_weight: float = 0.2,
        tda_weight: float = 0.1,
        sheaf_weight: float = 0.05,
        minimum_episode_size: int = 4,
    ) -> None:
        if min(top_k, beam_width, minimum_episode_size) < 1:
            raise ValueError("routing limits must be positive")
        if min(ot_weight, tda_weight, sheaf_weight) < 0.0:
            raise ValueError("routing weights must be nonnegative")
        self.repository = repository
        self.metric = metric
        self.sheaf = sheaf
        self.top_k = top_k
        self.beam_width = beam_width
        self.ot_weight = ot_weight
        self.tda_weight = tda_weight
        self.sheaf_weight = sheaf_weight
        self.minimum_episode_size = minimum_episode_size

    def add_task(self, task_id: int, latents: Tensor) -> None:
        """Register a task across repository, metric, and sparse sheaf."""

        existing = {
            other_id: record.coarse_embedding
            for other_id, record in self.repository.records.items()
        }
        self.repository.add_task(task_id, latents)
        self.metric.add_task(task_id)
        self.sheaf.add_task(
            task_id,
            self.repository.get(task_id).coarse_embedding,
            existing,
        )

    def update_task(self, task_id: int, latents: Tensor) -> None:
        self.repository.update_task(task_id, latents)

    def _query_coarse(self, latents: Tensor) -> Tensor:
        return self.repository.query_coarse(latents)

    def _geometric_scores(self, latents: Tensor, candidates: list[int]) -> Tensor:
        prototypes = self.repository.prototypes(candidates).to(latents.device, latents.dtype)
        values = []
        for index, task_id in enumerate(candidates):
            values.append(self.metric.distance_squared(latents, prototypes[index], task_id))
        return torch.stack(values, dim=1)

    def route(self, latents: Tensor, context: Tensor | None = None) -> RouteResult:
        """Return argmin local metric over hierarchy-selected candidates.

        OT/TDA are used only when ``context`` contains a genuine episode. For
        a single query, the geometric prototype score is the primary signal.
        """

        if latents.ndim != 2 or latents.shape[1] != self.repository.latent_dim:
            raise ValueError(f"latents must have shape [N, {self.repository.latent_dim}]")
        if not torch.isfinite(latents).all():
            raise ValueError("latents must be finite")
        start = perf_counter()
        if self.repository.task_count == 0:
            empty = torch.full((latents.shape[0],), -1, dtype=torch.long, device=latents.device)
            zero = latents.new_zeros((latents.shape[0], 0))
            return RouteResult(empty, [], 0, 0, 0.0, (perf_counter() - start) * 1000, zero, zero)
        coarse_queries = self._query_coarse(latents).detach().cpu()
        retrieval_start = perf_counter()
        candidate_lists: list[list[int]] = []
        comparisons = 0
        for coarse_query in coarse_queries:
            candidates, evaluated = self.repository.hierarchy.retrieve(
                coarse_query,
                top_k=min(self.top_k, self.repository.task_count),
                beam_width=self.beam_width,
            )
            if not candidates:
                candidates = list(self.repository.task_ids)
                evaluated = len(candidates)
            candidate_lists.append(candidates)
            comparisons += evaluated
        retrieval_ms = (perf_counter() - retrieval_start) * 1000
        candidates = sorted({task_id for task_list in candidate_lists for task_id in task_list})
        candidate_indices = {task_id: index for index, task_id in enumerate(candidates)}
        infinite = torch.tensor(float("inf"), device=latents.device, dtype=latents.dtype)
        geometric = infinite.expand(latents.shape[0], len(candidates)).clone()
        for row, row_candidates in enumerate(candidate_lists):
            row_scores = self._geometric_scores(latents[row : row + 1], row_candidates).squeeze(0)
            indices = torch.tensor(
                [candidate_indices[task_id] for task_id in row_candidates],
                device=latents.device,
                dtype=torch.long,
            )
            geometric[row, indices] = row_scores
        scores = geometric.clone()
        use_episode = context is not None and context.ndim == 2 and context.shape[0] >= self.minimum_episode_size
        if use_episode:
            query_distribution = self.repository.ot.build(context)
            query_topology = self.repository.tda.build(context)
            ot_scores = torch.stack(
                [self.repository.ot.distance(query_distribution, self.repository.get(task_id).distribution) for task_id in candidates]
            ).to(latents.device, latents.dtype)
            tda_scores = torch.stack(
                [self.repository.tda.distance(query_topology, self.repository.get(task_id).topology) for task_id in candidates]
            ).to(latents.device, latents.dtype)
            episode_scores = self.ot_weight * ot_scores + self.tda_weight * tda_scores
            finite_mask = torch.isfinite(scores)
            scores = scores + episode_scores.unsqueeze(0).masked_fill(~finite_mask, 0.0)
        prototypes = {
            task_id: self.repository.get(task_id).prototype.to(latents.device, latents.dtype)
            for task_id in self.repository.task_ids
        }
        compatibility = torch.stack(
            [self.sheaf.local_compatibility(task_id, latents, prototypes) for task_id in candidates], dim=1
        )
        scores = scores + self.sheaf_weight * compatibility.masked_fill(~torch.isfinite(scores), 0.0)
        selected_index = scores.argmin(dim=1)
        selected = torch.tensor(candidates, device=latents.device, dtype=torch.long)[selected_index]
        return RouteResult(
            selected_task_ids=selected,
            candidate_task_ids=candidates,
            candidate_comparisons=comparisons,
            hierarchy_depth=self.repository.hierarchy.depth,
            retrieval_latency_ms=retrieval_ms,
            total_latency_ms=(perf_counter() - start) * 1000,
            scores=scores,
            geometric_scores=geometric,
        )
