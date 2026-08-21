"""Incrementally maintained balanced task hierarchy."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class _HierarchyNode:
    """Internal tree node; leaves own task ids, internal nodes own children."""

    center: Tensor
    task_ids: list[int] = field(default_factory=list)
    children: list["_HierarchyNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children


class TaskHierarchy:
    """A small balanced tree with local split-on-overflow insertion.

    Internal routing uses a beam over child centers. Only task ids in the
    visited leaves are compared with the query, so the implementation measures
    actual candidate reduction instead of claiming logarithmic total work.
    """

    def __init__(self, branching: int = 4, leaf_capacity: int = 8) -> None:
        if min(branching, leaf_capacity) < 2:
            raise ValueError("branching and leaf_capacity must be at least 2")
        self.branching = branching
        self.leaf_capacity = leaf_capacity
        self.embeddings: dict[int, Tensor] = {}
        self.root: _HierarchyNode | None = None
        self.version = 0

    @property
    def task_count(self) -> int:
        return len(self.embeddings)

    @property
    def depth(self) -> int:
        def walk(node: _HierarchyNode | None) -> int:
            if node is None:
                return 0
            return 1 if node.is_leaf else 1 + max(walk(child) for child in node.children)

        return walk(self.root)

    @property
    def node_count(self) -> int:
        def walk(node: _HierarchyNode | None) -> int:
            if node is None:
                return 0
            return 1 + sum(walk(child) for child in node.children)

        return walk(self.root)

    def _center_for_ids(self, task_ids: list[int]) -> Tensor:
        return torch.stack([self.embeddings[task_id] for task_id in task_ids]).mean(dim=0)

    def _split_task_ids(self, task_ids: list[int]) -> list[list[int]]:
        if len(task_ids) <= self.leaf_capacity:
            return [task_ids]
        values = torch.stack([self.embeddings[task_id] for task_id in task_ids])
        axis = int(values.var(dim=0, unbiased=False).argmax().item())
        ordered = [task_id for _, task_id in sorted(zip(values[:, axis].tolist(), task_ids))]
        groups = min(self.branching, max(2, (len(ordered) + self.leaf_capacity - 1) // self.leaf_capacity))
        return [
            [int(value) for value in group.tolist()]
            for group in torch.tensor_split(torch.tensor(ordered), groups)
            if len(group)
        ]

    def _leaf(self, task_ids: list[int]) -> _HierarchyNode:
        return _HierarchyNode(center=self._center_for_ids(task_ids), task_ids=list(task_ids))

    def _make_balanced(self, task_ids: list[int]) -> _HierarchyNode:
        groups = self._split_task_ids(task_ids)
        if len(groups) == 1:
            return self._leaf(groups[0])
        children = [self._make_balanced(group) for group in groups]
        return _HierarchyNode(
            center=torch.stack([child.center for child in children]).mean(dim=0), children=children
        )

    def _split_internal(self, children: list[_HierarchyNode]) -> _HierarchyNode:
        values = torch.stack([child.center for child in children])
        axis = int(values.var(dim=0, unbiased=False).argmax().item())
        ordered = [child for _, child in sorted(zip(values[:, axis].tolist(), children), key=lambda item: item[0])]
        groups = min(self.branching, max(2, len(ordered) // 2))
        chunks = [list(chunk) for chunk in torch.tensor_split(torch.arange(len(ordered)), groups) if len(chunk)]
        grouped_children = [[ordered[int(index)] for index in chunk.tolist()] for chunk in chunks]
        return _HierarchyNode(
            center=torch.stack([child.center for child in children]).mean(dim=0),
            children=[
                _HierarchyNode(
                    center=torch.stack([child.center for child in group]).mean(dim=0),
                    children=group,
                )
                for group in grouped_children
            ],
        )

    def _insert_node(self, node: _HierarchyNode, task_id: int) -> _HierarchyNode:
        if node.is_leaf:
            node.task_ids.append(task_id)
            node.center = self._center_for_ids(node.task_ids)
            if len(node.task_ids) <= self.leaf_capacity:
                return node
            return self._make_balanced(node.task_ids)
        distances = torch.stack(
            [(child.center - self.embeddings[task_id]).square().sum() for child in node.children]
        )
        child_index = int(distances.argmin().item())
        node.children[child_index] = self._insert_node(node.children[child_index], task_id)
        node.center = torch.stack([child.center for child in node.children]).mean(dim=0)
        if len(node.children) > self.branching:
            return self._split_internal(node.children)
        return node

    def insert(self, task_id: int, embedding: Tensor) -> None:
        """Insert one task and split only the overflowing local path."""

        if embedding.ndim != 1 or embedding.numel() < 1:
            raise ValueError("embedding must be a non-empty vector")
        if not torch.isfinite(embedding).all():
            raise ValueError("embedding must be finite")
        if task_id in self.embeddings:
            raise KeyError(f"task {task_id} is already in the hierarchy")
        self.embeddings[task_id] = embedding.detach().float().cpu()
        self.root = self._leaf([task_id]) if self.root is None else self._insert_node(self.root, task_id)
        self.version += 1

    def update(self, task_id: int, embedding: Tensor) -> None:
        if task_id not in self.embeddings:
            raise KeyError(f"task {task_id} is not in the hierarchy")
        if embedding.ndim != 1 or not torch.isfinite(embedding).all():
            raise ValueError("embedding must be a finite vector")
        self.embeddings[task_id] = embedding.detach().float().cpu()
        self.root = self._make_balanced(sorted(self.embeddings))
        self.version += 1

    def retrieve(
        self, query: Tensor, top_k: int = 4, beam_width: int = 2
    ) -> tuple[list[int], int]:
        """Return top-k task candidates and the number of task comparisons."""

        if self.root is None:
            return [], 0
        if query.ndim != 1 or query.shape[0] != self.root.center.shape[0]:
            raise ValueError("query has incompatible coarse-embedding dimension")
        if min(top_k, beam_width) < 1:
            raise ValueError("top_k and beam_width must be positive")
        frontier = [self.root]
        while frontier and not frontier[0].is_leaf:
            next_frontier: list[_HierarchyNode] = []
            for node in frontier:
                distances = [(child.center - query).square().sum().item() for child in node.children]
                next_frontier.extend(
                    child
                    for _, child in sorted(zip(distances, node.children), key=lambda item: item[0])[:beam_width]
                )
            frontier = next_frontier
        candidate_ids = sorted({task_id for node in frontier for task_id in node.task_ids})
        candidate_ids.sort(key=lambda task_id: float((self.embeddings[task_id] - query).square().sum()))
        return candidate_ids[: min(top_k, len(candidate_ids))], len(candidate_ids)
