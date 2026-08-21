"""Shared-basis low-rank task adapters."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SharedLowRankAdapter(nn.Module):
    """Apply ``x + U diag(d_k) V^T x`` with shared U/V bases.

    Only the rank-sized coefficient vector ``d_k`` is allocated per task.
    Existing coefficients can be frozen after task consolidation.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 2, scale: float = 1.0) -> None:
        super().__init__()
        if min(in_features, out_features, rank) < 1:
            raise ValueError("adapter dimensions must be positive")
        if rank > min(in_features, out_features):
            raise ValueError("rank cannot exceed the input/output dimensions")
        if scale < 0.0:
            raise ValueError("scale must be nonnegative")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = scale
        self.shared_down = nn.Parameter(torch.randn(in_features, rank) / in_features**0.5)
        self.shared_up = nn.Parameter(torch.randn(out_features, rank) / out_features**0.5)
        self.task_coefficients = nn.ParameterDict()

    @property
    def task_ids(self) -> tuple[int, ...]:
        return tuple(sorted(int(key) for key in self.task_coefficients.keys()))

    @property
    def shared_parameter_count(self) -> int:
        return self.shared_down.numel() + self.shared_up.numel()

    @property
    def task_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.task_coefficients.values())

    def add_task(self, task_id: int, coefficients: Tensor | None = None) -> None:
        key = str(int(task_id))
        if key in self.task_coefficients:
            raise KeyError(f"adapter for task {task_id} already exists")
        if coefficients is None:
            value = torch.zeros(self.rank)
        else:
            if coefficients.shape != (self.rank,):
                raise ValueError("coefficients have the wrong shape")
            value = coefficients.detach().float().clone()
        self.task_coefficients[key] = nn.Parameter(value)

    def freeze_task(self, task_id: int) -> None:
        key = str(int(task_id))
        if key not in self.task_coefficients:
            raise KeyError(f"adapter for task {task_id} is not registered")
        self.task_coefficients[key].requires_grad_(False)

    def train_only_task(self, task_id: int) -> None:
        key = str(int(task_id))
        if key not in self.task_coefficients:
            raise KeyError(f"adapter for task {task_id} is not registered")
        for parameter in self.task_coefficients.values():
            parameter.requires_grad_(False)
        self.task_coefficients[key].requires_grad_(True)

    def forward(self, features: Tensor, task_id: int | None = None) -> Tensor:
        if features.shape[-1] != self.in_features:
            raise ValueError(f"features must end in width {self.in_features}")
        if task_id is None:
            return features
        key = str(int(task_id))
        if key not in self.task_coefficients:
            raise KeyError(f"adapter for task {task_id} is not registered")
        coefficients = self.task_coefficients[key]
        down = features @ self.shared_down
        residual = (down * coefficients) @ self.shared_up.T
        return features + self.scale * residual
