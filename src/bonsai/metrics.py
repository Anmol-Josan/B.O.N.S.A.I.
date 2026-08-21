"""Diagnostics for geometry, routing, parameter scaling, and interference."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from src.bonsai.system import BONSAISystem


def routing_margin(system: BONSAISystem) -> float:
    ids = system.repository.task_ids
    if len(ids) < 2:
        return 0.0
    values = []
    for index, left_id in enumerate(ids):
        left = system.repository.get(left_id).prototype
        for right_id in ids[index + 1 :]:
            right = system.repository.get(right_id).prototype
            values.append(float(system.metric.distance(left, right, left_id).detach()))
    return min(values) if values else 0.0


def within_task_radius(system: BONSAISystem, task_latents: dict[int, Tensor]) -> float:
    radii = []
    for task_id, latents in task_latents.items():
        if task_id not in system.repository.records:
            continue
        prototype = system.repository.get(task_id).prototype.to(latents.device, latents.dtype)
        distances = system.metric.distance(latents, prototype, task_id)
        if distances.numel():
            radii.append(float(distances.max().detach()))
    return max(radii) if radii else 0.0


def separation_ratio(margin: float, radius: float) -> float:
    if radius <= 0.0:
        return float("inf") if margin > 0.0 else 0.0
    return margin / (2.0 * radius)


def parameter_overhead(system: BONSAISystem) -> float:
    return system.parameter_overhead


def interference_drop(before: Iterable[float], after: Iterable[float]) -> float:
    """Mean accuracy drop on previously learned tasks in percentage points."""

    before_values = list(before)
    after_values = list(after)
    if len(before_values) != len(after_values):
        raise ValueError("before and after must have equal lengths")
    if not before_values:
        return 0.0
    return float(sum(before_value - after_value for before_value, after_value in zip(before_values, after_values)) / len(before_values))
