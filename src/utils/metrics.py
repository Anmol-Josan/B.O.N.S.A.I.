"""Continual-learning metrics and lightweight artifact serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


def average_accuracy(current_accuracies: Sequence[float]) -> float:
    """Compute A_T, the mean accuracy over all tasks seen at step T."""

    if not current_accuracies:
        return 0.0
    return float(sum(current_accuracies) / len(current_accuracies))


def forgetting_measure(accuracy_history: Sequence[Sequence[float]]) -> float:
    """Compute final average forgetting from the standard peak-before-final score.

    For each task ``i`` that is present before the final step, the reference is
    the best accuracy observed for that task at any evaluation step from its
    first exposure through the penultimate step.  The result may be negative
    when the final model improves an old task beyond every earlier checkpoint.
    """

    if len(accuracy_history) <= 1:
        return 0.0
    current = accuracy_history[-1]
    prior_task_count = min(len(current), len(accuracy_history) - 1)
    if prior_task_count == 0:
        return 0.0
    forgetting = []
    for task_id in range(prior_task_count):
        observed = [
            row[task_id]
            for row in accuracy_history[task_id:-1]
            if len(row) > task_id
        ]
        if not observed:
            continue
        best_before_final = max(observed)
        forgetting.append(best_before_final - current[task_id])
    if not forgetting:
        return 0.0
    return float(sum(forgetting) / len(forgetting))


def parameter_overhead(initial_parameters: int, total_parameters: int) -> float:
    """Return growth relative to the initial model as a percentage."""

    if initial_parameters <= 0:
        raise ValueError("initial_parameters must be positive")
    return 100.0 * (total_parameters - initial_parameters) / initial_parameters


def save_metrics_json(destination: str | Path, payload: dict) -> None:
    """Write a structured JSON artifact, creating its parent directory."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
