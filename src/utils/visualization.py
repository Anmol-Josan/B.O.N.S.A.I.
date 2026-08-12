"""Artifact plots for benchmark curves and critical-mask profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch


def _prepare_destination(destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def plot_accuracy_curves(
    curves: Mapping[str, Sequence[float]], destination: str | Path, title: str = "Continual accuracy"
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = _prepare_destination(destination)
    figure, axis = plt.subplots(figsize=(7, 4))
    for method, values in curves.items():
        axis.plot(range(1, len(values) + 1), values, marker="o", label=method)
    axis.set_xlabel("Task after which evaluation is performed")
    axis.set_ylabel("Average accuracy")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    if curves:
        axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def plot_mask_sparsity(
    masks_by_task: Mapping[int, Mapping[str, torch.Tensor]],
    destination: str | Path,
    title: str = "Critical-mask sparsity",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = _prepare_destination(destination)
    tasks = sorted(masks_by_task)
    sparsity = []
    for task in tasks:
        masks = masks_by_task[task]
        total = sum(mask.numel() for mask in masks.values())
        frozen = sum(mask.detach().bool().sum().item() for mask in masks.values())
        sparsity.append(frozen / total if total else 0.0)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(tasks, sparsity, marker="o")
    axis.set_xlabel("Task")
    axis.set_ylabel("Fraction of frozen entries")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)

