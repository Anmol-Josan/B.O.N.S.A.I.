from __future__ import annotations

import torch

from src.utils.visualization import plot_accuracy_curves, plot_mask_sparsity


def test_visualization_helpers_write_artifacts(tmp_path) -> None:
    accuracy_path = tmp_path / "accuracy.png"
    sparsity_path = tmp_path / "sparsity.png"
    plot_accuracy_curves({"BONSAI": [0.5, 0.7], "EWC": [0.4, 0.6]}, accuracy_path)
    plot_mask_sparsity(
        {1: {"layer.weight": torch.tensor([True, False, True, False])}}, sparsity_path
    )

    assert accuracy_path.exists() and accuracy_path.stat().st_size > 0
    assert sparsity_path.exists() and sparsity_path.stat().st_size > 0

