from __future__ import annotations

import torch

from src.models.dynamic_resnet import DynamicBackbone


def test_plateau_triggers_one_local_junction_with_bounded_growth() -> None:
    torch.manual_seed(0)
    model = DynamicBackbone(
        input_channels=1,
        num_classes=3,
        base_channels=16,
        junction_growth_ratio=0.04,
        plateau_patience=5,
        plateau_min_improvement=0.01,
    )
    initial_count = model.total_parameters

    triggered = [model.record_validation_loss(loss) for loss in [1.0, 0.999, 0.998, 0.997, 0.996, 0.995]]

    assert triggered[-1] is True
    assert sum(triggered) == 1
    assert len(model.junctions) == 1
    growth = (model.total_parameters - initial_count) / initial_count
    assert 0.03 <= growth <= 0.05


def test_dynamic_junction_preserves_logits_shape_and_improvement_resets_plateau() -> None:
    model = DynamicBackbone(
        input_channels=1,
        num_classes=4,
        base_channels=16,
        plateau_patience=2,
        plateau_min_improvement=0.01,
    )
    assert model(torch.randn(2, 1, 8, 8)).shape == (2, 4)

    assert model.record_validation_loss(1.0) is False
    assert model.record_validation_loss(0.999) is False
    assert model.record_validation_loss(0.98) is False
    assert len(model.junctions) == 0

