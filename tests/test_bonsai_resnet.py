from __future__ import annotations

import torch

from src.models.bonsai_resnet import BonsaiResNet18


def test_bonsai_resnet_exposes_vib_loss_and_dynamic_junction_growth() -> None:
    torch.manual_seed(0)
    model = BonsaiResNet18(num_classes=3, junction_growth_ratio=0.04, plateau_patience=2)
    initial = model.total_parameters
    logits = model(torch.randn(2, 3, 32, 32))

    assert logits.shape == (2, 3)
    assert model.kl_loss.ndim == 0
    assert model.kl_loss.item() >= 0.0
    assert model.record_validation_loss(1.0) is False
    assert model.record_validation_loss(0.999) is False
    assert model.record_validation_loss(0.998) is True
    assert model.total_parameters > initial

