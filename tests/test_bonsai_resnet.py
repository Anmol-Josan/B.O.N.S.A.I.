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


def test_bonsai_task_adapter_path_is_small_and_task_selectable() -> None:
    torch.manual_seed(1)
    model = BonsaiResNet18(num_classes=6, task_adapter_rank=2)
    initial = model.total_parameters
    model.add_task_path()
    after_first_path = model.total_parameters
    model.add_task_path()

    inputs = torch.randn(2, 3, 32, 32)
    logits0 = model(inputs, task_id=0)
    logits1 = model(inputs, task_id=1)

    assert after_first_path > initial
    assert model.total_parameters > after_first_path
    assert logits0.shape == logits1.shape == (2, 6)
    assert not torch.equal(logits0, logits1)
    assert sum(parameter.numel() for parameter in model.task_parameters(1)) < sum(
        parameter.numel() for parameter in model.parameters()
    )
