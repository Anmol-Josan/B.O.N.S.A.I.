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


def test_bonsai_can_route_real_image_features_without_a_task_id() -> None:
    torch.manual_seed(2)
    model = BonsaiResNet18(num_classes=6, task_adapter_rank=1)
    model.add_task_path()
    model.add_task_path()
    inputs = torch.randn(4, 3, 32, 32)
    model.register_task_route(0, inputs, torch.tensor([0, 1, 2, 0]), classes_per_task=3)
    model.register_task_route(1, inputs, torch.tensor([3, 4, 5, 3]), classes_per_task=3)

    predictions, selected_tasks, entropies = model.predict_task_free(
        inputs, classes_per_task=3, route_strategy="prototype"
    )

    assert predictions.shape == (4,)
    assert selected_tasks.shape == (4,)
    assert entropies.shape == (4, 2)
    assert selected_tasks.min() >= 0 and selected_tasks.max() < 2
    assert torch.isfinite(entropies).all()
    learned_predictions, learned_tasks, _ = model.predict_task_free(
        inputs, classes_per_task=3, route_strategy="learned"
    )
    assert learned_predictions.shape == (4,)
    assert learned_tasks.max() < 2


def test_bonsai_real_mode_uses_private_local_heads() -> None:
    model = BonsaiResNet18(num_classes=6, classes_per_task=3, task_adapter_rank=1)
    model.add_task_path()
    model.add_task_path()
    inputs = torch.randn(2, 3, 32, 32)

    assert model.task_logits(inputs, 0).shape == (2, 3)
    assert model.task_logits(inputs, 1).shape == (2, 3)


def test_bonsai_growth_tracker_resets_between_tasks() -> None:
    model = BonsaiResNet18(num_classes=3, plateau_patience=2)
    assert model.record_validation_loss(1.0) is False
    assert model.record_validation_loss(0.999) is False
    model.start_task()
    assert model.record_validation_loss(5.0) is False
    assert model.record_validation_loss(4.0) is False
    assert len(model.junctions) == 0
