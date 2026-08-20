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
    model = BonsaiResNet18(
        num_classes=6, classes_per_task=3, task_adapter_rank=1
    )
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
    compatibility_predictions, compatibility_tasks, _ = model.predict_task_free(
        inputs, classes_per_task=3, route_strategy="compatibility"
    )
    assert compatibility_predictions.shape == (4,)
    assert compatibility_tasks.max() < 2
    global_predictions, global_tasks, _ = model.predict_task_free(
        inputs, classes_per_task=3, route_strategy="global_argmax"
    )
    assert global_predictions.shape == (4,)
    assert global_tasks.max() < 2
    energy_predictions, energy_tasks, _ = model.predict_task_free(
        inputs, classes_per_task=3, route_strategy="local_energy"
    )
    assert energy_predictions.shape == (4,)
    assert energy_tasks.max() < 2
    direct_predictions, direct_tasks, _ = model.predict_task_free(
        inputs, classes_per_task=3, route_strategy="global_direct"
    )
    assert direct_predictions.shape == (4,)
    assert direct_tasks.max() < 2
    cosine_predictions, cosine_tasks, _ = model.predict_task_free(
        inputs, classes_per_task=3, route_strategy="cosine"
    )
    assert cosine_predictions.shape == (4,)
    assert cosine_tasks.max() < 2
    discovery_predictions, discovery_tasks, _ = model.predict_task_free(
        inputs, classes_per_task=3, route_strategy="discovery"
    )
    assert discovery_predictions.shape == (4,)
    assert discovery_tasks.max() < 2
    evidence_predictions, evidence_tasks, _ = model.predict_task_free(
        inputs, classes_per_task=3, route_strategy="evidence"
    )
    assert evidence_predictions.shape == (4,)
    assert evidence_tasks.max() < 2


def test_bonsai_real_mode_uses_private_local_heads() -> None:
    model = BonsaiResNet18(num_classes=6, classes_per_task=3, task_adapter_rank=1)
    model.add_task_path()
    model.add_task_path()
    inputs = torch.randn(2, 3, 32, 32)

    assert model.task_logits(inputs, 0).shape == (2, 3)
    assert model.task_logits(inputs, 1).shape == (2, 3)


def test_bonsai_real_mode_keeps_a_trainable_global_scaffold() -> None:
    torch.manual_seed(3)
    model = BonsaiResNet18(num_classes=6, classes_per_task=3, task_adapter_rank=1)
    model.add_task_path()
    inputs = torch.randn(3, 3, 32, 32)
    global_logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(global_logits, torch.tensor([0, 1, 2]))
    loss.backward()

    assert global_logits.shape == (3, 6)
    assert model.classifier.weight.grad is not None
    assert any(parameter is model.classifier.weight for parameter in model.task_parameters(0))


def test_task_free_compatibility_route_reuses_cached_path_features(monkeypatch) -> None:
    model = BonsaiResNet18(num_classes=6, classes_per_task=3, task_adapter_rank=1)
    model.add_task_path()
    model.add_task_path()
    calls = 0
    original_forward_features = model.forward_features

    def counted_forward_features(inputs, task_id=None):
        nonlocal calls
        calls += 1
        return original_forward_features(inputs, task_id=task_id)

    monkeypatch.setattr(model, "forward_features", counted_forward_features)
    model.predict_task_free(
        torch.randn(2, 3, 32, 32), classes_per_task=3, route_strategy="compatibility"
    )
    assert calls == 2


def test_task_parameter_groups_keep_old_paths_out_of_the_optimizer() -> None:
    model = BonsaiResNet18(num_classes=6, classes_per_task=3, task_adapter_rank=1)
    model.add_task_path()
    model.add_task_path()
    groups = model.task_parameter_groups(1, learning_rate=0.01, shared_learning_rate_scale=0.1)
    optimized = {
        id(parameter)
        for group in groups
        for parameter in group["params"]
    }

    assert id(next(model.route_compatibility_heads[0].parameters())) not in optimized
    assert id(next(model.route_compatibility_heads[1].parameters())) in optimized
    assert id(model.classifier.weight) in optimized
    assert not optimized.intersection(
        {id(parameter) for parameter in model.route_discovery_parameters()}
    )


def test_input_only_route_discovery_branch_is_compact_and_exposed() -> None:
    model = BonsaiResNet18(
        num_classes=6,
        classes_per_task=3,
        task_adapter_rank=1,
        route_discovery_hidden_dim=12,
    )
    assert model.route_discovery_logits(torch.randn(2, 3, 32, 32)).shape == (2, 2)
    assert sum(parameter.numel() for parameter in model.route_discovery_parameters()) < 100_000


def test_local_classifier_evidence_has_expected_calibration_features() -> None:
    logits = torch.randn(4, 3)
    features = BonsaiResNet18.route_evidence_features(logits)
    assert features.shape == (4, 7)
    assert torch.isfinite(features).all()


def test_task_parameter_groups_can_accelerate_the_global_classifier_separately() -> None:
    model = BonsaiResNet18(num_classes=6, classes_per_task=3, task_adapter_rank=1)
    model.add_task_path()
    groups = model.task_parameter_groups(
        0,
        learning_rate=0.01,
        shared_learning_rate_scale=0.1,
        classifier_learning_rate_scale=1.0,
    )
    classifier_group = next(
        group for group in groups if id(model.classifier.weight) in {id(p) for p in group["params"]}
    )
    assert classifier_group["lr"] == 0.01


def test_bonsai_supports_compact_nonlinear_route_heads() -> None:
    model = BonsaiResNet18(
        num_classes=6,
        classes_per_task=3,
        task_adapter_rank=1,
        route_hidden_dim=8,
    )
    model.add_task_path()
    logits = model.route_compatibility_heads[0](torch.randn(2, 512))

    assert logits.shape == (2, 1)


def test_bonsai_can_configure_nonzero_task_adapter_initialization() -> None:
    model = BonsaiResNet18(
        num_classes=6,
        classes_per_task=3,
        task_adapter_rank=1,
        adapter_residual_init_std=0.02,
    )
    model.add_task_path()
    assert model.task_adapters[0].up.weight.abs().sum() > 0


def test_bonsai_uses_a_nonlinear_shared_route_gate_when_configured() -> None:
    model = BonsaiResNet18(
        num_classes=6,
        classes_per_task=3,
        task_adapter_rank=1,
        route_hidden_dim=8,
    )
    assert isinstance(model.route_head, torch.nn.Sequential)
    assert model.route_head_output_features == 2
    assert model.route_logits_from_features(torch.randn(2, 512)).shape == (2, 2)


def test_bonsai_growth_tracker_resets_between_tasks() -> None:
    model = BonsaiResNet18(num_classes=3, plateau_patience=2)
    assert model.record_validation_loss(1.0) is False
    assert model.record_validation_loss(0.999) is False
    model.start_task()
    assert model.record_validation_loss(5.0) is False
    assert model.record_validation_loss(4.0) is False
    assert len(model.junctions) == 0
