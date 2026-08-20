from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from src.data.task_splits import split_dataset_by_classes
from src.models.bonsai_resnet import BonsaiResNet18
from src.algorithms.baselines import PNN
from src.utils.real_benchmark import (
    GlobalLabelView,
    RealBenchmarkConfig,
    _forward,
    _task_target,
    bonsai_training_loss,
    _fit_route_compatibility,
    _fit_route_discovery,
    _fit_route_evidence,
    _refresh_route_state,
    bonsai_route_training_loss,
    bonsai_feature_replay_loss,
    bonsai_local_replay_loss,
    _fit_global_head,
    _freeze_backbone_batchnorm,
    limit_task_samples,
    split_task_views,
    resolve_device,
    balanced_sample_positions,
)


def test_real_benchmark_method_selection_rejects_empty_or_unknown_methods() -> None:
    from src.utils.real_benchmark import run_real_suite

    base = dict(dataset="synthetic_images", data_root=Path("data"))
    try:
        run_real_suite(RealBenchmarkConfig(methods=(), **base))
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("empty method selection should fail")

    try:
        run_real_suite(RealBenchmarkConfig(methods=("unknown",), **base))
    except ValueError as error:
        assert "unknown benchmark methods" in str(error)
    else:
        raise AssertionError("unknown method selection should fail")


def test_resolve_device_keeps_cpu_path_available() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_balanced_route_sampling_covers_classes_round_robin() -> None:
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    positions = balanced_sample_positions(labels, max_samples=5)
    assert labels[positions].tolist() == [0, 1, 2, 0, 1]


def test_real_runner_view_exposes_global_labels_without_changing_task_indices() -> None:
    dataset = TensorDataset(torch.randn(8, 2), torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]))
    task = split_dataset_by_classes(dataset, classes_per_task=2)[1]
    view = GlobalLabelView(task)

    sample, label = view[0]
    assert torch.equal(sample, task[0][0])
    assert label.item() == task.global_labels[0].item()
    assert len(view) == len(task)


def test_train_validation_task_views_are_class_balanced_and_disjoint() -> None:
    labels = torch.tensor([0, 1, 2, 3] * 4)
    dataset = TensorDataset(torch.randn(16, 2), labels)
    tasks = split_dataset_by_classes(dataset, classes_per_task=2)
    train_tasks, validation_tasks = split_task_views(tasks, validation_fraction=0.25, seed=4)

    for train_task, validation_task in zip(train_tasks, validation_tasks):
        assert set(train_task.indices).isdisjoint(validation_task.indices)
        assert torch.bincount(train_task.labels, minlength=2).min() > 0
        assert torch.bincount(validation_task.labels, minlength=2).min() > 0


def test_limited_task_views_preserve_every_class_and_cap_samples() -> None:
    labels = torch.tensor([0, 1, 2, 3] * 5)
    tasks = split_dataset_by_classes(TensorDataset(torch.randn(20, 2), labels), classes_per_task=2)
    limited = limit_task_samples(tasks, max_samples_per_class=3, seed=8)

    for task in limited:
        assert len(task) == 6
        assert torch.bincount(task.labels, minlength=2).tolist() == [3, 3]


def test_bonsai_training_loss_includes_global_scaffold_gradient() -> None:
    model = BonsaiResNet18(num_classes=4, classes_per_task=2, task_adapter_rank=1)
    model.add_task_path()
    config = RealBenchmarkConfig(
        dataset="synthetic_images",
        data_root=Path("data"),
        global_loss_weight=1.0,
    )
    inputs = torch.randn(3, 3, 32, 32)
    labels = torch.tensor([0, 1, 0])

    loss = bonsai_training_loss(model, inputs, labels, task_id=0, classes_per_task=2, config=config)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.classifier.weight.grad is not None
    assert model.classifier.weight.grad.abs().sum() > 0


def test_bonsai_route_training_loss_replays_prior_task_route_memory() -> None:
    model = BonsaiResNet18(num_classes=4, classes_per_task=2, task_adapter_rank=1)
    model.add_task_path()
    model.add_task_path()
    config = RealBenchmarkConfig(
        dataset="synthetic_images",
        data_root=Path("data"),
        route_training_weight=1.0,
        route_replay_per_task=2,
    )
    inputs = torch.randn(3, 3, 32, 32)
    memory = [(torch.randn(2, 3, 32, 32), 0)]
    loss = bonsai_route_training_loss(
        model, inputs, task_id=1, route_memory=memory, config=config, device=torch.device("cpu")
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert next(model.route_head.parameters()).grad is not None


def test_global_head_calibration_updates_only_the_global_classifier() -> None:
    model = BonsaiResNet18(num_classes=4, classes_per_task=2, task_adapter_rank=1)
    model.add_task_path()
    config = RealBenchmarkConfig(
        dataset="synthetic_images", data_root=Path("data"), global_head_epochs=1
    )
    inputs = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([0, 1])
    model.eval()
    before_features = model.forward_features(inputs).detach().clone()
    before_backbone = model.backbone.conv1.weight.detach().clone()
    _fit_global_head(model, [(inputs, labels)], config, torch.device("cpu"))
    assert torch.allclose(before_features, model.forward_features(inputs).detach())
    assert torch.equal(before_backbone, model.backbone.conv1.weight)


def test_backbone_batchnorm_freeze_preserves_eval_mode_for_all_bn_layers() -> None:
    model = BonsaiResNet18(num_classes=4, classes_per_task=2, task_adapter_rank=1)
    model.train()
    _freeze_backbone_batchnorm(model)
    batch_norms = [
        module for module in model.backbone.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    assert batch_norms and all(not module.training for module in batch_norms)


def test_route_compatibility_heads_fit_on_positive_and_negative_tasks() -> None:
    model = BonsaiResNet18(num_classes=4, classes_per_task=2, task_adapter_rank=1)
    model.add_task_path()
    model.add_task_path()
    config = RealBenchmarkConfig(
        dataset="synthetic_images",
        data_root=Path("data"),
        route_compatibility_epochs=1,
    )
    memory = [
        (torch.randn(1, 3, 32, 32), 0),
        (torch.randn(1, 3, 32, 32), 1),
    ]

    _fit_route_compatibility(model, memory, config, torch.device("cpu"))
    predictions, selected_tasks, _ = model.predict_task_free(
        torch.randn(2, 3, 32, 32), classes_per_task=2, route_strategy="compatibility"
    )

    assert predictions.shape == (2,)
    assert selected_tasks.max() < 2


def test_route_state_refresh_recomputes_features_after_shared_rewire() -> None:
    model = BonsaiResNet18(num_classes=4, classes_per_task=2, task_adapter_rank=1)
    model.add_task_path()
    model.add_task_path()
    inputs0 = torch.randn(2, 3, 32, 32)
    inputs1 = torch.randn(2, 3, 32, 32)
    labels0 = torch.tensor([0, 1])
    labels1 = torch.tensor([2, 3])
    config = RealBenchmarkConfig(
        dataset="synthetic_images",
        data_root=Path("data"),
        route_head_epochs=1,
        route_compatibility_epochs=1,
    )
    model.eval()
    model.register_task_route(0, inputs0, labels0, classes_per_task=2)
    before = model._route_prototypes[0].clone()
    with torch.no_grad():
        model.backbone.conv1.weight.add_(0.01)
    calibration_memory = [
        (inputs0, labels0, 0),
        (inputs1, labels1, 1),
    ]
    route_memory = [(inputs0, 0), (inputs1, 1)]
    _refresh_route_state(
        model,
        calibration_memory,
        route_memory,
        config,
        torch.device("cpu"),
        classes_per_task=2,
    )
    assert len(model._route_prototypes) == 2
    assert not torch.allclose(before, model._route_prototypes[0])


def test_input_only_route_discovery_calibration_updates_only_its_branch() -> None:
    model = BonsaiResNet18(num_classes=4, classes_per_task=2, task_adapter_rank=1)
    memory = [
        (torch.randn(4, 3, 32, 32), torch.tensor([0, 0, 1, 1]), 0),
        (torch.randn(4, 3, 32, 32), torch.tensor([2, 2, 3, 3]), 1),
    ]
    config = RealBenchmarkConfig(
        dataset="synthetic_images",
        data_root=Path("data"),
        learning_rate=0.01,
        route_discovery_epochs=1,
    )
    before = model.backbone.conv1.weight.detach().clone()
    discovery_before = next(model.route_discovery_encoder.parameters()).detach().clone()
    _fit_route_discovery(model, memory, config, torch.device("cpu"))
    discovery_after = next(model.route_discovery_encoder.parameters()).detach()
    assert torch.equal(before, model.backbone.conv1.weight)
    assert not torch.equal(discovery_before, discovery_after)


def test_route_evidence_calibration_updates_only_path_calibrators() -> None:
    model = BonsaiResNet18(num_classes=4, classes_per_task=2, task_adapter_rank=1)
    model.add_task_path()
    model.add_task_path()
    route_memory = [
        (torch.randn(3, 3, 32, 32), 0),
        (torch.randn(3, 3, 32, 32), 1),
    ]
    config = RealBenchmarkConfig(
        dataset="synthetic_images",
        data_root=Path("data"),
        learning_rate=0.01,
        route_evidence_epochs=1,
    )
    before = next(model.route_evidence_heads[0].parameters()).detach().clone()
    backbone_before = model.backbone.conv1.weight.detach().clone()
    _fit_route_evidence(model, route_memory, config, torch.device("cpu"))
    after = next(model.route_evidence_heads[0].parameters()).detach()
    assert not torch.equal(before, after)
    assert torch.equal(backbone_before, model.backbone.conv1.weight)


def test_feature_replay_loss_backpropagates_through_the_old_task_path() -> None:
    model = BonsaiResNet18(num_classes=4, classes_per_task=2, task_adapter_rank=1)
    model.add_task_path()
    inputs = torch.randn(3, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        target_features = model.forward_features(inputs, task_id=0).cpu()
    model.train()
    config = RealBenchmarkConfig(
        dataset="synthetic_images",
        data_root=Path("data"),
        feature_replay_weight=1.0,
        feature_replay_per_task=2,
    )
    loss = bonsai_feature_replay_loss(
        model,
        [(inputs, 0, target_features)],
        config,
        torch.device("cpu"),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.backbone.conv1.weight.grad is not None


def test_local_replay_loss_backpropagates_through_previous_task_heads() -> None:
    model = BonsaiResNet18(num_classes=4, classes_per_task=2, task_adapter_rank=1)
    model.add_task_path()
    model.add_task_path()
    inputs = torch.randn(3, 3, 32, 32)
    labels = torch.tensor([0, 1, 0])
    config = RealBenchmarkConfig(
        dataset="synthetic_images",
        data_root=Path("data"),
        local_replay_weight=1.0,
        local_replay_per_task=2,
    )
    loss = bonsai_local_replay_loss(
        model,
        [(inputs, labels, 0)],
        classes_per_task=2,
        config=config,
        device=torch.device("cpu"),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.backbone.conv1.weight.grad is not None


def test_pnn_baseline_uses_local_task_class_slice() -> None:
    model = PNN(num_classes=6)
    model.add_task_column()
    labels = torch.tensor([3, 4])
    inputs = torch.randn(2, 3, 32, 32)

    logits = _forward(model, inputs, task_id=1, classes_per_task=3)
    targets = _task_target(model, labels, task_id=1, classes_per_task=3)

    assert logits.shape == (2, 3)
    assert targets.tolist() == [0, 1]
