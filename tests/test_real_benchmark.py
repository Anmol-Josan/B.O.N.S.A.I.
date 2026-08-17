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
    limit_task_samples,
    split_task_views,
)


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


def test_pnn_baseline_uses_local_task_class_slice() -> None:
    model = PNN(num_classes=6)
    model.add_task_column()
    labels = torch.tensor([3, 4])
    inputs = torch.randn(2, 3, 32, 32)

    logits = _forward(model, inputs, task_id=1, classes_per_task=3)
    targets = _task_target(model, labels, task_id=1, classes_per_task=3)

    assert logits.shape == (2, 3)
    assert targets.tolist() == [0, 1]
