from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from src.data.task_splits import split_dataset_by_classes
from src.utils.real_benchmark import GlobalLabelView, limit_task_samples, split_task_views


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
