from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from src.data.task_splits import split_dataset_by_classes


def test_class_split_maps_labels_locally_without_cross_task_leakage() -> None:
    inputs = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    tasks = split_dataset_by_classes(TensorDataset(inputs, labels), classes_per_task=2)

    assert len(tasks) == 2
    assert [len(task) for task in tasks] == [6, 6]
    assert set(tasks[0].global_labels.tolist()) == {0, 1}
    assert set(tasks[1].global_labels.tolist()) == {2, 3}
    assert set(tasks[0].labels.tolist()) == {0, 1}
    assert set(tasks[1].labels.tolist()) == {0, 1}
    assert set(tasks[0].indices).isdisjoint(set(tasks[1].indices))


def test_split_rejects_incomplete_class_groups() -> None:
    dataset = TensorDataset(torch.randn(3, 2), torch.tensor([0, 1, 2]))

    try:
        split_dataset_by_classes(dataset, classes_per_task=2)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("expected incomplete class groups to be rejected")

