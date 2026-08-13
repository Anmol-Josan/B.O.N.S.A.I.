from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from src.data.task_splits import split_dataset_by_classes
from src.utils.real_benchmark import GlobalLabelView


def test_real_runner_view_exposes_global_labels_without_changing_task_indices() -> None:
    dataset = TensorDataset(torch.randn(8, 2), torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]))
    task = split_dataset_by_classes(dataset, classes_per_task=2)[1]
    view = GlobalLabelView(task)

    sample, label = view[0]
    assert torch.equal(sample, task[0][0])
    assert label.item() == task.global_labels[0].item()
    assert len(view) == len(task)

