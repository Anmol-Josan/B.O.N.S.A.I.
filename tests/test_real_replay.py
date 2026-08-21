from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from src.benchmarks.real_replay import (
    CompactCIFARNet,
    RealReplayConfig,
    _balanced_positions,
    _task_order,
    run_real_replay_method,
)
from src.data.task_splits import ClassIncrementalTask


def test_real_replay_backbone_and_memory_budget_are_explicit() -> None:
    model = CompactCIFARNet()
    assert sum(parameter.numel() for parameter in model.parameters()) == 122884
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    selected = _balanced_positions(labels, 3)
    assert selected.numel() == 3
    assert torch.bincount(labels[selected], minlength=3).tolist() == [1, 1, 1]


def test_task_order_generator_is_deterministic_and_nonidentity() -> None:
    assert _task_order(5, 7, 0) == [0, 1, 2, 3, 4]
    assert _task_order(5, 7, 1) == _task_order(5, 7, 1)
    assert sorted(_task_order(5, 7, 1)) == [0, 1, 2, 3, 4]


def test_matched_real_replay_smoke_without_dataset_io() -> None:
    train_tasks = []
    test_tasks = []
    for task_id in range(2):
        labels = torch.tensor([task_id * 2, task_id * 2 + 1] * 2)
        inputs = torch.randn(4, 3, 32, 32)
        base = TensorDataset(inputs, labels)
        train_tasks.append(
            ClassIncrementalTask(
                base_dataset=base,
                indices=list(range(4)),
                global_labels=labels,
                local_labels=labels - task_id * 2,
                task_id=task_id,
                classes=(task_id * 2, task_id * 2 + 1),
            )
        )
        test_tasks.append(train_tasks[-1])
    config = RealReplayConfig(
        methods=("ATGFR",),
        epochs_per_task=1,
        batch_size=4,
        memory_per_task=2,
        train_samples_per_class=2,
        test_samples_per_class=2,
    )
    record = run_real_replay_method(
        "ATGFR", train_tasks, test_tasks, [0, 1], seed=7, config=config
    )
    assert record["memory_images"] == 4
    assert record["memory_scalar_elements"] > 4 * 3 * 32 * 32
    assert 0.0 <= record["average_accuracy"] <= 1.0
