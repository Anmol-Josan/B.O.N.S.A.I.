from __future__ import annotations

import torch

from src.data.synthetic import make_synthetic_tasks


def test_synthetic_tasks_are_disjoint_and_reproducible() -> None:
    tasks_a = make_synthetic_tasks(num_tasks=3, classes_per_task=2, samples_per_class=5, seed=4)
    tasks_b = make_synthetic_tasks(num_tasks=3, classes_per_task=2, samples_per_class=5, seed=4)

    assert len(tasks_a) == 3
    assert all(len(task) == 10 for task in tasks_a)
    assert torch.equal(tasks_a[1].inputs, tasks_b[1].inputs)
    assert set(tasks_a[0].labels.tolist()).isdisjoint(set(tasks_a[1].global_labels.tolist()))
    assert set(tasks_a[0].global_labels.tolist()) == {0, 1}
    assert set(tasks_a[2].global_labels.tolist()) == {4, 5}

