from __future__ import annotations

import torch

from src.data.synthetic_images import make_synthetic_image_tasks


def test_structured_image_tasks_are_deterministic_and_class_disjoint() -> None:
    train_a, test_a = make_synthetic_image_tasks(
        num_tasks=3,
        classes_per_task=2,
        train_samples_per_class=4,
        test_samples_per_class=2,
        seed=12,
    )
    train_b, test_b = make_synthetic_image_tasks(
        num_tasks=3,
        classes_per_task=2,
        train_samples_per_class=4,
        test_samples_per_class=2,
        seed=12,
    )

    assert len(train_a) == len(test_a) == 3
    assert torch.equal(
        torch.stack([train_a[0][index][0] for index in range(len(train_a[0]))]),
        torch.stack([train_b[0][index][0] for index in range(len(train_b[0]))]),
    )
    assert torch.equal(test_a[-1].global_labels, test_b[-1].global_labels)
    assert set(train_a[0].global_labels.tolist()).isdisjoint(train_a[1].global_labels.tolist())
    assert all(len(task) == 8 for task in train_a)
    assert all(len(task) == 4 for task in test_a)


def test_structured_image_classes_remain_distinct_beyond_twelve_classes() -> None:
    train_tasks, _ = make_synthetic_image_tasks(
        num_tasks=5,
        classes_per_task=3,
        train_samples_per_class=2,
        test_samples_per_class=1,
        noise=0.0,
        seed=5,
    )

    class_zero = train_tasks[0].global_labels.eq(0).nonzero(as_tuple=False)[0].item()
    class_twelve = train_tasks[4].global_labels.eq(12).nonzero(as_tuple=False)[0].item()
    image_zero = train_tasks[0].base_dataset[train_tasks[0].indices[class_zero]][0]
    image_twelve = train_tasks[4].base_dataset[train_tasks[4].indices[class_twelve]][0]

    assert not torch.equal(image_zero, image_twelve)
