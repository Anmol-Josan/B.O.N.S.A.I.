"""Deterministic structured image tasks for fast architecture comparisons."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from src.data.task_splits import ClassIncrementalTask, split_dataset_by_classes


def _class_prototype(global_class: int, image_size: int = 32) -> Tensor:
    """Create a class-specific colored geometric prototype."""

    if image_size < 16:
        raise ValueError("image_size must be at least 16")
    image = torch.zeros(3, image_size, image_size)
    channel = global_class % 3
    prototype_group = global_class // 12
    rows = torch.arange(image_size).view(-1, 1)
    columns = torch.arange(image_size).view(1, -1)
    pattern = global_class % 4
    row_shift = (prototype_group * 3) % image_size
    column_shift = (prototype_group * 5) % image_size
    rows = (rows + row_shift) % image_size
    columns = (columns + column_shift) % image_size
    if pattern == 0:
        mask = (rows % 8) < 4
    elif pattern == 1:
        mask = (columns % 8) < 4
    elif pattern == 2:
        mask = ((rows + columns) % 8) < 4
    else:
        mask = ((rows // 4 + columns // 4) % 2) == 0
    image[channel] = mask.float()
    marker_size = 3
    marker_row = (prototype_group * 7) % (image_size - marker_size + 1)
    marker_column = (prototype_group * 11) % (image_size - marker_size + 1)
    marker_channel = (channel + 1) % 3
    image[marker_channel, marker_row : marker_row + marker_size, marker_column : marker_column + marker_size] = 0.8
    return image


def make_synthetic_image_tasks(
    num_tasks: int = 4,
    classes_per_task: int = 3,
    train_samples_per_class: int = 24,
    test_samples_per_class: int = 12,
    image_size: int = 32,
    noise: float = 0.25,
    seed: int = 7,
) -> tuple[list[ClassIncrementalTask], list[ClassIncrementalTask]]:
    """Return disjoint train/test class-incremental image task views."""

    if min(num_tasks, classes_per_task, train_samples_per_class, test_samples_per_class) < 1:
        raise ValueError("task and sample counts must be positive")
    if noise < 0.0:
        raise ValueError("noise must be nonnegative")
    total_classes = num_tasks * classes_per_task
    prototypes = torch.stack([_class_prototype(index, image_size) for index in range(total_classes)])

    def make_dataset(samples_per_class: int, generator_seed: int) -> TensorDataset:
        generator = torch.Generator().manual_seed(generator_seed)
        images: list[Tensor] = []
        labels: list[Tensor] = []
        for global_class in range(total_classes):
            class_images = prototypes[global_class].unsqueeze(0).repeat(samples_per_class, 1, 1, 1)
            class_images = class_images + noise * torch.randn(
                class_images.shape, generator=generator
            )
            images.append(class_images.clamp(0.0, 1.0))
            labels.append(torch.full((samples_per_class,), global_class, dtype=torch.long))
        inputs = torch.cat(images)
        targets = torch.cat(labels)
        permutation = torch.randperm(inputs.shape[0], generator=generator)
        return TensorDataset(inputs[permutation], targets[permutation])

    train_tasks = split_dataset_by_classes(
        make_dataset(train_samples_per_class, seed), classes_per_task=classes_per_task
    )
    test_tasks = split_dataset_by_classes(
        make_dataset(test_samples_per_class, seed + 1), classes_per_task=classes_per_task
    )
    return train_tasks, test_tasks
