"""Sequential ResNet-18 benchmark runner for the image task splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.algorithms.baselines import EWC, PNN, SI, PackNet, ResNetBaseline
from src.algorithms.mask_manager import MaskManager
from src.algorithms.rewire import RewireEngine
from src.data.benchmarks import build_split_cifar100, build_split_tiny_imagenet
from src.data.task_splits import ClassIncrementalTask
from src.models.bonsai_resnet import BonsaiResNet18
from src.utils.metrics import average_accuracy, forgetting_measure, parameter_overhead
from src.utils.reproducibility import seed_everything
from src.utils.results import save_records_csv, summarize_records, write_summary_artifacts
from src.utils.visualization import plot_accuracy_curves, plot_mask_sparsity


@dataclass(frozen=True)
class RealBenchmarkConfig:
    dataset: str
    data_root: Path
    methods: tuple[str, ...] = ("BONSAI", "EWC", "SI", "PackNet", "PNN")
    output_dir: Path = Path("results/real_benchmark")
    seeds: tuple[int, ...] = (7, 17, 27, 37, 47)
    epochs_per_task: int = 5
    batch_size: int = 128
    learning_rate: float = 0.001
    num_workers: int = 0
    download: bool = False
    device: str = "cpu"
    task_adapter_rank: int = 8
    adapter_residual_init_std: float = 0.0
    rewire_strength: float = 0.15
    max_frozen_fraction: float | None = 0.65
    validation_fraction: float = 0.1
    validation_seed: int = 0
    route_strategy: str = "compatibility"
    prototype_weight: float = 1.0
    route_head_weight: float = 1.0
    route_calibration_samples: int = 256
    route_memory_samples: int = 64
    route_head_epochs: int = 3
    route_compatibility_epochs: int = 3
    route_hidden_dim: int = 64
    route_discovery_hidden_dim: int = 32
    route_discovery_epochs: int = 10
    route_evidence_epochs: int = 3
    global_loss_weight: float = 0.5
    global_replay_per_task: int = 64
    global_replay_weight: float = 0.5
    global_head_epochs: int = 0
    shared_learning_rate_scale: float = 0.1
    classifier_learning_rate_scale: float = 1.0
    freeze_backbone_bn: bool = False
    global_route_weight: float = 1.0
    route_training_weight: float = 0.0
    route_replay_per_task: int = 16
    feature_replay_weight: float = 0.0
    feature_replay_per_task: int = 16
    local_replay_weight: float = 0.0
    local_replay_per_task: int = 16
    max_samples_per_class: int | None = None


class GlobalLabelView(Dataset[tuple[Any, Tensor]]):
    """Convert a local-label task view into a global-class training view."""

    def __init__(self, task: ClassIncrementalTask) -> None:
        self.task = task

    def __len__(self) -> int:
        return len(self.task)

    def __getitem__(self, index: int) -> tuple[Any, Tensor]:
        inputs, _ = self.task[index]
        return inputs, self.task.global_labels[index]


def resolve_device(device_name: str) -> torch.device:
    """Resolve standard PyTorch devices plus the optional Windows DirectML backend."""

    normalized = device_name.strip().lower()
    if normalized in {"dml", "directml", "privateuseone", "privateuseone:0"}:
        try:
            import torch_directml
        except ImportError as error:  # pragma: no cover - optional runtime
            raise ImportError(
                "DirectML support requires the optional 'torch-directml' package"
            ) from error
        return torch_directml.device()
    return torch.device(device_name)


def split_task_views(
    tasks: list[ClassIncrementalTask],
    validation_fraction: float,
    seed: int = 0,
) -> tuple[list[ClassIncrementalTask], list[ClassIncrementalTask]]:
    """Create class-balanced, index-disjoint train/validation task views."""

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    generator = torch.Generator().manual_seed(seed)
    train_tasks: list[ClassIncrementalTask] = []
    validation_tasks: list[ClassIncrementalTask] = []
    for task in tasks:
        train_positions: list[Tensor] = []
        validation_positions: list[Tensor] = []
        for local_class in range(len(task.classes)):
            class_positions = (task.labels == local_class).nonzero(as_tuple=False).flatten()
            permutation = class_positions[torch.randperm(class_positions.numel(), generator=generator)]
            if validation_fraction == 0.0 or permutation.numel() < 2:
                validation_count = 0
            else:
                validation_count = min(
                    permutation.numel() - 1,
                    max(1, round(permutation.numel() * validation_fraction)),
                )
            validation_positions.append(permutation[:validation_count])
            train_positions.append(permutation[validation_count:])
        train_position_tensor = torch.cat(train_positions) if train_positions else torch.empty(0, dtype=torch.long)
        validation_position_tensor = (
            torch.cat(validation_positions)
            if validation_positions
            else torch.empty(0, dtype=torch.long)
        )

        def make_view(positions: Tensor) -> ClassIncrementalTask:
            return ClassIncrementalTask(
                base_dataset=task.base_dataset,
                indices=[task.indices[int(position)] for position in positions.tolist()],
                global_labels=task.global_labels[positions],
                local_labels=task.labels[positions],
                task_id=task.task_id,
                classes=task.classes,
            )

        train_tasks.append(make_view(train_position_tensor))
        validation_tasks.append(make_view(validation_position_tensor))
    return train_tasks, validation_tasks


def balanced_sample_positions(labels: Tensor, max_samples: int) -> Tensor:
    """Return deterministic round-robin positions covering every class."""

    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    if labels.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    class_positions = [
        (labels == class_id).nonzero(as_tuple=False).flatten()
        for class_id in torch.unique(labels, sorted=True).tolist()
    ]
    selected: list[int] = []
    offset = 0
    while len(selected) < min(max_samples, labels.numel()):
        added = False
        for positions in class_positions:
            if offset < positions.numel() and len(selected) < max_samples:
                selected.append(int(positions[offset]))
                added = True
        if not added:
            break
        offset += 1
    return torch.tensor(selected, dtype=torch.long)


def limit_task_samples(
    tasks: list[ClassIncrementalTask],
    max_samples_per_class: int | None,
    seed: int = 0,
) -> list[ClassIncrementalTask]:
    """Optionally make a deterministic class-balanced subset of each task."""

    if max_samples_per_class is None:
        return tasks
    if max_samples_per_class < 1:
        raise ValueError("max_samples_per_class must be positive")
    generator = torch.Generator().manual_seed(seed)
    limited: list[ClassIncrementalTask] = []
    for task in tasks:
        positions: list[Tensor] = []
        for local_class in range(len(task.classes)):
            class_positions = (task.labels == local_class).nonzero(as_tuple=False).flatten()
            permutation = class_positions[torch.randperm(class_positions.numel(), generator=generator)]
            positions.append(permutation[:max_samples_per_class])
        selected = torch.cat(positions) if positions else torch.empty(0, dtype=torch.long)
        limited.append(
            ClassIncrementalTask(
                base_dataset=task.base_dataset,
                indices=[task.indices[int(position)] for position in selected.tolist()],
                global_labels=task.global_labels[selected],
                local_labels=task.labels[selected],
                task_id=task.task_id,
                classes=task.classes,
            )
        )
    return limited


def load_real_tasks(
    config: RealBenchmarkConfig, train: bool = True
) -> list[ClassIncrementalTask]:
    try:
        from torchvision import transforms
    except ImportError as error:  # pragma: no cover - optional runtime
        raise ImportError("torchvision is required for image benchmarks") from error
    if config.dataset == "cifar100":
        tasks = build_split_cifar100(
            config.data_root,
            train=train,
            download=config.download,
            transform=transforms.ToTensor(),
        )
    elif config.dataset == "tinyimagenet":
        transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
        tasks = build_split_tiny_imagenet(config.data_root, train=train, transform=transform)
    else:
        raise ValueError("dataset must be 'cifar100' or 'tinyimagenet'")
    return limit_task_samples(tasks, config.max_samples_per_class, seed=0 if train else 1)


def _make_model(
    method: str,
    num_classes: int,
    task_adapter_rank: int = 8,
    classes_per_task: int | None = None,
    route_hidden_dim: int = 64,
    route_discovery_hidden_dim: int = 32,
    adapter_residual_init_std: float = 0.0,
) -> nn.Module:
    models = {
        "BONSAI": lambda: BonsaiResNet18(
            num_classes=num_classes,
            task_adapter_rank=task_adapter_rank,
            classes_per_task=classes_per_task,
            route_hidden_dim=route_hidden_dim,
            route_discovery_hidden_dim=route_discovery_hidden_dim,
            adapter_residual_init_std=adapter_residual_init_std,
        ),
        "EWC": lambda: EWC(num_classes=num_classes),
        "SI": lambda: SI(num_classes=num_classes),
        "PackNet": lambda: PackNet(num_classes=num_classes),
        "PNN": lambda: PNN(num_classes=num_classes, task_classes=classes_per_task),
    }
    if method not in models:
        raise ValueError(f"unknown method {method}; expected {tuple(models)}")
    return models[method]()


def _forward(
    model: nn.Module,
    inputs: Tensor,
    task_id: int,
    classes_per_task: int | None = None,
) -> Tensor:
    if isinstance(model, PNN):
        logits = model(inputs, task_id=task_id)
        if classes_per_task is None or logits.shape[-1] == classes_per_task:
            return logits
        start = task_id * classes_per_task
        return logits[:, start : start + classes_per_task]
    if isinstance(model, BonsaiResNet18):
        return model.task_logits(inputs, task_id=task_id)
    return model(inputs)


def _task_target(
    model: nn.Module, labels: Tensor, task_id: int, classes_per_task: int
) -> Tensor:
    """Use local labels for BONSAI heads and global labels for baselines."""

    if isinstance(model, (BonsaiResNet18, PNN)):
        return labels - task_id * classes_per_task
    return labels


def bonsai_training_loss(
    model: BonsaiResNet18,
    inputs: Tensor,
    global_labels: Tensor,
    task_id: int,
    classes_per_task: int,
    config: RealBenchmarkConfig,
) -> Tensor:
    """Train a local task path while retaining a shared global scaffold."""

    local_logits = model.task_logits(inputs, task_id=task_id)
    local_loss = nn.functional.cross_entropy(
        local_logits,
        _task_target(model, global_labels, task_id, classes_per_task),
    )
    loss = local_loss
    if config.global_loss_weight > 0.0:
        loss = loss + config.global_loss_weight * nn.functional.cross_entropy(
            model(inputs), global_labels
        )
    return loss + model.beta * model.kl_loss


def bonsai_route_training_loss(
    model: BonsaiResNet18,
    inputs: Tensor,
    task_id: int,
    route_memory: list[tuple[Tensor, int]],
    config: RealBenchmarkConfig,
    device: torch.device,
) -> Tensor:
    """Train the shared task gate with current examples and bounded replay."""

    if config.route_training_weight < 0.0:
        raise ValueError("route_training_weight must be nonnegative")
    if config.route_replay_per_task < 0:
        raise ValueError("route_replay_per_task must be nonnegative")
    if config.feature_replay_weight < 0.0:
        raise ValueError("feature_replay_weight must be nonnegative")
    if config.feature_replay_per_task < 0:
        raise ValueError("feature_replay_per_task must be nonnegative")
    if config.local_replay_weight < 0.0:
        raise ValueError("local_replay_weight must be nonnegative")
    if config.local_replay_per_task < 0:
        raise ValueError("local_replay_per_task must be nonnegative")
    if config.route_training_weight == 0.0 or model.route_head is None:
        return torch.zeros((), device=device)
    route_inputs = [inputs]
    route_targets = [
        torch.full((inputs.shape[0],), task_id, dtype=torch.long, device=device)
    ]
    if config.route_replay_per_task > 0:
        for memory_inputs, memory_task_id in route_memory:
            replay_inputs = memory_inputs[: config.route_replay_per_task].to(device)
            if replay_inputs.numel() == 0:
                continue
            route_inputs.append(replay_inputs)
            route_targets.append(
                torch.full(
                    (replay_inputs.shape[0],),
                    memory_task_id,
                    dtype=torch.long,
                    device=device,
                )
            )
    logits = model.route_logits(torch.cat(route_inputs, dim=0))[:, : task_id + 1]
    targets = torch.cat(route_targets, dim=0)
    return config.route_training_weight * nn.functional.cross_entropy(logits, targets)


def bonsai_feature_replay_loss(
    model: BonsaiResNet18,
    feature_memory: list[tuple[Tensor, int, Tensor]],
    config: RealBenchmarkConfig,
    device: torch.device,
) -> Tensor:
    """Preserve old task-path geometry with bounded normalized feature replay."""

    if config.feature_replay_weight < 0.0:
        raise ValueError("feature_replay_weight must be nonnegative")
    if config.feature_replay_per_task < 0:
        raise ValueError("feature_replay_per_task must be nonnegative")
    if config.feature_replay_weight == 0.0 or not feature_memory:
        return torch.zeros((), device=device)
    losses: list[Tensor] = []
    for inputs, task_id, target_features in feature_memory:
        replay_inputs = inputs[: config.feature_replay_per_task].to(device)
        replay_targets = target_features[: config.feature_replay_per_task].to(device)
        if replay_inputs.numel() == 0:
            continue
        current_features = model.forward_features(replay_inputs, task_id=task_id)
        current_features = nn.functional.normalize(current_features, dim=-1)
        replay_targets = nn.functional.normalize(replay_targets, dim=-1)
        cosine = (current_features * replay_targets).sum(dim=-1).mean()
        losses.append(torch.ones((), device=device, dtype=cosine.dtype) - cosine)
    if not losses:
        return torch.zeros((), device=device)
    return config.feature_replay_weight * torch.stack(losses).mean()


def bonsai_local_replay_loss(
    model: BonsaiResNet18,
    local_memory: list[tuple[Tensor, Tensor, int]],
    classes_per_task: int,
    config: RealBenchmarkConfig,
    device: torch.device,
) -> Tensor:
    """Replay old task heads while updating the shared representation."""

    if config.local_replay_weight < 0.0:
        raise ValueError("local_replay_weight must be nonnegative")
    if config.local_replay_per_task < 0:
        raise ValueError("local_replay_per_task must be nonnegative")
    if config.local_replay_weight == 0.0 or not local_memory:
        return torch.zeros((), device=device)
    losses: list[Tensor] = []
    for inputs, labels, task_id in local_memory:
        replay_inputs = inputs[: config.local_replay_per_task].to(device)
        replay_labels = labels[: config.local_replay_per_task].to(device)
        if replay_inputs.numel() == 0:
            continue
        local_logits = model.task_logits(replay_inputs, task_id=task_id)
        local_labels = replay_labels.long() - task_id * classes_per_task
        losses.append(nn.functional.cross_entropy(local_logits, local_labels))
    if not losses:
        return torch.zeros((), device=device)
    return config.local_replay_weight * torch.stack(losses).mean()


def _mean_task_loss(
    model: nn.Module,
    dataset: Dataset,
    task_id: int,
    config: RealBenchmarkConfig,
    device: torch.device,
) -> float:
    loader = _loader(GlobalLabelView(dataset), config, shuffle=False)
    total_loss = 0.0
    total_count = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            loss = nn.functional.cross_entropy(
                _forward(model, inputs, task_id, len(dataset.classes)),
                _task_target(model, labels, task_id, len(dataset.classes)),
            )
            total_loss += float(loss.item()) * labels.numel()
            total_count += labels.numel()
    return total_loss / total_count if total_count else 0.0


def _route_calibration_batch(
    dataset: ClassIncrementalTask,
    config: RealBenchmarkConfig,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Collect a bounded calibration batch without materializing a full task."""

    if config.route_calibration_samples < 1:
        raise ValueError("route_calibration_samples must be positive")
    positions = balanced_sample_positions(dataset.labels, config.route_calibration_samples)
    if positions.numel() == 0:
        raise ValueError("cannot calibrate a route from an empty task")
    inputs = torch.stack([dataset[int(position)][0] for position in positions])
    labels = dataset.global_labels[positions]
    return inputs.to(device), labels.to(device)


def _fit_route_head(
    model: BonsaiResNet18,
    route_memory: list[tuple[Tensor, int]],
    config: RealBenchmarkConfig,
    device: torch.device,
) -> None:
    """Fit only the compact task gate on bounded route exemplars."""

    if model.route_head is None or not route_memory:
        return
    if config.route_head_epochs < 1:
        raise ValueError("route_head_epochs must be positive")
    if config.route_memory_samples < 1:
        raise ValueError("route_memory_samples must be positive")
    optimizer = torch.optim.Adam(model.route_head.parameters(), lr=config.learning_rate)
    was_training = model.training
    model.eval()
    for _ in range(config.route_head_epochs):
        optimizer.zero_grad(set_to_none=True)
        losses: list[Tensor] = []
        for inputs, task_id in route_memory:
            with torch.no_grad():
                features = model.forward_features(inputs.to(device))
            losses.append(
                nn.functional.cross_entropy(
                    model.route_logits_from_features(features),
                    torch.full(
                        (features.shape[0],), task_id, dtype=torch.long, device=device
                    ),
                )
            )
        torch.stack(losses).mean().backward()
        optimizer.step()
    if was_training:
        model.train()


def _fit_global_head(
    model: BonsaiResNet18,
    global_memory: list[tuple[Tensor, Tensor]],
    config: RealBenchmarkConfig,
    device: torch.device,
) -> None:
    """Calibrate the global class scaffold on bounded replay features."""

    if not global_memory:
        return
    if config.global_head_epochs < 0:
        raise ValueError("global_head_epochs must be nonnegative")
    if config.global_head_epochs == 0:
        return
    was_training = model.training
    model.eval()
    feature_batches: list[Tensor] = []
    label_batches: list[Tensor] = []
    with torch.no_grad():
        for inputs, labels in global_memory:
            feature_batches.append(model.forward_features(inputs.to(device)).detach())
            label_batches.append(labels.to(device))
    features = torch.cat(feature_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=config.learning_rate)
    for _ in range(config.global_head_epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model.classifier(features), labels)
        loss.backward()
        optimizer.step()
    if was_training:
        model.train()


def _fit_route_compatibility(
    model: BonsaiResNet18,
    route_memory: list[tuple[Tensor, int]],
    config: RealBenchmarkConfig,
    device: torch.device,
) -> None:
    """Fit one-vs-rest path discriminators on bounded route memory."""

    if not route_memory:
        return
    if config.route_compatibility_epochs < 1:
        raise ValueError("route_compatibility_epochs must be positive")
    was_training = model.training
    model.eval()
    with torch.no_grad():
        feature_cache = [
            [
                model.forward_features(inputs.to(device), task_id=task_id).detach()
                for inputs, _ in route_memory
            ]
            for task_id in range(len(model.task_adapters))
        ]
    optimizer = torch.optim.Adam(model.route_compatibility_heads.parameters(), lr=config.learning_rate)
    for _ in range(config.route_compatibility_epochs):
        optimizer.zero_grad(set_to_none=True)
        losses: list[Tensor] = []
        for task_id, task_features in enumerate(feature_cache):
            head = model.route_compatibility_heads[task_id]
            for memory_id, features in enumerate(task_features):
                targets = torch.full(
                    (features.shape[0],),
                    float(memory_id == task_id),
                    dtype=torch.float32,
                    device=device,
                )
                losses.append(
                    nn.functional.binary_cross_entropy_with_logits(
                        head(features).squeeze(-1), targets
                    )
                )
        torch.stack(losses).mean().backward()
        optimizer.step()
    if was_training:
        model.train()


def _refresh_route_state(
    model: BonsaiResNet18,
    calibration_memory: list[tuple[Tensor, Tensor, int]],
    route_memory: list[tuple[Tensor, int]],
    config: RealBenchmarkConfig,
    device: torch.device,
    classes_per_task: int,
) -> None:
    """Recompute route statistics after shared representation changes.

    BONSAI rewires available shared weights between tasks. Any prototypes or
    compatibility-head features computed before that rewire are stale for all
    previously learned paths, so calibration must be replayed after the
    rewire. The bounded image memory keeps this refresh deterministic and
    avoids retaining the full task datasets on the accelerator.
    """

    if not calibration_memory:
        return
    for inputs, labels, task_id in calibration_memory:
        model.register_task_route(
            task_id,
            inputs.to(device),
            labels.to(device),
            classes_per_task=classes_per_task,
        )
    _fit_route_discovery(model, calibration_memory, config, device)
    _fit_route_head(model, route_memory, config, device)
    _fit_route_compatibility(model, route_memory, config, device)
    _fit_route_evidence(model, route_memory, config, device)


def _fit_route_evidence(
    model: BonsaiResNet18,
    route_memory: list[tuple[Tensor, int]],
    config: RealBenchmarkConfig,
    device: torch.device,
) -> None:
    """Fit path-wise calibrators on local classifier evidence."""

    if not route_memory:
        return
    if config.route_evidence_epochs < 1:
        raise ValueError("route_evidence_epochs must be positive")
    if len(model.route_evidence_heads) != len(model.task_adapters):
        return
    was_training = model.training
    model.eval()
    with torch.no_grad():
        feature_cache = []
        for task_id in range(len(model.task_adapters)):
            task_features = []
            for inputs, _ in route_memory:
                features = model.forward_features(inputs.to(device), task_id=task_id)
                local_logits = model.task_heads[task_id](features)
                task_features.append(model.route_evidence_features(local_logits).detach())
            feature_cache.append(task_features)
    optimizer = torch.optim.SGD(
        model.route_evidence_heads.parameters(),
        lr=config.learning_rate * 10.0,
        momentum=0.9,
    )
    for _ in range(config.route_evidence_epochs):
        optimizer.zero_grad(set_to_none=True)
        losses: list[Tensor] = []
        for task_id, task_features in enumerate(feature_cache):
            head = model.route_evidence_heads[task_id]
            for memory_id, features in enumerate(task_features):
                targets = torch.full(
                    (features.shape[0],),
                    float(memory_id == task_id),
                    dtype=torch.float32,
                    device=device,
                )
                losses.append(
                    nn.functional.binary_cross_entropy_with_logits(
                        head(features).squeeze(-1), targets
                    )
                )
        torch.stack(losses).mean().backward()
        optimizer.step()
    if was_training:
        model.train()


def _fit_route_discovery(
    model: BonsaiResNet18,
    calibration_memory: list[tuple[Tensor, Tensor, int]],
    config: RealBenchmarkConfig,
    device: torch.device,
) -> None:
    """Fit the isolated input-only global-class detector."""

    if not calibration_memory:
        return
    if config.route_discovery_epochs < 1:
        raise ValueError("route_discovery_epochs must be positive")
    parameters = model.route_discovery_parameters()
    if not parameters:
        return
    inputs = torch.cat(
        [memory_inputs for memory_inputs, _, _ in calibration_memory], dim=0
    ).to(device)
    targets = torch.cat(
        [
            labels.long() for _, labels, _ in calibration_memory
        ],
        dim=0,
    ).to(device)
    was_training = model.training
    model.eval()
    # SGD avoids the DirectML Adam lerp fallback and is sufficient for this
    # tiny supervised calibration problem.
    optimizer = torch.optim.SGD(parameters, lr=config.learning_rate * 10.0, momentum=0.9)
    for _ in range(config.route_discovery_epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(
            model.route_discovery_class_logits(inputs), targets
        )
        loss.backward()
        optimizer.step()
    if was_training:
        model.train()


def _refresh_feature_memory(
    model: BonsaiResNet18,
    route_memory: list[tuple[Tensor, int]],
    config: RealBenchmarkConfig,
    device: torch.device,
) -> list[tuple[Tensor, int, Tensor]]:
    """Snapshot bounded old-path features after the latest shared rewire."""

    if config.feature_replay_per_task < 0:
        raise ValueError("feature_replay_per_task must be nonnegative")
    if not route_memory or config.feature_replay_per_task == 0:
        return []
    was_training = model.training
    model.eval()
    memory: list[tuple[Tensor, int, Tensor]] = []
    with torch.no_grad():
        for inputs, task_id in route_memory:
            selected_inputs = inputs[: config.feature_replay_per_task].to(device)
            if selected_inputs.numel() == 0:
                continue
            features = model.forward_features(selected_inputs, task_id=task_id)
            memory.append(
                (
                    selected_inputs.detach().cpu(),
                    task_id,
                    features.detach().cpu(),
                )
            )
    if was_training:
        model.train()
    return memory


def _task_free_predict(
    model: nn.Module,
    inputs: Tensor,
    task_count: int,
    classes_per_task: int,
    config: RealBenchmarkConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Route an image without a task ID for BONSAI and PNN baselines."""

    if isinstance(model, BonsaiResNet18):
        return model.predict_task_free(
            inputs,
            classes_per_task=classes_per_task,
            route_strategy=config.route_strategy,
            prototype_weight=config.prototype_weight,
            route_head_weight=config.route_head_weight,
            global_route_weight=config.global_route_weight,
        )
    if isinstance(model, PNN):
        if task_count < 1 or task_count > len(model.columns):
            raise ValueError("task_count is outside the allocated PNN columns")
        was_training = model.training
        model.eval()
        logits = torch.stack(
            [model(inputs, task_id=task_id) for task_id in range(task_count)], dim=1
        )
        if logits.shape[-1] == classes_per_task:
            local_logits = logits
        else:
            local_logits = torch.stack(
                [
                    logits[:, task_id, task_id * classes_per_task : (task_id + 1) * classes_per_task]
                    for task_id in range(task_count)
                ],
                dim=1,
            )
        probabilities = local_logits.softmax(dim=-1)
        entropies = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        selected_tasks = entropies.argmin(dim=1)
        batch_indices = torch.arange(inputs.shape[0], device=inputs.device)
        local_predictions = local_logits[batch_indices, selected_tasks].argmax(dim=1)
        predictions = selected_tasks * classes_per_task + local_predictions
        if was_training:
            model.train()
        return predictions, selected_tasks, entropies

    was_training = model.training
    model.eval()
    logits = model(inputs)
    predictions = logits.argmax(dim=1)
    selected_tasks = torch.zeros(inputs.shape[0], dtype=torch.long, device=inputs.device)
    entropies = torch.zeros(inputs.shape[0], 1, device=inputs.device)
    if was_training:
        model.train()
    return predictions, selected_tasks, entropies


def _loader(dataset: Dataset, config: RealBenchmarkConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=False,
    )


def _freeze_backbone_batchnorm(model: BonsaiResNet18) -> None:
    """Freeze shared BatchNorm running statistics while retaining affine grads."""

    for module in model.backbone.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def run_real_method(
    method: str,
    tasks: list[ClassIncrementalTask],
    seed: int,
    config: RealBenchmarkConfig,
    evaluation_tasks: list[ClassIncrementalTask] | None = None,
    validation_tasks: list[ClassIncrementalTask] | None = None,
) -> tuple[dict, list[list[float]], dict[int, dict[str, Tensor]]]:
    """Train one method sequentially and return metrics, history, and masks."""

    seed_everything(seed)
    device = resolve_device(config.device)
    evaluation_tasks = tasks if evaluation_tasks is None else evaluation_tasks
    if validation_tasks is not None and len(validation_tasks) != len(tasks):
        raise ValueError("validation_tasks must have the same number of tasks as tasks")
    if config.global_loss_weight < 0.0:
        raise ValueError("global_loss_weight must be nonnegative")
    if config.global_replay_per_task < 0:
        raise ValueError("global_replay_per_task must be nonnegative")
    if config.global_replay_weight < 0.0:
        raise ValueError("global_replay_weight must be nonnegative")
    if config.global_head_epochs < 0:
        raise ValueError("global_head_epochs must be nonnegative")
    if config.shared_learning_rate_scale < 0.0:
        raise ValueError("shared_learning_rate_scale must be nonnegative")
    if config.classifier_learning_rate_scale < 0.0:
        raise ValueError("classifier_learning_rate_scale must be nonnegative")
    if config.route_training_weight < 0.0:
        raise ValueError("route_training_weight must be nonnegative")
    if config.route_replay_per_task < 0:
        raise ValueError("route_replay_per_task must be nonnegative")
    classes_per_task = len(tasks[0].classes)
    num_classes = len(evaluation_tasks) * len(evaluation_tasks[0].classes)
    model = _make_model(
        method,
        num_classes,
        config.task_adapter_rank,
        classes_per_task=classes_per_task,
        route_hidden_dim=config.route_hidden_dim,
        route_discovery_hidden_dim=config.route_discovery_hidden_dim,
        adapter_residual_init_std=config.adapter_residual_init_std,
    ).to(device)
    initial_parameters = sum(parameter.numel() for parameter in model.parameters())
    ewc = model if isinstance(model, EWC) else None
    si = model if isinstance(model, SI) else None
    packnet = model if isinstance(model, PackNet) else None
    bonsai_manager = (
        MaskManager(
            saliency_quantile=0.8,
            max_frozen_fraction=config.max_frozen_fraction,
        )
        if method == "BONSAI"
        else None
    )
    accuracy_history: list[list[float]] = []
    task_free_accuracy_history: list[list[float]] = []
    mask_history: dict[int, dict[str, Tensor]] = {}
    route_memory: list[tuple[Tensor, int]] = []
    route_calibration_memory: list[tuple[Tensor, Tensor, int]] = []
    feature_memory: list[tuple[Tensor, int, Tensor]] = []
    local_memory: list[tuple[Tensor, Tensor, int]] = []
    global_replay: list[tuple[Tensor, Tensor]] = []

    for task_id, task in enumerate(tasks):
        if isinstance(model, BonsaiResNet18):
            model.add_task_path()
            model.start_task()
        if isinstance(model, PNN) and task_id > 0:
            # PNN allocates columns lazily; move each new column to the selected
            # backend because ``model.to(device)`` ran before the column existed.
            model.add_task_column().to(device)
        global_task = GlobalLabelView(task)
        training_loader = _loader(global_task, config, shuffle=True)
        optimizer_parameters = (
            list(model.columns[-1].parameters()) if isinstance(model, PNN) else list(model.parameters())
        )
        if isinstance(model, BonsaiResNet18):
            optimizer_parameters = model.task_parameter_groups(
                task_id,
                learning_rate=config.learning_rate,
                shared_learning_rate_scale=config.shared_learning_rate_scale,
                classifier_learning_rate_scale=config.classifier_learning_rate_scale,
            )
        optimizer = torch.optim.Adam(optimizer_parameters, lr=config.learning_rate)
        task_start = {
            name: parameter.detach().clone() for name, parameter in model.named_parameters()
        }
        model.train()
        if isinstance(model, BonsaiResNet18) and config.freeze_backbone_bn and task_id > 0:
            _freeze_backbone_batchnorm(model)
        if isinstance(model, PNN):
            for previous_column in model.columns[:-1]:
                previous_column.eval()
            model.columns[-1].train()
        for epoch_index in range(config.epochs_per_task):
            for batch_index, (inputs, labels) in enumerate(training_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                if isinstance(model, BonsaiResNet18):
                    loss = bonsai_training_loss(
                        model, inputs, labels, task_id, classes_per_task, config
                    )
                    loss = loss + bonsai_route_training_loss(
                        model, inputs, task_id, route_memory, config, device
                    )
                    loss = loss + bonsai_feature_replay_loss(
                        model, feature_memory, config, device
                    )
                    loss = loss + bonsai_local_replay_loss(
                        model, local_memory, classes_per_task, config, device
                    )
                    if global_replay and config.global_replay_weight > 0.0:
                        replay_inputs, replay_labels = global_replay[
                            (batch_index + epoch_index) % len(global_replay)
                        ]
                        replay_inputs = replay_inputs.to(device)
                        replay_labels = replay_labels.to(device)
                        loss = loss + (
                            config.global_loss_weight
                            * config.global_replay_weight
                            * nn.functional.cross_entropy(model(replay_inputs), replay_labels)
                        )
                else:
                    loss = nn.functional.cross_entropy(
                        _forward(model, inputs, task_id, classes_per_task),
                        _task_target(model, labels, task_id, classes_per_task),
                    )
                if ewc is not None:
                    loss = loss + ewc.ewc_penalty()
                if si is not None:
                    loss = loss + si.si_penalty()
                loss.backward()
                if packnet is not None:
                    packnet.apply_gradient_masks()
                optimizer.step()
            if isinstance(model, BonsaiResNet18):
                validation_task = task if validation_tasks is None else validation_tasks[task_id]
                validation_loss = _mean_task_loss(model, validation_task, task_id, config, device)
                if model.record_validation_loss(validation_loss):
                    optimizer = torch.optim.Adam(
                        model.task_parameters(task_id), lr=config.learning_rate
                    )
                model.train()
                if config.freeze_backbone_bn and task_id > 0:
                    _freeze_backbone_batchnorm(model)
        if ewc is not None:
            ewc.consolidate(_loader(global_task, config, shuffle=False), max_batches=32)
        if si is not None:
            si.update_synaptic_importance(task_start)
            si.consolidate_task()
        if packnet is not None:
            packnet.prune_by_magnitude(0.8)
        if isinstance(model, BonsaiResNet18):
            route_inputs, route_labels = _route_calibration_batch(task, config, device)
            memory_inputs = route_inputs[: config.route_memory_samples].detach().cpu()
            route_memory.append((memory_inputs, task_id))
            route_calibration_memory.append(
                (route_inputs.detach().cpu(), route_labels.detach().cpu(), task_id)
            )
            local_memory.append(
                (route_inputs.detach().cpu(), route_labels.detach().cpu(), task_id)
            )
            if config.global_replay_per_task > 0:
                global_replay.append(
                    (
                        route_inputs[: config.global_replay_per_task].detach().cpu(),
                        route_labels[: config.global_replay_per_task].detach().cpu(),
                    )
                )
                _fit_global_head(model, global_replay, config, device)
        if bonsai_manager is not None:
            inputs, labels = next(iter(_loader(global_task, config, shuffle=False)))
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad(set_to_none=True)
            saliency_loss = nn.functional.cross_entropy(
                _forward(model, inputs, task_id, classes_per_task),
                _task_target(model, labels, task_id, classes_per_task),
            )
            saliency = bonsai_manager.compute_saliency(model, saliency_loss)
            masks = bonsai_manager.build_critical_masks(
                saliency,
                excluded_masks=bonsai_manager.critical_masks,
                total_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            )
            bonsai_manager.freeze_critical(model, masks)
            mask_history[task_id + 1] = {
                name: value.detach().cpu().clone() for name, value in bonsai_manager.critical_masks.items()
            }
            if task_id + 1 < len(tasks):
                excluded_names = {
                    name
                    for name, _ in model.named_parameters()
                    if name.startswith("task_adapters.")
                    or name.startswith("task_stage_adapters.")
                    or name.startswith("task_heads.")
                    or name.startswith("route_compatibility_heads.")
                    or name.startswith("route_head.")
                    or name.startswith("route_discovery_encoder.")
                    or name.startswith("route_discovery_head.")
                    or name.startswith("route_evidence_heads.")
                    or name.startswith("classifier.")
                    or name.startswith("vib.logvar_layer.")
                }
                RewireEngine(
                    strategy="orthogonal",
                    seed=seed + task_id,
                    strength=config.rewire_strength,
                ).rewire(
                    model,
                    bonsai_manager.critical_masks,
                    exclude_names=excluded_names,
                )
        if isinstance(model, BonsaiResNet18):
            _refresh_route_state(
                model,
                route_calibration_memory,
                route_memory,
                config,
                device,
                classes_per_task,
            )
            feature_memory = _refresh_feature_memory(
                model, route_memory, config, device
            )
        current_accuracies: list[float] = []
        task_free_accuracies: list[float] = []
        route_correct = 0
        route_count = 0
        for evaluation_task_id in range(task_id + 1):
            evaluation_loader = _loader(
                GlobalLabelView(evaluation_tasks[evaluation_task_id]), config, shuffle=False
            )
            correct = 0
            count = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in evaluation_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    predictions = _forward(
                        model, inputs, evaluation_task_id, classes_per_task
                    ).argmax(dim=1)
                    target = _task_target(
                        model, labels, evaluation_task_id, classes_per_task
                    )
                    correct += int((predictions == target).sum().item())
                    count += labels.numel()
            current_accuracies.append(correct / count if count else 0.0)
            task_free_correct = 0
            task_free_count = 0
            task_free_route_correct = 0
            with torch.no_grad():
                for inputs, labels in evaluation_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    predictions, selected_tasks, _ = _task_free_predict(
                        model,
                        inputs,
                        task_count=task_id + 1,
                        classes_per_task=classes_per_task,
                        config=config,
                    )
                    task_free_correct += int((predictions == labels).sum().item())
                    task_free_count += labels.numel()
                    task_free_route_correct += int(
                        (selected_tasks == evaluation_task_id).sum().item()
                    )
            task_free_accuracies.append(
                task_free_correct / task_free_count if task_free_count else 0.0
            )
            route_correct += task_free_route_correct
            route_count += task_free_count
        accuracy_history.append(current_accuracies)
        task_free_accuracy_history.append(task_free_accuracies)

    record = {
        "dataset": config.dataset,
        "method": method,
        "seed": seed,
        "average_accuracy": average_accuracy(accuracy_history[-1]),
        "task_free_average_accuracy": average_accuracy(task_free_accuracy_history[-1]),
        "forgetting": forgetting_measure(accuracy_history),
        "task_free_forgetting": forgetting_measure(task_free_accuracy_history),
        "parameter_overhead_percent": parameter_overhead(
            initial_parameters, sum(parameter.numel() for parameter in model.parameters())
        ),
    }
    if isinstance(model, (BonsaiResNet18, PNN)):
        record["task_free_route_accuracy"] = route_correct / route_count if route_count else 0.0
    record["task_free_accuracy_curve"] = [
        average_accuracy(row) for row in task_free_accuracy_history
    ]
    return record, accuracy_history, mask_history


def run_real_suite(config: RealBenchmarkConfig) -> tuple[list[dict], list[dict]]:
    """Run the selected methods across all configured seeds on a real dataset."""

    available_methods = ("BONSAI", "EWC", "SI", "PackNet", "PNN")
    if not config.methods:
        raise ValueError("methods must contain at least one benchmark method")
    unknown_methods = tuple(method for method in config.methods if method not in available_methods)
    if unknown_methods:
        raise ValueError(f"unknown benchmark methods: {unknown_methods}")

    raw_training_tasks = load_real_tasks(config, train=True)
    tasks, validation_tasks = split_task_views(
        raw_training_tasks,
        validation_fraction=config.validation_fraction,
        seed=config.validation_seed,
    )
    evaluation_tasks = load_real_tasks(config, train=False)
    methods = config.methods
    records: list[dict] = []
    histories: dict[str, list[list[list[float]]]] = {method: [] for method in methods}
    first_masks: dict[int, dict[str, Tensor]] = {}
    for method in methods:
        for seed in config.seeds:
            record, history, masks = run_real_method(
                method,
                tasks,
                seed,
                config,
                evaluation_tasks=evaluation_tasks,
                validation_tasks=validation_tasks,
            )
            records.append(record)
            histories[method].append(history)
            if method == "BONSAI" and not first_masks:
                first_masks = masks
    summaries = summarize_records(records)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_records_csv(output_dir / "runs.csv", records)
    write_summary_artifacts(output_dir / "summary.json", summaries)
    curves = {}
    for method, runs in histories.items():
        max_tasks = max(len(history) for history in runs)
        curves[method] = [
            sum(history[task_id][-1] for history in runs if len(history) > task_id) / len(runs)
            for task_id in range(max_tasks)
        ]
    plot_accuracy_curves(curves, output_dir / "accuracy_curves.png", title=f"{config.dataset} accuracy")
    if first_masks:
        plot_mask_sparsity(first_masks, output_dir / "mask_sparsity.png")
    return records, summaries
