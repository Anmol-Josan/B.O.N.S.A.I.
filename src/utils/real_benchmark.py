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
    output_dir: Path = Path("results/real_benchmark")
    seeds: tuple[int, ...] = (7, 17, 27, 37, 47)
    epochs_per_task: int = 5
    batch_size: int = 128
    learning_rate: float = 0.001
    num_workers: int = 0
    download: bool = False
    device: str = "cpu"
    task_adapter_rank: int = 8
    rewire_strength: float = 0.15
    max_frozen_fraction: float | None = 0.65
    validation_fraction: float = 0.1
    validation_seed: int = 0
    route_strategy: str = "prototype"
    prototype_weight: float = 1.0
    route_head_weight: float = 1.0
    route_calibration_samples: int = 256
    route_memory_samples: int = 64
    route_head_epochs: int = 3
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
) -> nn.Module:
    models = {
        "BONSAI": lambda: BonsaiResNet18(
            num_classes=num_classes,
            task_adapter_rank=task_adapter_rank,
            classes_per_task=classes_per_task,
        ),
        "EWC": lambda: EWC(num_classes=num_classes),
        "SI": lambda: SI(num_classes=num_classes),
        "PackNet": lambda: PackNet(num_classes=num_classes),
        "PNN": lambda: PNN(num_classes=num_classes),
    }
    if method not in models:
        raise ValueError(f"unknown method {method}; expected {tuple(models)}")
    return models[method]()


def _forward(model: nn.Module, inputs: Tensor, task_id: int) -> Tensor:
    if isinstance(model, PNN):
        return model(inputs, task_id=task_id)
    if isinstance(model, BonsaiResNet18):
        return model.task_logits(inputs, task_id=task_id)
    return model(inputs)


def _task_target(
    model: nn.Module, labels: Tensor, task_id: int, classes_per_task: int
) -> Tensor:
    """Use local labels for BONSAI heads and global labels for baselines."""

    if isinstance(model, BonsaiResNet18) and model.classes_per_task is not None:
        return labels - task_id * classes_per_task
    return labels


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
                _forward(model, inputs, task_id),
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
    inputs_list: list[Tensor] = []
    labels_list: list[Tensor] = []
    collected = 0
    for inputs, labels in _loader(GlobalLabelView(dataset), config, shuffle=False):
        inputs_list.append(inputs)
        labels_list.append(labels)
        collected += labels.numel()
        if collected >= config.route_calibration_samples:
            break
    if not inputs_list:
        raise ValueError("cannot calibrate a route from an empty task")
    inputs = torch.cat(inputs_list, dim=0)[: config.route_calibration_samples]
    labels = torch.cat(labels_list, dim=0)[: config.route_calibration_samples]
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
        )
    if isinstance(model, PNN):
        if task_count < 1 or task_count > len(model.columns):
            raise ValueError("task_count is outside the allocated PNN columns")
        was_training = model.training
        model.eval()
        logits = torch.stack(
            [model(inputs, task_id=task_id) for task_id in range(task_count)], dim=1
        )
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
    device = torch.device(config.device)
    evaluation_tasks = tasks if evaluation_tasks is None else evaluation_tasks
    if validation_tasks is not None and len(validation_tasks) != len(tasks):
        raise ValueError("validation_tasks must have the same number of tasks as tasks")
    classes_per_task = len(tasks[0].classes)
    num_classes = len(evaluation_tasks) * len(evaluation_tasks[0].classes)
    model = _make_model(
        method,
        num_classes,
        config.task_adapter_rank,
        classes_per_task=classes_per_task,
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

    for task_id, task in enumerate(tasks):
        if isinstance(model, BonsaiResNet18):
            model.add_task_path()
            model.start_task()
        if isinstance(model, PNN) and task_id > 0:
            model.add_task_column()
        global_task = GlobalLabelView(task)
        training_loader = _loader(global_task, config, shuffle=True)
        optimizer_parameters = (
            list(model.columns[-1].parameters()) if isinstance(model, PNN) else list(model.parameters())
        )
        if isinstance(model, BonsaiResNet18):
            optimizer_parameters = model.task_parameters(task_id)
        optimizer = torch.optim.Adam(optimizer_parameters, lr=config.learning_rate)
        task_start = {
            name: parameter.detach().clone() for name, parameter in model.named_parameters()
        }
        model.train()
        for _ in range(config.epochs_per_task):
            for inputs, labels in training_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(
                    _forward(model, inputs, task_id),
                    _task_target(model, labels, task_id, classes_per_task),
                )
                if isinstance(model, BonsaiResNet18):
                    loss = loss + model.beta * model.kl_loss
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
        if ewc is not None:
            ewc.consolidate(_loader(global_task, config, shuffle=False), max_batches=32)
        if si is not None:
            si.update_synaptic_importance(task_start)
            si.consolidate_task()
        if packnet is not None:
            packnet.prune_by_magnitude(0.8)
        if isinstance(model, BonsaiResNet18):
            route_inputs, route_labels = _route_calibration_batch(task, config, device)
            model.register_task_route(
                task_id,
                route_inputs,
                route_labels,
                classes_per_task=classes_per_task,
            )
            memory_inputs = route_inputs[: config.route_memory_samples].detach().cpu()
            route_memory.append((memory_inputs, task_id))
            _fit_route_head(model, route_memory, config, device)
        if bonsai_manager is not None:
            inputs, labels = next(iter(_loader(global_task, config, shuffle=False)))
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad(set_to_none=True)
            saliency_loss = nn.functional.cross_entropy(
                _forward(model, inputs, task_id),
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
                    predictions = _forward(model, inputs, evaluation_task_id).argmax(dim=1)
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
    """Run all five methods across all configured seeds on a real dataset."""

    raw_training_tasks = load_real_tasks(config, train=True)
    tasks, validation_tasks = split_task_views(
        raw_training_tasks,
        validation_fraction=config.validation_fraction,
        seed=config.validation_seed,
    )
    evaluation_tasks = load_real_tasks(config, train=False)
    methods = ("BONSAI", "EWC", "SI", "PackNet", "PNN")
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
