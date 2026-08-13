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


class GlobalLabelView(Dataset[tuple[Any, Tensor]]):
    """Convert a local-label task view into a global-class training view."""

    def __init__(self, task: ClassIncrementalTask) -> None:
        self.task = task

    def __len__(self) -> int:
        return len(self.task)

    def __getitem__(self, index: int) -> tuple[Any, Tensor]:
        inputs, _ = self.task[index]
        return inputs, self.task.global_labels[index]


def load_real_tasks(config: RealBenchmarkConfig) -> list[ClassIncrementalTask]:
    try:
        from torchvision import transforms
    except ImportError as error:  # pragma: no cover - optional runtime
        raise ImportError("torchvision is required for image benchmarks") from error
    if config.dataset == "cifar100":
        return build_split_cifar100(
            config.data_root,
            train=True,
            download=config.download,
            transform=transforms.ToTensor(),
        )
    if config.dataset == "tinyimagenet":
        transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
        return build_split_tiny_imagenet(config.data_root, train=True, transform=transform)
    raise ValueError("dataset must be 'cifar100' or 'tinyimagenet'")


def _make_model(method: str, num_classes: int) -> nn.Module:
    models = {
        "BONSAI": lambda: ResNetBaseline(num_classes=num_classes),
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
    return model(inputs)


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
) -> tuple[dict, list[list[float]], dict[int, dict[str, Tensor]]]:
    """Train one method sequentially and return metrics, history, and masks."""

    seed_everything(seed)
    device = torch.device(config.device)
    num_classes = len(tasks) * len(tasks[0].classes)
    model = _make_model(method, num_classes).to(device)
    initial_parameters = sum(parameter.numel() for parameter in model.parameters())
    ewc = model if isinstance(model, EWC) else None
    si = model if isinstance(model, SI) else None
    packnet = model if isinstance(model, PackNet) else None
    bonsai_manager = MaskManager(saliency_quantile=0.8) if method == "BONSAI" else None
    accuracy_history: list[list[float]] = []
    mask_history: dict[int, dict[str, Tensor]] = {}

    for task_id, task in enumerate(tasks):
        if isinstance(model, PNN) and task_id > 0:
            model.add_task_column()
        global_task = GlobalLabelView(task)
        training_loader = _loader(global_task, config, shuffle=True)
        optimizer_parameters = (
            list(model.columns[-1].parameters()) if isinstance(model, PNN) else list(model.parameters())
        )
        optimizer = torch.optim.Adam(optimizer_parameters, lr=config.learning_rate)
        task_start = {
            name: parameter.detach().clone() for name, parameter in model.named_parameters()
        }
        model.train()
        for _ in range(config.epochs_per_task):
            for inputs, labels in training_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(_forward(model, inputs, task_id), labels)
                if ewc is not None:
                    loss = loss + ewc.ewc_penalty()
                if si is not None:
                    loss = loss + si.si_penalty()
                loss.backward()
                if packnet is not None:
                    packnet.apply_gradient_masks()
                optimizer.step()
        if ewc is not None:
            ewc.consolidate(_loader(global_task, config, shuffle=False), max_batches=32)
        if si is not None:
            si.update_synaptic_importance(task_start)
            si.consolidate_task()
        if packnet is not None:
            packnet.prune_by_magnitude(0.8)
        if bonsai_manager is not None:
            inputs, labels = next(iter(_loader(global_task, config, shuffle=False)))
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad(set_to_none=True)
            saliency_loss = nn.functional.cross_entropy(model(inputs), labels)
            saliency = bonsai_manager.compute_saliency(model, saliency_loss)
            masks = bonsai_manager.build_critical_masks(saliency)
            bonsai_manager.freeze_critical(model, masks)
            mask_history[task_id + 1] = {
                name: value.detach().cpu().clone() for name, value in bonsai_manager.critical_masks.items()
            }
            if task_id + 1 < len(tasks):
                RewireEngine(strategy="orthogonal", seed=seed + task_id).rewire(
                    model, bonsai_manager.critical_masks
                )
        current_accuracies: list[float] = []
        for evaluation_task_id in range(task_id + 1):
            evaluation_loader = _loader(GlobalLabelView(tasks[evaluation_task_id]), config, shuffle=False)
            correct = 0
            count = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in evaluation_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    predictions = _forward(model, inputs, evaluation_task_id).argmax(dim=1)
                    correct += int((predictions == labels).sum().item())
                    count += labels.numel()
            current_accuracies.append(correct / count if count else 0.0)
        accuracy_history.append(current_accuracies)

    record = {
        "dataset": config.dataset,
        "method": method,
        "seed": seed,
        "average_accuracy": average_accuracy(accuracy_history[-1]),
        "forgetting": forgetting_measure(accuracy_history),
        "parameter_overhead_percent": parameter_overhead(
            initial_parameters, sum(parameter.numel() for parameter in model.parameters())
        ),
    }
    return record, accuracy_history, mask_history


def run_real_suite(config: RealBenchmarkConfig) -> tuple[list[dict], list[dict]]:
    """Run all five methods across all configured seeds on a real dataset."""

    tasks = load_real_tasks(config)
    methods = ("BONSAI", "EWC", "SI", "PackNet", "PNN")
    records: list[dict] = []
    histories: dict[str, list[list[list[float]]]] = {method: [] for method in methods}
    first_masks: dict[int, dict[str, Tensor]] = {}
    for method in methods:
        for seed in config.seeds:
            record, history, masks = run_real_method(method, tasks, seed, config)
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

