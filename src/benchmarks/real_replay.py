"""Real-image replay benchmark with matched capacity and explicit budgets.

This module is deliberately mechanism-focused.  ER, DER++, ER-ACE, and ATGFR
all use the same compact convolutional backbone, optimizer, task stream, and
number of stored exemplars.  Only the replay target and loss differ.  The
result is not presented as a universal leaderboard; it closes the specific
apple-to-apples evidence gap identified in the review.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.task_splits import ClassIncrementalTask
from src.utils.metrics import average_accuracy, forgetting_measure
from src.utils.real_benchmark import RealBenchmarkConfig, load_real_tasks
from src.utils.reproducibility import seed_everything


VALID_METHODS = ("ER", "DER++", "ER-ACE", "ATGFR")


@dataclass(frozen=True)
class RealReplayConfig:
    data_root: Path = Path("data")
    dataset: str = "cifar100"
    methods: tuple[str, ...] = VALID_METHODS
    seeds: tuple[int, ...] = (7, 17, 27)
    order_count: int = 3
    epochs_per_task: int = 1
    batch_size: int = 64
    learning_rate: float = 1e-3
    memory_per_task: int = 20
    train_samples_per_class: int = 16
    test_samples_per_class: int = 40
    replay_weight: float = 1.0
    distill_weight: float = 1.0
    feature_weight: float = 0.25
    temperature: float = 2.0
    relation_floor: float = 0.35
    drift_budget: float = 0.05
    thermostat_gain: float = 2.0
    device: str = "cpu"


class CompactCIFARNet(nn.Module):
    """A small, fixed-capacity CIFAR backbone used by every method."""

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.feature = nn.Linear(128, 128)
        self.classifier = nn.Linear(128, num_classes)

    def forward_with_features(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        features = F.relu(self.feature(self.encoder(inputs)))
        return self.classifier(features), features

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_with_features(inputs)[0]


@dataclass
class ReplayRecord:
    task_id: int
    inputs: Tensor
    labels: Tensor
    logits: Tensor | None
    features: Tensor | None
    prototype: Tensor

    @property
    def scalar_elements(self) -> int:
        values = self.inputs.numel() + self.labels.numel() + self.prototype.numel()
        if self.logits is not None:
            values += self.logits.numel()
        if self.features is not None:
            values += self.features.numel()
        return int(values)


def _materialize(task: ClassIncrementalTask) -> TensorDataset:
    inputs = torch.stack([task[index][0] for index in range(len(task))]).float()
    return TensorDataset(inputs, task.global_labels.clone().long())


def _balanced_positions(labels: Tensor, limit: int) -> Tensor:
    if limit < 1:
        raise ValueError("memory_per_task must be positive")
    by_class = {
        int(label): (labels == label).nonzero(as_tuple=False).flatten().tolist()
        for label in torch.unique(labels, sorted=True).tolist()
    }
    selected: list[int] = []
    offset = 0
    while len(selected) < min(limit, labels.numel()):
        added = False
        for positions in by_class.values():
            if offset < len(positions) and len(selected) < limit:
                selected.append(positions[offset])
                added = True
        if not added:
            break
        offset += 1
    return torch.tensor(selected, dtype=torch.long)


def _task_order(task_count: int, seed: int, order_id: int) -> list[int]:
    if order_id == 0:
        return list(range(task_count))
    generator = torch.Generator().manual_seed(seed + 1009 * order_id)
    return torch.randperm(task_count, generator=generator).tolist()


def _replay_loss(
    method: str,
    model: CompactCIFARNet,
    memory: list[ReplayRecord],
    current_features: Tensor,
    current_task_id: int,
    config: RealReplayConfig,
    seen_classes: Tensor,
    device: torch.device,
) -> Tensor:
    if not memory:
        return current_features.new_zeros(())
    terms: list[Tensor] = []
    temperature = config.temperature
    current_prototype = current_features.detach().mean(dim=0)
    seen_mask = torch.zeros(100, dtype=torch.bool, device=device)
    seen_mask[seen_classes.to(device)] = True
    for record in memory:
        inputs = record.inputs.to(device)
        labels = record.labels.to(device)
        logits, features = model.forward_with_features(inputs)
        replay_logits = logits
        if method == "ER-ACE":
            replay_logits = logits.masked_fill(~seen_mask.unsqueeze(0), -1e4)
        term = F.cross_entropy(replay_logits, labels)
        if method == "DER++" and record.logits is not None:
            term = term + config.distill_weight * F.mse_loss(logits, record.logits.to(device))
        if method == "ATGFR":
            relation = config.relation_floor + (1.0 - config.relation_floor) * (
                F.cosine_similarity(current_prototype.unsqueeze(0), record.prototype.to(device).unsqueeze(0))
                .clamp_min(0.0)
                .item()
            )
            if record.logits is not None:
                distill = F.kl_div(
                    F.log_softmax(logits / temperature, dim=-1),
                    F.softmax(record.logits.to(device) / temperature, dim=-1),
                    reduction="batchmean",
                ) * temperature**2
                term = term + config.distill_weight * distill
            if record.features is not None:
                drift = float(
                    (features.detach() - record.features.to(device)).square().mean()
                    / record.features.to(device).square().mean().clamp_min(1e-6)
                )
                thermostat = min(
                    8.0,
                    1.0 + config.thermostat_gain * max(0.0, drift / config.drift_budget - 1.0),
                )
                term = term + config.feature_weight * F.mse_loss(
                    F.normalize(features, dim=-1),
                    F.normalize(record.features.to(device), dim=-1),
                )
                relation *= thermostat
            term = relation * term
        terms.append(term)
    return config.replay_weight * torch.stack(terms).mean()


@torch.no_grad()
def _consolidate(
    method: str,
    model: CompactCIFARNet,
    dataset: TensorDataset,
    task_id: int,
    config: RealReplayConfig,
    device: torch.device,
) -> ReplayRecord:
    inputs, labels = dataset.tensors
    indices = _balanced_positions(labels, config.memory_per_task)
    selected_inputs = inputs[indices].cpu()
    selected_labels = labels[indices].cpu()
    logits, features = model.forward_with_features(selected_inputs.to(device))
    keep_logits = method in {"DER++", "ATGFR"}
    keep_features = method == "ATGFR"
    return ReplayRecord(
        task_id=task_id,
        inputs=selected_inputs,
        labels=selected_labels,
        logits=logits.detach().cpu() if keep_logits else None,
        features=features.detach().cpu() if keep_features else None,
        prototype=features.detach().cpu().mean(dim=0),
    )


@torch.no_grad()
def _evaluate(
    model: CompactCIFARNet,
    tasks: list[TensorDataset],
    device: torch.device,
    batch_size: int,
) -> list[float]:
    model.eval()
    values: list[float] = []
    for dataset in tasks:
        correct = 0
        total = 0
        for inputs, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            predictions = model(inputs.to(device)).argmax(dim=1).cpu()
            correct += int((predictions == labels).sum())
            total += labels.numel()
        values.append(correct / total if total else 0.0)
    return values


def run_real_replay_method(
    method: str,
    train_tasks: list[ClassIncrementalTask],
    test_tasks: list[ClassIncrementalTask],
    order: list[int],
    seed: int,
    config: RealReplayConfig,
) -> dict:
    if method not in VALID_METHODS:
        raise ValueError(f"unknown method {method}; expected {VALID_METHODS}")
    seed_everything(seed)
    device = torch.device(config.device)
    ordered_train = [_materialize(train_tasks[index]) for index in order]
    ordered_test = [_materialize(test_tasks[index]) for index in order]
    model = CompactCIFARNet(num_classes=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    memory: list[ReplayRecord] = []
    accuracy_history: list[list[float]] = []
    start = perf_counter()
    step_count = 0
    task_seconds: list[float] = []
    seen_classes = torch.empty(0, dtype=torch.long)
    for task_id, dataset in enumerate(ordered_train):
        task_start = perf_counter()
        task_labels = dataset.tensors[1]
        seen_classes = torch.unique(torch.cat((seen_classes, task_labels)), sorted=True)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        model.train()
        for _ in range(config.epochs_per_task):
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, features = model.forward_with_features(inputs)
                if method == "ER-ACE":
                    mask = torch.zeros(100, dtype=torch.bool, device=device)
                    mask[seen_classes.to(device)] = True
                    logits = logits.masked_fill(~mask.unsqueeze(0), -1e4)
                loss = F.cross_entropy(logits, labels)
                loss = loss + _replay_loss(
                    method, model, memory, features, task_id, config, seen_classes, device
                )
                loss.backward()
                optimizer.step()
                step_count += 1
        memory.append(_consolidate(method, model, dataset, task_id, config, device))
        accuracy_history.append(_evaluate(model, ordered_test[: task_id + 1], device, config.batch_size))
        task_seconds.append(perf_counter() - task_start)
    final_accuracy = average_accuracy(accuracy_history[-1])
    memory_elements = sum(record.scalar_elements for record in memory)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "dataset": config.dataset,
        "method": method,
        "seed": seed,
        "order_id": None,
        "task_order": order,
        "num_tasks": len(order),
        "train_samples_per_class": config.train_samples_per_class,
        "test_samples_per_class": config.test_samples_per_class,
        "memory_per_task": config.memory_per_task,
        "memory_images": len(memory) * config.memory_per_task,
        "memory_scalar_elements": memory_elements,
        "parameter_count": parameter_count,
        "parameter_overhead_percent": 0.0,
        "average_accuracy": final_accuracy,
        "forgetting": forgetting_measure(accuracy_history),
        "task_free_average_accuracy": final_accuracy,
        "task_free_forgetting": forgetting_measure(accuracy_history),
        "wall_time_seconds": perf_counter() - start,
        "task_seconds": task_seconds,
        "train_steps": step_count,
        "accuracy_history": accuracy_history,
    }


def run_real_replay_suite(config: RealReplayConfig) -> list[dict]:
    """Run all methods over identity plus predeclared shuffled task orders."""

    if config.dataset != "cifar100":
        raise ValueError("the matched real replay suite currently supports CIFAR-100")
    if config.order_count < 1 or config.epochs_per_task < 1:
        raise ValueError("order_count and epochs_per_task must be positive")
    unknown = tuple(method for method in config.methods if method not in VALID_METHODS)
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    train_config = RealBenchmarkConfig(
        dataset="cifar100",
        data_root=config.data_root,
        max_samples_per_class=config.train_samples_per_class,
        device=config.device,
    )
    test_config = RealBenchmarkConfig(
        dataset="cifar100",
        data_root=config.data_root,
        max_samples_per_class=config.test_samples_per_class,
        device=config.device,
    )
    train_tasks = load_real_tasks(train_config, train=True)
    test_tasks = load_real_tasks(test_config, train=False)
    if len(train_tasks) != 10 or len(test_tasks) != 10:
        raise ValueError("Split-CIFAR-100 must contain ten tasks")
    records: list[dict] = []
    for seed in config.seeds:
        for order_id in range(config.order_count):
            order = _task_order(len(train_tasks), int(seed), order_id)
            for method in config.methods:
                record = run_real_replay_method(
                    method, train_tasks, test_tasks, order, int(seed), config
                )
                record["order_id"] = order_id
                records.append(record)
    return records


def summarize_real_replay(records: Iterable[dict]) -> list[dict]:
    """Aggregate by method and task order without dropping any run."""

    grouped: dict[tuple[str, int], list[dict]] = {}
    for record in records:
        grouped.setdefault((record["method"], int(record["order_id"])), []).append(record)
    summaries: list[dict] = []
    for (method, order_id), rows in sorted(grouped.items()):
        summary = {
            "method": method,
            "order_id": order_id,
            "runs": len(rows),
            "average_accuracy_mean": float(torch.tensor([row["average_accuracy"] for row in rows]).mean()),
            "average_accuracy_std": float(torch.tensor([row["average_accuracy"] for row in rows]).std(unbiased=False)) if len(rows) > 1 else 0.0,
            "forgetting_mean": float(torch.tensor([row["forgetting"] for row in rows]).mean()),
            "forgetting_std": float(torch.tensor([row["forgetting"] for row in rows]).std(unbiased=False)) if len(rows) > 1 else 0.0,
            "wall_time_seconds_mean": float(torch.tensor([row["wall_time_seconds"] for row in rows]).mean()),
            "memory_scalar_elements": rows[0]["memory_scalar_elements"],
        }
        summaries.append(summary)
    return summaries


def config_as_dict(config: RealReplayConfig) -> dict:
    payload = asdict(config)
    payload["data_root"] = str(config.data_root)
    return payload
