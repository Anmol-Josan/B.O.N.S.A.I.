"""Capacity-matched classical baselines on the current BONSAI prediction core.

This module implements the comparison between the current compact VIB/MLP
backbone and standard continual-learning controls under one shared stream,
optimizer schedule, and parameter accounting convention.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.bonsai.model import BONSAIModel
from src.data.synthetic_images import make_synthetic_image_tasks
from src.utils.metrics import average_accuracy, forgetting_measure


CURRENT_BASELINE_METHODS = (
    "current_ewc",
    "current_si",
    "current_packnet",
    "current_pnn",
)


@dataclass
class _EWCState:
    references: dict[str, Tensor]
    fisher: dict[str, Tensor]
    strength: float = 1.0

    @classmethod
    def empty(cls, strength: float = 1.0) -> "_EWCState":
        return cls(references={}, fisher={}, strength=strength)

    def penalty(self, model: nn.Module) -> Tensor:
        if not self.references:
            return next(model.parameters()).new_zeros(())
        terms = [
            (self.fisher[name].to(parameter.device) * (parameter - self.references[name].to(parameter.device)).square()).mean()
            for name, parameter in model.named_parameters()
            if name in self.references
        ]
        return self.strength * torch.stack(terms).mean() if terms else next(model.parameters()).new_zeros(())

    def consolidate(self, model: nn.Module, inputs: Tensor, labels: Tensor, batch_size: int) -> None:
        model.eval()
        fisher = {
            name: torch.zeros_like(parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.is_floating_point()
        }
        batches = 0
        for start in range(0, inputs.shape[0], batch_size):
            output = model(inputs[start : start + batch_size], sample=False)
            loss = F.cross_entropy(output.logits, labels[start : start + batch_size])
            gradients = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
            for (name, parameter), gradient in zip(model.named_parameters(), gradients):
                if gradient is not None and name in fisher:
                    fisher[name].add_(gradient.detach().square())
            batches += 1
        if batches == 0:
            raise ValueError("cannot consolidate EWC with an empty task")
        fisher = {name: value / batches for name, value in fisher.items()}
        if self.fisher:
            fisher = {
                name: 0.5 * self.fisher[name] + 0.5 * value
                for name, value in fisher.items()
            }
        self.fisher = fisher
        self.references = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if name in fisher
        }

    @property
    def stored_elements(self) -> int:
        return sum(value.numel() for value in self.references.values()) + sum(
            value.numel() for value in self.fisher.values()
        )


@dataclass
class _SIState:
    importance: dict[str, Tensor]
    references: dict[str, Tensor]
    path: dict[str, Tensor]
    task_start: dict[str, Tensor]
    strength: float = 1.0
    damping: float = 0.1

    @classmethod
    def empty(cls, strength: float = 1.0) -> "_SIState":
        return cls(importance={}, references={}, path={}, task_start={}, strength=strength)

    def begin_task(self, model: nn.Module) -> None:
        self.task_start = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.is_floating_point()
        }
        self.path = {
            name: torch.zeros_like(parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.is_floating_point()
        }

    def penalty(self, model: nn.Module) -> Tensor:
        if not self.references:
            return next(model.parameters()).new_zeros(())
        terms = [
            (self.importance[name].to(parameter.device) * (parameter - self.references[name].to(parameter.device)).square()).mean()
            for name, parameter in model.named_parameters()
            if name in self.references
        ]
        return self.strength * torch.stack(terms).mean() if terms else next(model.parameters()).new_zeros(())

    def observe_step(self, model: nn.Module, before: dict[str, Tensor], gradients: dict[str, Tensor]) -> None:
        for name, parameter in model.named_parameters():
            if name in self.path and name in gradients:
                delta = parameter.detach() - before[name]
                self.path[name].add_((-gradients[name] * delta).detach())

    def consolidate(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name not in self.path:
                continue
            delta = parameter.detach() - self.task_start[name]
            omega = (self.path[name] / (delta.square() + self.damping)).clamp_min(0.0)
            previous = self.importance.get(name)
            self.importance[name] = omega if previous is None else previous + omega
        self.references = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if name in self.path
        }
        self.path = {}
        self.task_start = {}

    @property
    def stored_elements(self) -> int:
        return sum(value.numel() for value in self.references.values()) + sum(
            value.numel() for value in self.importance.values()
        )


class _PackNetState:
    """Magnitude pruning with a persistent trainable mask."""

    def __init__(self) -> None:
        self.trainable: dict[str, Tensor] = {}

    def initialize(self, model: nn.Module) -> None:
        if self.trainable:
            return
        self.trainable = {
            name: torch.ones_like(parameter, dtype=torch.bool)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.is_floating_point()
        }

    def apply_gradient_masks(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and name in self.trainable:
                parameter.grad.mul_(self.trainable[name].to(parameter.grad.device))

    def prune(self, model: nn.Module, fraction: float = 0.5) -> None:
        self.initialize(model)
        candidates = [
            parameter.detach().abs()[self.trainable[name]]
            for name, parameter in model.named_parameters()
            if name in self.trainable and self.trainable[name].any()
        ]
        if not candidates:
            return
        values = torch.cat(candidates)
        freeze_count = max(1, int(values.numel() * fraction))
        freeze_count = min(freeze_count, values.numel())
        threshold = torch.topk(values, freeze_count, largest=True, sorted=False).values.min()
        for name, parameter in model.named_parameters():
            if name in self.trainable:
                selected = self.trainable[name] & parameter.detach().abs().ge(threshold)
                self.trainable[name] = self.trainable[name] & ~selected

    @property
    def stored_elements(self) -> int:
        return sum(value.numel() for value in self.trainable.values())


def _fit_column(
    model: BONSAIModel,
    inputs: Tensor,
    labels: Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    penalty=None,
    gradient_mask=None,
    si_state: _SIState | None = None,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    inputs = inputs.float()
    labels = labels.long()
    for _ in range(epochs):
        permutation = torch.randperm(inputs.shape[0])
        model.train()
        for start in range(0, inputs.shape[0], batch_size):
            batch_indices = permutation[start : start + batch_size]
            batch_inputs = inputs[batch_indices]
            batch_labels = labels[batch_indices]
            before = (
                {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                if si_state is not None
                else None
            )
            output = model(batch_inputs, sample=True)
            loss = F.cross_entropy(output.logits, batch_labels)
            loss = loss + output.vib.kl * model.encoder.beta
            if penalty is not None:
                loss = loss + penalty()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = (
                {
                    name: parameter.grad.detach().clone()
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                }
                if si_state is not None
                else None
            )
            if gradient_mask is not None:
                gradient_mask(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            if si_state is not None and before is not None and gradients is not None:
                si_state.observe_step(model, before, gradients)


def _task_tensors(tasks) -> list[tuple[Tensor, Tensor]]:
    return [
        (
            torch.stack([task[index][0] for index in range(len(task))]).flatten(start_dim=1),
            task.global_labels.clone(),
        )
        for task in tasks
    ]


def _evaluate(
    method: str,
    model: BONSAIModel | None,
    columns: nn.ModuleList | None,
    test_tasks: list[tuple[Tensor, Tensor]],
) -> list[float]:
    accuracies: list[float] = []
    for task_id, (inputs, labels) in enumerate(test_tasks):
        active = columns[task_id] if columns is not None else model
        assert active is not None
        with torch.no_grad():
            output = active(inputs, sample=False)
            prediction = output.logits.argmax(dim=1)
        accuracies.append(float((prediction == labels).float().mean()))
    return accuracies


def run_current_backbone_comparison(
    seeds: Iterable[int] = (7, 17, 27, 37, 47),
    num_tasks: int = 4,
    classes_per_task: int = 3,
    train_samples_per_class: int = 24,
    test_samples_per_class: int = 12,
    image_size: int = 32,
    noise: float = 0.1,
    epochs: int = 5,
    batch_size: int = 32,
    methods: Iterable[str] = CURRENT_BASELINE_METHODS,
    learning_rate: float = 3e-3,
    ewc_strength: float = 1.0,
    si_strength: float = 1.0,
    packnet_prune_fraction: float = 0.5,
) -> list[dict]:
    """Compare EWC, SI, PackNet, and PNN on the current VIB/MLP core.

    Hyperparameters are fixed defaults shared across all seeds.  EWC and SI
    are online bounded-state versions; PackNet freezes the largest half of
    currently free weights after each task; PNN creates one frozen current-core
    column per task.  The global-classifier controls have no task oracle at
    evaluation, while PNN is explicitly marked task-aware-only.
    """

    method_names = tuple(methods)
    valid = set(CURRENT_BASELINE_METHODS)
    if not method_names or any(method not in valid for method in method_names):
        raise ValueError(f"methods must be a non-empty subset of {sorted(valid)}")
    if not 0.0 < packnet_prune_fraction <= 1.0:
        raise ValueError("packnet_prune_fraction must be in (0, 1]")
    input_dim = 3 * image_size * image_size
    total_classes = num_tasks * classes_per_task
    records: list[dict] = []
    for seed in seeds:
        train_tasks, test_tasks = make_synthetic_image_tasks(
            num_tasks=num_tasks,
            classes_per_task=classes_per_task,
            train_samples_per_class=train_samples_per_class,
            test_samples_per_class=test_samples_per_class,
            image_size=image_size,
            noise=noise,
            seed=int(seed),
        )
        train_tensors = _task_tensors(train_tasks)
        test_tensors = _task_tensors(test_tasks)
        for method in method_names:
            torch.manual_seed(int(seed))
            start_time = perf_counter()
            model: BONSAIModel | None = None
            columns: nn.ModuleList | None = None
            ewc = _EWCState.empty(ewc_strength) if method == "current_ewc" else None
            si = _SIState.empty(si_strength) if method == "current_si" else None
            packnet = _PackNetState() if method == "current_packnet" else None
            if method == "current_pnn":
                columns = nn.ModuleList()
            else:
                model = BONSAIModel(
                    input_dim=input_dim,
                    num_classes=total_classes,
                    hidden_dim=64,
                    latent_dim=16,
                    vib_beta=1e-3,
                    adapter_rank=2,
                )
            accuracy_history: list[list[float]] = []
            for task_id, (inputs, labels) in enumerate(train_tensors):
                if columns is not None:
                    column = BONSAIModel(
                        input_dim=input_dim,
                        num_classes=total_classes,
                        hidden_dim=64,
                        latent_dim=16,
                        vib_beta=1e-3,
                        adapter_rank=2,
                    )
                    columns.append(column)
                    _fit_column(column, inputs, labels, epochs, batch_size, learning_rate)
                    for parameter in column.parameters():
                        parameter.requires_grad_(False)
                else:
                    assert model is not None
                    if si is not None:
                        si.begin_task(model)
                    _fit_column(
                        model,
                        inputs,
                        labels,
                        epochs,
                        batch_size,
                        learning_rate,
                        penalty=(
                            (lambda: ewc.penalty(model))
                            if ewc is not None
                            else ((lambda: si.penalty(model)) if si is not None else None)
                        ),
                        gradient_mask=(packnet.apply_gradient_masks if packnet is not None else None),
                        si_state=si,
                    )
                    if ewc is not None:
                        ewc.consolidate(model, inputs, labels, batch_size)
                    if si is not None:
                        si.consolidate(model)
                    if packnet is not None:
                        packnet.prune(model, packnet_prune_fraction)
                accuracy_history.append(
                    _evaluate(
                        method,
                        model,
                        columns,
                        test_tensors[: task_id + 1],
                    )
                )
            final_accuracies = accuracy_history[-1]
            if model is not None:
                parameter_count = sum(parameter.numel() for parameter in model.parameters())
                base_parameter_count = parameter_count
                if ewc is not None:
                    state_elements = ewc.stored_elements
                elif si is not None:
                    state_elements = si.stored_elements
                elif packnet is not None:
                    state_elements = packnet.stored_elements
                else:
                    state_elements = 0
                task_free_accuracy = average_accuracy(final_accuracies)
                task_free_forgetting = forgetting_measure(accuracy_history)
                task_free_route_accuracy = 1.0
            else:
                assert columns is not None
                parameter_count = sum(parameter.numel() for parameter in columns.parameters())
                base_parameter_count = sum(parameter.numel() for parameter in columns[0].parameters())
                state_elements = 0
                task_free_accuracy = None
                task_free_forgetting = None
                task_free_route_accuracy = None
            records.append(
                {
                    "seed": int(seed),
                    "method": method,
                    "backbone": "current_bonsai_vib_mlp",
                    "dataset": "synthetic_images",
                    "num_tasks": num_tasks,
                    "classes_per_task": classes_per_task,
                    "train_samples_per_class": train_samples_per_class,
                    "test_samples_per_class": test_samples_per_class,
                    "image_size": image_size,
                    "noise": noise,
                    "epochs": epochs,
                    "task_aware_average_accuracy": average_accuracy(final_accuracies),
                    "forgetting": forgetting_measure(accuracy_history),
                    "task_free_average_accuracy": task_free_accuracy,
                    "task_free_forgetting": task_free_forgetting,
                    "task_free_route_accuracy": task_free_route_accuracy,
                    "parameter_count": parameter_count,
                    "parameter_overhead_percent": 100.0 * (parameter_count / max(base_parameter_count, 1) - 1.0),
                    "consolidation_memory_elements": state_elements,
                    "replay_memory_elements": 0,
                    "consolidation_memory_fraction": state_elements / max(parameter_count, 1),
                    "training_time_seconds": perf_counter() - start_time,
                    "task_free_supported": model is not None,
                    "ewc_strength": ewc_strength if method == "current_ewc" else None,
                    "si_strength": si_strength if method == "current_si" else None,
                    "packnet_prune_fraction": packnet_prune_fraction if method == "current_packnet" else None,
                }
            )
    return records
