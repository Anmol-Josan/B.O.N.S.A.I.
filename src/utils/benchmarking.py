"""Reproducible compact benchmark and ablation execution utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from src.algorithms.continual import ToyContinualLearner
from src.algorithms.mask_manager import MaskManager
from src.algorithms.rewire import RewireEngine
from src.data.synthetic import make_synthetic_tasks
from src.utils.metrics import average_accuracy, forgetting_measure, parameter_overhead
from src.utils.reproducibility import seed_everything
from src.utils.results import save_records_csv, summarize_records, write_summary_artifacts
from src.utils.visualization import plot_accuracy_curves, plot_mask_sparsity


@dataclass(frozen=True)
class ToyBenchmarkConfig:
    """Fast configuration for CI and development-scale BONSAI experiments."""

    seeds: tuple[int, ...] = (7, 17, 27, 37, 47)
    epochs_per_task: int = 5
    samples_per_class: int = 32
    num_tasks: int = 2
    classes_per_task: int = 2
    input_dim: int = 2
    hidden_dim: int = 16
    noise: float = 0.35
    shared_encoder_updates: bool = False
    adapter_rank: int = 1
    encoder_learning_rate_scale: float = 0.1
    rewire_strength: float = 0.15
    max_frozen_fraction: float | None = 0.65
    route_strategy: str = "prototype"
    replay_per_task: int = 16
    replay_weight: float = 0.5
    output_dir: Path = Path("results/toy_benchmark")
    use_wandb: bool = False


def run_toy_bonsai(seed: int, variant: str, config: ToyBenchmarkConfig) -> dict:
    """Run one two-task BONSAI/ablation experiment and return scalar metrics."""

    if variant not in {
        "full",
        "no_ib",
        "no_orthogonal_rewire",
        "fixed_capacity",
        "no_replay",
    }:
        raise ValueError(f"unknown ablation variant: {variant}")
    seed_everything(seed)
    tasks = make_synthetic_tasks(
        num_tasks=config.num_tasks,
        classes_per_task=config.classes_per_task,
        samples_per_class=config.samples_per_class,
        input_dim=config.input_dim,
        noise=config.noise,
        seed=seed,
    )
    adapter_rank = 0 if variant == "fixed_capacity" else config.adapter_rank
    learner = ToyContinualLearner(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        classes_per_task=config.classes_per_task,
        num_tasks=config.num_tasks,
        beta=0.0 if variant == "no_ib" else 0.001,
        adapter_rank=adapter_rank,
        encoder_learning_rate_scale=config.encoder_learning_rate_scale,
        lazy_task_paths=True,
        replay_per_task=config.replay_per_task,
        replay_weight=0.0 if variant == "no_replay" else config.replay_weight,
    )
    initial_parameters = learner.total_parameters
    manager = MaskManager(
        saliency_quantile=0.8,
        max_frozen_fraction=config.max_frozen_fraction,
    )
    strategy = "gaussian" if variant == "no_orthogonal_rewire" else "orthogonal"
    history: list[list[float]] = []
    critical_fraction_curve: list[float] = []
    for task_id, task in enumerate(tasks):
        learner.allocate_task_path(task_id)
        if task_id > 0:
            if config.shared_encoder_updates:
                encoder_masks = {
                    name[len("encoder.") :]: mask
                    for name, mask in manager.critical_masks.items()
                    if name.startswith("encoder.")
                }
                RewireEngine(
                    strategy=strategy,
                    seed=seed + task_id,
                    strength=config.rewire_strength,
                ).rewire(
                    learner.encoder,
                    encoder_masks,
                    exclude_names={"logvar_layer.weight", "logvar_layer.bias"},
                )
            else:
                RewireEngine(strategy=strategy, seed=seed + task_id, strength=config.rewire_strength).rewire(
                    learner.heads[task_id],
                    {
                        "weight": torch.zeros_like(learner.heads[task_id].weight, dtype=torch.bool),
                        "bias": torch.zeros_like(learner.heads[task_id].bias, dtype=torch.bool),
                    },
                )
        learner.train_task(
            task_id,
            task,
            epochs=config.epochs_per_task,
            batch_size=32,
            update_encoder=(task_id == 0 or config.shared_encoder_updates),
            encoder_learning_rate_scale=config.encoder_learning_rate_scale,
        )
        learner.store_replay(task_id, task)
        learner.register_task_route(task_id, task)
        if variant == "no_ib":
            # Ablation A: remove the IB objective and use magnitude pruning.
            saliency = {
                name: parameter.detach().abs().clone()
                for name, parameter in learner.named_parameters()
                if parameter.requires_grad
            }
        else:
            saliency = manager.compute_saliency(learner, learner.loss_on_task(task_id, task))
        masks = manager.build_critical_masks(
            saliency,
            excluded_masks=manager.critical_masks,
            total_parameter_count=learner.total_parameters,
        )
        manager.freeze_critical(learner, masks)
        critical_fraction_curve.append(manager.frozen_parameter_count / learner.total_parameters)
        history.append([learner.accuracy(seen_id, tasks[seen_id]) for seen_id in range(task_id + 1)])
    task_one_before = history[0][0]
    task_one_after = history[-1][0]
    task_two = history[-1][-1]
    inputs = torch.cat([task.inputs for task in tasks])
    expected_routes = torch.cat(
        [torch.full((len(task),), task_id, dtype=torch.long) for task_id, task in enumerate(tasks)]
    )
    _, selected_routes, _ = learner.predict_with_entropy(
        inputs, route_strategy=config.route_strategy
    )
    _, entropy_routes, _ = learner.predict_with_entropy(inputs, route_strategy="entropy")
    _, prototype_routes, _ = learner.predict_with_entropy(inputs, route_strategy="prototype")
    return {
        "dataset": "synthetic",
        "noise": config.noise,
        "method": "BONSAI",
        "variant": variant,
        "seed": seed,
        "average_accuracy": average_accuracy(history[-1]),
        "forgetting": forgetting_measure(history),
        "parameter_overhead_percent": parameter_overhead(initial_parameters, learner.total_parameters),
        "task_one_accuracy_before": task_one_before,
        "task_one_accuracy_after": task_one_after,
        "task_two_accuracy": task_two,
        "route_accuracy": (selected_routes == expected_routes).float().mean().item(),
        "route_accuracy_entropy": (entropy_routes == expected_routes).float().mean().item(),
        "route_accuracy_prototype": (prototype_routes == expected_routes).float().mean().item(),
        "accuracy_curve": [average_accuracy(row) for row in history],
        "critical_fraction_curve": critical_fraction_curve,
        "critical_fraction": manager.frozen_parameter_count / learner.total_parameters,
    }


def run_toy_suite(config: ToyBenchmarkConfig) -> tuple[list[dict], list[dict]]:
    """Run five-seed BONSAI and all mandatory ablations, writing artifacts."""

    variants = ("full", "no_ib", "no_orthogonal_rewire", "fixed_capacity", "no_replay")
    records = [
        run_toy_bonsai(seed, variant, config)
        for variant in variants
        for seed in config.seeds
    ]
    summaries = summarize_records(records, group_keys=("dataset", "method", "variant"))
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_records_csv(output_dir / "runs.csv", records)
    write_summary_artifacts(output_dir / "summary.json", summaries)
    curves = {}
    for variant in variants:
        variant_records = [record for record in records if record["variant"] == variant]
        max_tasks = max(len(record["accuracy_curve"]) for record in variant_records)
        curves[variant] = [
            sum(
                record["accuracy_curve"][task_id]
                for record in variant_records
                if len(record["accuracy_curve"]) > task_id
            )
            / len(variant_records)
            for task_id in range(max_tasks)
        ]
    plot_accuracy_curves(curves, output_dir / "accuracy_curves.png", title="BONSAI toy ablations")
    max_tasks = max(len(record["critical_fraction_curve"]) for record in records)
    mask_profiles = {
        task_id + 1: {
            "critical_fraction": torch.tensor(
                [
                    sum(
                        record["critical_fraction_curve"][task_id]
                        for record in records
                        if len(record["critical_fraction_curve"]) > task_id
                    )
                    / len(records)
                ]
            )
        }
        for task_id in range(max_tasks)
    }
    plot_mask_sparsity(mask_profiles, output_dir / "mask_sparsity.png")
    _log_to_wandb_if_available(records, summaries, config.use_wandb)
    return records, summaries


def _log_to_wandb_if_available(records: Sequence[dict], summaries: Sequence[dict], enabled: bool) -> None:
    if not enabled or not os.getenv("WANDB_API_KEY"):
        return
    try:  # pragma: no cover - network/account dependent
        import wandb

        with wandb.init(project="bonsai-continual-learning", reinit="finish"):
            for record in records:
                wandb.log(record)
            wandb.log({"summary_rows": len(summaries)})
    except Exception as error:  # pragma: no cover - network/account dependent
        print(f"wandb logging skipped: {error}")
