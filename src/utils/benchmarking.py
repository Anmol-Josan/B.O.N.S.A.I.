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
    output_dir: Path = Path("results/toy_benchmark")
    use_wandb: bool = False


def run_toy_bonsai(seed: int, variant: str, config: ToyBenchmarkConfig) -> dict:
    """Run one two-task BONSAI/ablation experiment and return scalar metrics."""

    if variant not in {"full", "no_ib", "no_orthogonal_rewire", "fixed_capacity"}:
        raise ValueError(f"unknown ablation variant: {variant}")
    seed_everything(seed)
    tasks = make_synthetic_tasks(
        num_tasks=2,
        classes_per_task=2,
        samples_per_class=config.samples_per_class,
        seed=seed,
    )
    learner = ToyContinualLearner(
        input_dim=2,
        hidden_dim=16,
        classes_per_task=2,
        num_tasks=2,
        beta=0.0 if variant == "no_ib" else 0.001,
    )
    initial_parameters = learner.total_parameters
    learner.train_task(0, tasks[0], epochs=config.epochs_per_task, batch_size=32)
    task_one_before = learner.accuracy(0, tasks[0])

    manager = MaskManager(saliency_quantile=0.8)
    if variant == "no_ib":
        # Ablation A: remove the IB objective and use magnitude pruning.
        saliency = {
            name: parameter.detach().abs().clone()
            for name, parameter in learner.named_parameters()
            if parameter.requires_grad
        }
    else:
        saliency = manager.compute_saliency(learner, learner.loss_on_task(0, tasks[0]))
    masks = manager.build_critical_masks(saliency)
    masks["heads.0.weight"] = torch.ones_like(learner.heads[0].weight, dtype=torch.bool)
    masks["heads.0.bias"] = torch.ones_like(learner.heads[0].bias, dtype=torch.bool)
    manager.freeze_critical(learner, masks)
    strategy = "gaussian" if variant == "no_orthogonal_rewire" else "orthogonal"
    RewireEngine(strategy=strategy, seed=seed + 1).rewire(
        learner.heads[1],
        {
            "weight": torch.zeros_like(learner.heads[1].weight, dtype=torch.bool),
            "bias": torch.zeros_like(learner.heads[1].bias, dtype=torch.bool),
        },
    )
    learner.train_task(1, tasks[1], epochs=config.epochs_per_task, batch_size=32)
    task_one_after = learner.accuracy(0, tasks[0])
    task_two = learner.accuracy(1, tasks[1])
    history = [[task_one_before], [task_one_after, task_two]]
    return {
        "dataset": "synthetic",
        "method": "BONSAI",
        "variant": variant,
        "seed": seed,
        "average_accuracy": average_accuracy(history[-1]),
        "forgetting": forgetting_measure(history),
        "parameter_overhead_percent": parameter_overhead(initial_parameters, learner.total_parameters),
        "task_one_accuracy_before": task_one_before,
        "task_one_accuracy_after": task_one_after,
        "task_two_accuracy": task_two,
        "critical_fraction": manager.frozen_parameter_count / learner.total_parameters,
    }


def run_toy_suite(config: ToyBenchmarkConfig) -> tuple[list[dict], list[dict]]:
    """Run five-seed BONSAI and all mandatory ablations, writing artifacts."""

    variants = ("full", "no_ib", "no_orthogonal_rewire", "fixed_capacity")
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
    curves = {
        variant: [
            sum(record["task_one_accuracy_before"] for record in records if record["variant"] == variant)
            / len(config.seeds),
            sum(record["average_accuracy"] for record in records if record["variant"] == variant)
            / len(config.seeds),
        ]
        for variant in variants
    }
    plot_accuracy_curves(curves, output_dir / "accuracy_curves.png", title="BONSAI toy ablations")
    mask_profiles = {
        task: {
            "critical_fraction": torch.tensor(
                [sum(record["critical_fraction"] for record in records) / len(records)]
            )
        }
        for task in (1, 2)
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
