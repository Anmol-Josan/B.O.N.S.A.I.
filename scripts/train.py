"""Lightweight end-to-end training entry point for modular BONSAI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.reproducibility import seed_everything
from src.bonsai.atgtr import AdaptiveTaskGraphTrustRegion
from src.bonsai.continual import BONSAITrainer, InterferenceProtector, LossWeights
from src.bonsai.replay import AdaptiveTaskGraphFunctionalReplay
from src.bonsai.rgsc import TopologyGatedRiemannianSubspaceConsolidator
from src.bonsai.system import BONSAISystem
from src.data.synthetic import make_synthetic_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a BONSAI experiment")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--num-tasks", type=int, default=4)
    parser.add_argument("--classes-per-task", type=int, default=2)
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs-per-task", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument(
        "--continual-method",
        choices=("ewc", "tgrsc", "atgtr", "atgfr"),
        default="tgrsc",
    )
    args = parser.parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = make_synthetic_tasks(
        num_tasks=args.num_tasks,
        classes_per_task=args.classes_per_task,
        samples_per_class=args.samples_per_class,
        input_dim=args.input_dim,
        seed=args.seed,
    )
    system = BONSAISystem(
        input_dim=args.input_dim,
        num_classes=args.num_tasks * args.classes_per_task,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        repository_kwargs={"coarse_dim": min(6, args.latent_dim)},
    )
    use_tgrsc = args.continual_method == "tgrsc"
    use_atgtr = args.continual_method == "atgtr"
    use_atgfr = args.continual_method == "atgfr"
    trainer = BONSAITrainer(
        system,
        learning_rate=args.learning_rate,
        weights=LossWeights(functional_replay=1.0 if use_atgfr else 0.0),
        protection=InterferenceProtector(
            0.0 if use_tgrsc or use_atgtr or use_atgfr else 1.0
        ),
        subspace_protection=(
            TopologyGatedRiemannianSubspaceConsolidator(
                rank=4,
                metric=system.metric,
                shared_prefixes=("encoder.mu_layer.", "encoder.logvar_layer."),
            )
            if use_tgrsc
            else (
                AdaptiveTaskGraphTrustRegion(
                    rank=4,
                    trust_fraction=0.5,
                    damping=1e-3,
                    max_constraints=24,
                    metric=system.metric,
                    shared_prefixes=("encoder.mu_layer.", "encoder.logvar_layer."),
                )
                if use_atgtr
                else None
            )
        ),
        functional_replay=(
            AdaptiveTaskGraphFunctionalReplay(
                replay_per_task=8,
                replay_strength=1.0,
                distill_strength=1.0,
                feature_strength=0.25,
                temperature=2.0,
                relation_floor=0.35,
                drift_budget=0.05,
                thermostat_gain=2.0,
                max_multiplier=8.0,
            )
            if use_atgfr
            else None
        ),
    )
    history: list[dict] = []
    for task in tasks:
        losses = trainer.fit_task(
            task.task_id,
            task.inputs,
            task.global_labels,
            epochs=args.epochs_per_task,
            batch_size=args.batch_size,
        )
        with torch.no_grad():
            task_aware = system.model(task.inputs, task_id=task.task_id, sample=False)
            task_aware_accuracy = float(
                (task_aware.logits.argmax(dim=1) == task.global_labels).float().mean()
            )
            route = system.route(task.inputs)
            task_free_predictions = []
            for selected_task in sorted(set(route.selected_task_ids.tolist())):
                positions = route.selected_task_ids == selected_task
                output = system.model(task.inputs[positions], task_id=selected_task, sample=False)
                task_free_predictions.append(
                    (positions.nonzero(as_tuple=False).flatten(), output.logits.argmax(dim=1))
                )
            routed_labels = task.global_labels.new_full(task.global_labels.shape, -1)
            for positions, predictions in task_free_predictions:
                routed_labels[positions] = predictions
            task_free_accuracy = float((routed_labels == task.global_labels).float().mean())
        history.append(
            {
                "task_id": task.task_id,
                "loss": losses[-1]["loss"],
                "task_aware_accuracy": task_aware_accuracy,
                "task_free_accuracy": task_free_accuracy,
                "candidate_count": route.candidate_comparisons / max(len(task), 1),
            }
        )
    payload = {
        "seed": args.seed,
        "continual_method": args.continual_method,
        "tasks": history,
        "parameter_overhead": system.parameter_overhead,
        "consolidation_memory_elements": trainer.protection.stored_elements
        + (
            trainer.subspace_protection.stored_elements
            if trainer.subspace_protection is not None
            else 0
        )
        + (
            trainer.functional_replay.stored_elements
            if trainer.functional_replay is not None
            else 0
        ),
        "replay_memory_elements": (
            trainer.functional_replay.stored_elements
            if trainer.functional_replay is not None
            else 0
        ),
        "consolidation_memory_fraction": (
            trainer.protection.stored_elements
            + (
                trainer.subspace_protection.stored_elements
                if trainer.subspace_protection is not None
                else 0
            )
            + (
                trainer.functional_replay.stored_elements
                if trainer.functional_replay is not None
                else 0
            )
        )
        / max(system.total_parameters, 1),
        "hierarchy_depth": system.repository.hierarchy.depth,
        "task_count": system.repository.task_count,
    }
    destination = args.output_dir / "training_metrics.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
