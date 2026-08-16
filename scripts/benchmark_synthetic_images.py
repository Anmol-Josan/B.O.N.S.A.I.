"""Compare BONSAI and the ResNet baselines on structured image tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic_images import make_synthetic_image_tasks
from src.utils.real_benchmark import RealBenchmarkConfig, run_real_method, split_task_views
from src.utils.results import save_records_csv, summarize_records, write_summary_artifacts
from src.utils.visualization import plot_accuracy_curves, plot_mask_sparsity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/synthetic_image_benchmark"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--num-tasks", type=int, default=4)
    parser.add_argument("--classes-per-task", type=int, default=3)
    parser.add_argument("--train-samples-per-class", type=int, default=24)
    parser.add_argument("--test-samples-per-class", type=int, default=12)
    parser.add_argument("--noise", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--task-adapter-rank", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--route-strategy",
        choices=("entropy", "prototype", "hybrid", "learned"),
        default="learned",
    )
    args = parser.parse_args()

    methods = ("BONSAI", "EWC", "SI", "PackNet", "PNN")
    records: list[dict] = []
    histories: dict[str, list[list[list[float]]]] = {method: [] for method in methods}
    first_masks = {}
    for seed in args.seeds:
        train_tasks, test_tasks = make_synthetic_image_tasks(
            num_tasks=args.num_tasks,
            classes_per_task=args.classes_per_task,
            train_samples_per_class=args.train_samples_per_class,
            test_samples_per_class=args.test_samples_per_class,
            noise=args.noise,
            seed=seed,
        )
        train_tasks, validation_tasks = split_task_views(
            train_tasks, validation_fraction=args.validation_fraction, seed=seed
        )
        config = RealBenchmarkConfig(
            dataset="synthetic_images",
            data_root=Path("data"),
            output_dir=args.output_dir,
            seeds=(seed,),
            epochs_per_task=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            task_adapter_rank=args.task_adapter_rank,
            route_strategy=args.route_strategy,
            route_head_epochs=3,
        )
        for method in methods:
            record, history, masks = run_real_method(
                method,
                train_tasks,
                seed=seed,
                config=config,
                evaluation_tasks=test_tasks,
                validation_tasks=validation_tasks,
            )
            record["noise"] = args.noise
            records.append(record)
            histories[method].append(history)
            if method == "BONSAI" and not first_masks:
                first_masks = masks

    summaries = summarize_records(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_records_csv(args.output_dir / "runs.csv", records)
    write_summary_artifacts(args.output_dir / "summary.json", summaries)
    curves = {
        method: [
            sum(history[task_id][-1] for history in runs if len(history) > task_id) / len(runs)
            for task_id in range(max(len(history) for history in runs))
        ]
        for method, runs in histories.items()
    }
    plot_accuracy_curves(curves, args.output_dir / "accuracy_curves.png", title="Synthetic image accuracy")
    if first_masks:
        plot_mask_sparsity(first_masks, args.output_dir / "mask_sparsity.png")
    print(f"wrote {len(records)} runs and {len(summaries)} summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
