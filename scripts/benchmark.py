"""Run the reproducible BONSAI benchmark/ablation suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.benchmarking import ToyBenchmarkConfig, run_toy_suite
from src.utils.real_benchmark import RealBenchmarkConfig, run_real_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("synthetic", "cifar100", "tinyimagenet"), default="synthetic")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/toy_benchmark"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--num-tasks", type=int, default=2)
    parser.add_argument("--classes-per-task", type=int, default=2)
    parser.add_argument("--input-dim", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--shared-encoder-updates", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.dataset == "synthetic":
        records, summaries = run_toy_suite(
            ToyBenchmarkConfig(
                seeds=tuple(args.seeds),
                epochs_per_task=args.epochs,
                samples_per_class=args.samples_per_class,
                num_tasks=args.num_tasks,
                classes_per_task=args.classes_per_task,
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                shared_encoder_updates=args.shared_encoder_updates,
                output_dir=args.output_dir,
                use_wandb=args.wandb,
            )
        )
    else:
        records, summaries = run_real_suite(
            RealBenchmarkConfig(
                dataset=args.dataset,
                data_root=args.data_root,
                seeds=tuple(args.seeds),
                epochs_per_task=args.epochs,
                output_dir=args.output_dir,
                download=args.download,
            )
        )
    print(f"wrote {len(records)} runs and {len(summaries)} summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
