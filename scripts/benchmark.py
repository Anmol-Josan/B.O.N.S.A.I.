"""Run the reproducible BONSAI benchmark/ablation suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.benchmarking import ToyBenchmarkConfig, run_toy_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/toy_benchmark"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    records, summaries = run_toy_suite(
        ToyBenchmarkConfig(
            seeds=tuple(args.seeds),
            epochs_per_task=args.epochs,
            samples_per_class=args.samples_per_class,
            output_dir=args.output_dir,
            use_wandb=args.wandb,
        )
    )
    print(f"wrote {len(records)} runs and {len(summaries)} summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
