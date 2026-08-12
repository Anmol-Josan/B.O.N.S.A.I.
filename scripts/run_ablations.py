"""Run the mandatory IB, rewiring, and capacity ablations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.benchmarking import ToyBenchmarkConfig, run_toy_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/ablations"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    args = parser.parse_args()
    run_toy_suite(ToyBenchmarkConfig(seeds=tuple(args.seeds), output_dir=args.output_dir))
    print(f"ablation artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
