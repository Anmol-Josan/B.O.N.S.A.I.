"""Measure the modular Topological--Riemannian BONSAI architecture.

The scaling suite operates on precomputed latent task clouds so repository
costs are isolated from encoder-training cost. The optional training ablation
is a separate small continual-learning experiment and is not mixed into the
scaling claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bonsai.evaluation import (
    run_continual_method_comparison,
    run_continual_robustness_matrix,
    run_router_ablation,
    run_scaling_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/architecture_metrics.json"))
    parser.add_argument("--task-counts", type=int, nargs="+", default=[4, 8, 16, 32, 50])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--comparison-seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--skip-training-ablation", action="store_true")
    parser.add_argument("--run-robustness-matrix", action="store_true")
    args = parser.parse_args()
    payload = {
        "scaling": run_scaling_benchmark(task_counts=args.task_counts, seed=args.seed),
        "router_ablation": run_router_ablation(task_count=min(8, max(args.task_counts)), seed=args.seed),
    }
    if not args.skip_training_ablation:
        payload["training_ablation"] = run_continual_method_comparison(
            seeds=args.comparison_seeds
        )
        if args.run_robustness_matrix:
            payload["continual_robustness_matrix"] = run_continual_robustness_matrix(
                seeds=args.comparison_seeds
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
