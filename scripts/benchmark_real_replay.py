"""Run the matched real-image replay review suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmarks.real_replay import (
    VALID_METHODS,
    RealReplayConfig,
    config_as_dict,
    run_real_replay_suite,
    summarize_real_replay,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("results/real_replay_cifar100_review.json")
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--methods", nargs="+", choices=VALID_METHODS, default=list(VALID_METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--order-count", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--memory-per-task", type=int, default=20)
    parser.add_argument("--train-samples-per-class", type=int, default=16)
    parser.add_argument("--test-samples-per-class", type=int, default=40)
    args = parser.parse_args()
    config = RealReplayConfig(
        data_root=args.data_root,
        methods=tuple(args.methods),
        seeds=tuple(args.seeds),
        order_count=args.order_count,
        epochs_per_task=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        memory_per_task=args.memory_per_task,
        train_samples_per_class=args.train_samples_per_class,
        test_samples_per_class=args.test_samples_per_class,
    )
    records = run_real_replay_suite(config)
    payload = {
        "config": config_as_dict(config),
        "records": records,
        "summaries": summarize_real_replay(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summaries"], indent=2))
    print(f"wrote {len(records)} runs to {args.output}")


if __name__ == "__main__":
    main()
