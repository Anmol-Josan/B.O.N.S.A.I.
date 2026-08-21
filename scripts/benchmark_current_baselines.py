"""Benchmark EWC, SI, PackNet, and PNN on the current BONSAI core."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bonsai.matched_baselines import run_current_backbone_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/current_backbone_baselines.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--num-tasks", type=int, default=4)
    parser.add_argument("--classes-per-task", type=int, default=3)
    parser.add_argument("--train-samples-per-class", type=int, default=24)
    parser.add_argument("--test-samples-per-class", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ewc-strength", type=float, default=1.0)
    parser.add_argument("--si-strength", type=float, default=1.0)
    parser.add_argument("--packnet-prune-fraction", type=float, default=0.5)
    args = parser.parse_args()
    records = run_current_backbone_comparison(
        seeds=args.seeds,
        num_tasks=args.num_tasks,
        classes_per_task=args.classes_per_task,
        train_samples_per_class=args.train_samples_per_class,
        test_samples_per_class=args.test_samples_per_class,
        image_size=args.image_size,
        noise=args.noise,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ewc_strength=args.ewc_strength,
        si_strength=args.si_strength,
        packnet_prune_fraction=args.packnet_prune_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2))
    print(f"wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
