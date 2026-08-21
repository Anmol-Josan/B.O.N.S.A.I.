"""Run the factorial ATGFR component ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bonsai.evaluation import run_atgfr_component_ablation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/atgfr_component_ablation.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--num-tasks", type=int, default=4)
    parser.add_argument("--classes-per-task", type=int, default=3)
    parser.add_argument("--train-samples-per-class", type=int, default=16)
    parser.add_argument("--test-samples-per-class", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    records = run_atgfr_component_ablation(
        seeds=args.seeds,
        num_tasks=args.num_tasks,
        classes_per_task=args.classes_per_task,
        train_samples_per_class=args.train_samples_per_class,
        test_samples_per_class=args.test_samples_per_class,
        image_size=args.image_size,
        noise=args.noise,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2))
    print(f"wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
