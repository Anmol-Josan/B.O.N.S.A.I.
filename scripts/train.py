"""Entry point for a configurable BONSAI experiment.

The actual training components are imported lazily so repository setup and unit
tests do not require a dataset download or a wandb login.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.reproducibility import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a BONSAI experiment")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("BONSAI training entry point initialized; select a benchmark config to train.")


if __name__ == "__main__":
    main()
