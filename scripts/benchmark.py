"""Benchmark entry point placeholder for the validated experiment runner."""

from __future__ import annotations

from src.utils.reproducibility import seed_everything


def main() -> None:
    seed_everything(7)
    print("BONSAI benchmark entry point initialized; use the experiment runner configuration.")


if __name__ == "__main__":
    main()

