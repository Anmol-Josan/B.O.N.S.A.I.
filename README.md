# BONSAI

BONSAI (Bottleneck-guided Orthogonal Network Subgraph Allocation for Incremental learning) is a research implementation for dynamic neural subgraph rewiring with variational information-bottleneck regularization.

The repository is intentionally developed test-first. Core components are small, deterministic, and usable independently before launching benchmark-scale experiments.

## Quick start

```powershell
python -m pip install -e ".[dev]"
pytest
```

The benchmark script defaults to compact synthetic data and runs BONSAI plus the three mandatory ablations across five seeds:

```powershell
python scripts/benchmark.py
```

Dataset-backed runs use the leakage-safe task factories and common ResNet-18 baseline runner:

```powershell
python scripts/benchmark.py --dataset cifar100 --data-root data/cifar100 --download
python scripts/benchmark.py --dataset tinyimagenet --data-root data/tiny-imagenet-200
```

Results are written to the configured `results/` directory as CSV/JSON summaries and PNG plots. W&B logging is opt-in with `--wandb` and `WANDB_API_KEY`.
