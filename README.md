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

For a larger shared-representation stress test:

```powershell
python scripts/benchmark.py --output-dir results/scaling_10tasks_5classes --seeds 7 17 27 --epochs 5 --samples-per-class 32 --num-tasks 10 --classes-per-task 5 --input-dim 20 --hidden-dim 64 --shared-encoder-updates
```

The larger synthetic run uses lazy task-path allocation, rank-1 residual adapters,
non-overlapping saliency masks with a 65% cumulative budget, a small rehearsal
buffer, residual orthogonal rewiring, and prototype-assisted task routing by
default. Use `--route-strategy entropy` to reproduce the original entropy-only
selector, or `--replay-per-task 0` for a no-replay control.

The image runner now evaluates on the held-out CIFAR-100 test or TinyImageNet
validation task split and allocates a low-rank task adapter for BONSAI. A real
dataset run is still required before making a claim against EWC, SI, PackNet, or
PNN; the synthetic artifacts are mechanism tests, not a substitute for that
benchmark.

Results are written to the configured `results/` directory as CSV/JSON summaries and PNG plots. W&B logging is opt-in with `--wandb` and `WANDB_API_KEY`.
