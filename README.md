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

On Windows with an Intel Iris Xe or another DirectX 12 GPU, install the optional
DirectML backend and select it explicitly:

```powershell
python -m pip install -e ".[directml]"
python scripts/benchmark.py --dataset cifar100 --data-root data/cifar100 --device dml --methods BONSAI PNN
```

DirectML may fall back to CPU for unsupported operators (the current Adam update
does this), so benchmark wall-clock time rather than assuming every workload is
GPU-faster. The runner accepts `cpu`, `cuda`, and `dml`/`directml` device names.

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
validation task split, uses a train-only validation partition for growth
triggers, and allocates a low-rank task adapter for BONSAI. Its class-incremental
evaluation routes images without passing a task ID. A real
dataset run is still required before making a claim against EWC, SI, PackNet, or
PNN; the synthetic artifacts are mechanism tests, not a substitute for that
benchmark.

For a larger image-side mechanism test without downloading a dataset:

```powershell
python scripts/benchmark_synthetic_images.py --output-dir results/synthetic_image_benchmark_4tasks --seeds 7 --num-tasks 4 --classes-per-task 3 --epochs 1 --noise 0.1 --task-adapter-rank 1 --route-strategy learned
```

BONSAI trains the shared ResNet representation together with the current task's
low-rank stage adapters and local head, adds the VIB KL term, and retains a
global class-scaffold objective with bounded rehearsal. It then allocates a
non-overlapping saliency mask and rewires only available shared weights. A
small one-vs-rest compatibility discriminator is fit for each task path from
bounded positive/negative exemplars; this is the recommended task-ID-free
router because raw entropy is not calibrated across paths.

For the stronger routing configuration used in the larger stress comparison:

```powershell
python scripts/benchmark_synthetic_images.py --output-dir results/bonsai_3seed_compatibility_rank4_4tasks --seeds 7 17 27 --num-tasks 4 --classes-per-task 3 --epochs 5 --noise 0.1 --task-adapter-rank 4 --route-strategy compatibility --route-compatibility-epochs 10 --methods BONSAI
```

The current-source comparison artifacts are stored in
`results/bonsai_3seed_current_4tasks`, `results/pnn_3seed_current_bnfixed_4tasks`,
`results/bonsai_8tasks_unique_mlp64`, and `results/pnn_8tasks_unique_bnfixed`.

The first larger real-data comparison is in
`results/cifar100_10task_dml_bonsai_pnn_32x3_3seed`. It uses 10 sequential
10-class CIFAR-100 tasks, three seeds, three epochs per task, and 32 examples
per class on the DirectML device. BONSAI reaches `21.43% +/- 0.53%`
task-aware average accuracy with `4.23%` parameter overhead, versus PNN at
`11.97% +/- 0.58%` with `900%` overhead. BONSAI's task-aware forgetting is
slightly negative (`-1.27` percentage points), but its task-free route
accuracy is only `11.34%`, close to the `10%` ten-task chance level. The
task-free selector is therefore the main open research problem; the current
results support a strong task-aware efficiency claim, not yet a solved
class-incremental routing claim.

The route ablations and negative controls are retained beside the main result,
including balanced prototype/compatibility routing, fused routing, route
replay, global-head calibration, BN freezing, and rank-8 adapter capacity.

The latest research iteration also refreshes all route prototypes and
compatibility calibrators after shared rewiring, preventing stale route
statistics. Optional `feature_replay_weight` and `local_replay_weight`
controls preserve old feature geometry or replay old task-local heads. In a
matched 10-task CIFAR-100 pilot, feature replay improved route accuracy by
about 0.75 percentage points at a small task-aware cost. Local replay was
validated on the synthetic suite only in this iteration; both controls remain
off by default until they are validated across multiple seeds.

Results are written to the configured `results/` directory as CSV/JSON summaries and PNG plots. W&B logging is opt-in with `--wandb` and `WANDB_API_KEY`.
