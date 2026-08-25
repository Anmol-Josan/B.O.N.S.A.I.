# BONSAI

BONSAI (Bottleneck-guided Orthogonal Network Subgraph Allocation for Incremental learning) is a research implementation for dynamic neural subgraph rewiring with variational information-bottleneck regularization.

## Research scope and golden path

The repository includes both the evidence-backed paper path and a broader
experimental sandbox. The minimum supported claim is scoped to ATGFR's
functional replay mechanism under bounded-memory task-incremental learning;
the project does not claim state-of-the-art continual learning or solved
task-free routing. See [docs/RESEARCH_SCOPE.md](docs/RESEARCH_SCOPE.md) for
the claim boundary, metric definitions, evidence status, and reporting rules.

The canonical five-seed comparison is reproducible with:

```powershell
./scripts/reproduce_paper.ps1
```

Add `-IncludeCifarPilot` only when the CIFAR-100 data is available. Synthetic
latent-cloud, TGRSC, ATGTR, OT/TDA/sheaf, and routing experiments are retained
as diagnostics or exploratory variants and must not be presented as additional
validated contributions without a matched ablation.

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

The image runner evaluates on the held-out CIFAR-100 test or TinyImageNet
validation task split, uses a train-only validation partition for growth
triggers, and allocates a low-rank task adapter for BONSAI. Its class-incremental
evaluation routes images without passing a task ID. The matched replay review
suite now supplies the real-image comparison against ER, DER++, and ER-ACE;
the current-core benchmark now supplies the same-backbone EWC, SI, PackNet,
and PNN comparison.

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

## IEEE paper

The publication-formatted source is `paper/BONSAI_IEEE.tex`, with the local
IEEEtran class, BibTeX style, bibliography, and seven generated figures. The
compiled PDF is `output/pdf/BONSAI_IEEE.pdf`. From the repository root, a
MiKTeX/TeX Live build is:

```powershell
Set-Location paper
pdflatex -interaction=nonstopmode BONSAI_IEEE.tex
bibtex BONSAI_IEEE
pdflatex -interaction=nonstopmode BONSAI_IEEE.tex
pdflatex -interaction=nonstopmode BONSAI_IEEE.tex
```

The manuscript compares the current modular results with EWC, SI, PackNet,
and PNN on the same current VIB/MLP core. It includes the five-seed primary
benchmark, the complete four-cell task/feature grid, repository scaling,
ablations, architecture trade-offs, and reproducible citations.

The review-expansion artifacts add the experiments that were previously
missing from that manuscript:

```powershell
python scripts/benchmark_real_replay.py --output results/real_replay_cifar100_review.json --data-root data/cifar100 --seeds 7 17 --order-count 3 --epochs 2 --train-samples-per-class 8 --test-samples-per-class 20 --memory-per-task 20 --batch-size 64
python scripts/benchmark_atgfr_ablation.py --output results/atgfr_component_ablation.json --seeds 7 17 27 --epochs 3
```

The first command evaluates ER, DER++, ER-ACE, and ATGFR on the same
compact-CNN backbone, exemplar count, Split-CIFAR-100 stream, optimizer, and
task orders. It records wall-clock time and scalar memory for labels-only,
logit, and feature targets. The second command independently removes replay
terms, OT, H0 persistence, graph geometry, and the drift thermostat. Results
are intentionally retained even when an ablation is weaker; the current
mass-preserving relation normalization keeps graph weighting from silently
changing the total replay budget.

## Topological--Riemannian BONSAI architecture

The current architecture is implemented in `src/bonsai/`:

```text
VIB encoder -> task repository/hierarchy -> OT/TDA scoring
            -> local SPD Riemannian routing -> sparse sheaf compatibility
            -> shared low-rank adapter -> classifier
```

The modules are intentionally separate: `vib.py`, `repository.py`, `hierarchy.py`,
`ot.py`, `tda.py`, `geometry.py`, `sheaf.py`, `adapters.py`, `router.py`,
`continual.py`, `metrics.py`, and `evaluation.py`. `BONSAISystem` composes them.
Repository OT and H0 persistence descriptors are cached at task insertion. A
single-example route uses prototypes and local geometry; OT/TDA are only added
when a real episode/context batch is supplied. The TDA implementation is exact
zero-dimensional Vietoris--Rips persistence via an MST, rather than an
unsupported claim about higher-dimensional holes. The tangent `Log` map is the
explicit local chart approximation `z - p`, and the metric is guaranteed SPD by
bounded positive diagonal terms plus PSD low-rank updates.

Run the architecture-specific checks and measured scaling/ablation suite with:

```powershell
pytest -q
python scripts/benchmark_architecture.py --output results/architecture_metrics_full.json
python scripts/benchmark_architecture.py --output results/architecture_metrics_atgfr.json --comparison-seeds 7 17 27 --run-robustness-matrix
```

The scaling command measures 4, 8, 16, 32, and 50 task repositories without
pretending that precomputed latent-cloud routing is evidence that an encoder
learned those representations. See `ARCHITECTURE_REPORT.md` for the measured
results, mathematical assessment, deviations, and remaining research issues.

The current continual-learning research variant is enabled by default in the
new training entry point:

```powershell
python scripts/train.py --continual-method tgrsc
```

TGRSC (Topology-Gated Riemannian Subspace Consolidation) stores a compact
gradient subspace for the shared VIB encoder, gates retention by cached OT/TDA
task similarity, and projects only directionally conflicting updates. It is a
research variant related to gradient-projection methods, not a claim of
universal prior-art novelty; its matched multi-seed evidence and limitations
are recorded in `ARCHITECTURE_REPORT.md`.

The repository also includes ATGTR (Adaptive Task-Graph Trust-Region), enabled
with `python scripts/train.py --continual-method atgtr`. ATGTR solves a small
regularized quadratic projection only when a proposed gradient would violate a
bounded old-task loss-increase constraint. Its constraint rows are relation-
gated by the cached OT/TDA/Riemannian task graph, with a tight mean-gradient
budget and relaxed orthogonal-subspace budgets. The fixed four-cell stress grid
in `results/architecture_metrics_atgfr.json` covers 2/8 tasks crossed with 8/128
input features over three seeds; it is intentionally reported alongside the
baseline and TGRSC rather than as a selected best case. The default ATGTR path
uses a four-task compact reservoir so consolidation memory remains bounded.

The forgetting-focused variant is ATGFR (Adaptive Task-Graph Functional
Replay), enabled with `python scripts/train.py --continual-method atgfr`. It
stores a small class-balanced training coreset per task, distills
the old logits/features, and increases replay strength only when measured old
feature drift exceeds a budget. In the corrected five-seed new-version image
benchmark with mass-preserving relation normalization it reduced task-aware
forgetting from `0.3778` to `-0.2000` and raised task-aware accuracy from
`0.1000` to `0.6333`. The complete component ablation shows that functional
replay is supported, while every OT/TDA/thermostat contribution is not yet
universally beneficial.

To evaluate the new modular BONSAI variants and their matched current-core
controls on the same structured image stream, run:

```powershell
python scripts/benchmark_new_version.py --output results/new_version_image_comparison_5seed.json --seeds 7 17 27 37 47 --num-tasks 4 --classes-per-task 3 --train-samples-per-class 24 --test-samples-per-class 12 --image-size 32 --noise 0.1 --epochs 5 --batch-size 32
python scripts/benchmark_current_baselines.py --output results/current_backbone_baselines.json --seeds 7 17 27 37 47 --num-tasks 4 --classes-per-task 3 --train-samples-per-class 24 --test-samples-per-class 12 --image-size 32 --noise 0.1 --epochs 5 --batch-size 32
```

The second command runs EWC, SI, PackNet, and PNN on the same 199,148-parameter
current VIB/MLP core. PNN is task-aware-only and uses four frozen copies;
EWC/SI/PackNet have global class heads and task-free inference.
