# BONSAI research scope

This repository contains a research implementation and an experiment sandbox.
The sandbox is larger than the evidence-supported claim. The following scope is
the contract for the current paper and for new experiments.

## Minimum viable scientific claim

Under a bounded-memory, task-incremental stream, ATGFR is a promising
functional-replay mechanism for reducing forgetting relative to the matched
diagonal BONSAI control. The claim is limited to the structured-image study
and the low-data Split-CIFAR-100 pilot currently checked into `results/`.

This is not a claim of state-of-the-art continual learning, universal
superiority, or solved task-free/class-incremental routing. Task-free accuracy,
synthetic latent-cloud scaling, and architecture stress tests are diagnostics,
not primary evidence for the claim above.

## Canonical evaluation path

The canonical path is the five-seed structured-image comparison plus the
matched current-core baselines:

```powershell
python scripts/benchmark_new_version.py --output results/new_version_image_comparison_5seed.json --seeds 7 17 27 37 47 --num-tasks 4 --classes-per-task 3 --train-samples-per-class 24 --test-samples-per-class 12 --image-size 32 --noise 0.1 --epochs 5 --batch-size 32
python scripts/benchmark_current_baselines.py --output results/current_backbone_baselines.json --seeds 7 17 27 37 47 --num-tasks 4 --classes-per-task 3 --train-samples-per-class 24 --test-samples-per-class 12 --image-size 32 --noise 0.1 --epochs 5 --batch-size 32
```

The primary metrics are task-aware average accuracy and peak-before-final
forgetting. Task-free accuracy and route accuracy are secondary diagnostics;
they must always be reported separately because they evaluate an additional
router rather than the same operational setting.

The real-image pilot is a required secondary check, not a replacement for a
larger standard-dataset study:

```powershell
python scripts/benchmark_real_replay.py --output results/real_replay_cifar100_review.json --data-root data/cifar100 --seeds 7 17 27 --order-count 3 --epochs 2 --train-samples-per-class 8 --test-samples-per-class 20 --memory-per-task 20 --batch-size 64
```

## Evidence status

| Component or result | Status | How it may be described |
| --- | --- | --- |
| ATGFR functional replay | Supported in the current matched studies | Primary scoped contribution |
| VIB, adapters, repository, hierarchy, router | Implemented substrate | Engineering design; not individually novel by default |
| TGRSC and ATGTR | Exploratory variants | Ablations/future work unless independently validated |
| OT, TDA, sheaf, local geometry | Partially supported and sometimes neutral or harmful | Diagnostic components; do not imply benefit without the relevant ablation |
| Synthetic latent-cloud scaling | Diagnostic | Numerical/scaling evidence only; not learned-representation evidence |
| Task-free routing | Open problem | Report separately and conservatively |

“Implemented” means that code exists and tests or a run exercise it. “Supported”
means that a predeclared comparison and ablation provide evidence for the
associated claim. These terms are intentionally not interchangeable.

## Reporting rules for new results

Every new result should record the dataset, stream/order, seeds, training
budget, backbone, parameter count, replay memory, wall-clock time, and whether
the task ID was available at inference. Report mean and uncertainty across
seeds when there is more than one seed. Preserve negative controls and failed
ablations beside the selected result; do not move them to an unreferenced
output directory.

Exploratory scripts remain useful, but they should be labeled as such in their
output directory and should not silently replace the canonical result files.
