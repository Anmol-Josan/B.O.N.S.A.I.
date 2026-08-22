# Topological--Riemannian BONSAI implementation report

## Architecture implemented

The modular implementation lives under `src/bonsai/`.

| Component | Location | Implementation |
|---|---|---|
| VIB | `vib.py` | Diagonal Gaussian posterior, reparameterized latent, analytic KL to N(0,I). |
| Hierarchical retrieval | `hierarchy.py`, `repository.py` | Incremental split-on-overflow balanced tree, task prototypes/coarse embeddings, beam retrieval. |
| OT | `ot.py` | Cached fixed-projection sliced-Wasserstein-1 descriptors with representative samples. |
| TDA | `tda.py` | Cached exact H0 Vietoris--Rips persistence summary from an MST, histogram/statistics descriptor. |
| Riemannian routing | `geometry.py`, `router.py` | Bounded SPD diagonal base metric plus PSD low-rank task update; local `Log_p(z)=z-p`; candidate argmin. |
| Sheaf | `sheaf.py` | Sparse nearest-task graph, shared restriction map, endpoint scalar edge gates, compatibility energy. |
| Adapter | `adapters.py`, `model.py` | Shared `U diag(d_k) V^T` basis; rank-sized task coefficient vectors; old coefficients can be frozen. |
| Continual learning | `continual.py`, `rgsc.py`, `atgtr.py`, `replay.py` | Frozen old task coefficients, diagonal protection, topology-gated subspace consolidation, adaptive trust-region projection, and task-graph functional replay with drift control. |
| Composition/evaluation | `system.py`, `evaluation.py` | Full pipeline, scaling metrics, router ablations, and small training ablation. |

The task objective is implemented in `BONSAITrainer` as task CE plus independently
weighted route, VIB, geometry, OT consistency, sheaf, and interference terms.
The predictive loss is the practical surrogate for `-I(Z;T)`; the VIB KL is the
variational upper-bound surrogate for `I(X;Z)`.

## Research iteration: TGRSC, ATGTR, and ATGFR

The new continual-learning variant is **Topology-Gated Riemannian Subspace
Consolidation (TGRSC)** in `rgsc.py`. It is deliberately scoped to the shared
VIB encoder, not the classifier rows for classes that have not arrived yet.
After each task it stores a rank-limited SVD basis of observed encoder-gradient
vectors and a reference parameter point. For a new task, retention uses

```text
L_TGRSC = Σ_u w(t,u) || diag(s_u)^(1/2) Q_u^T(θ - θ_u) ||²
```

where `w(t,u)` is a floor-bounded similarity from cached sliced-Wasserstein and
H0 persistence distances, multiplied by a bounded local Riemannian prototype
gate. Its gradient step removes only signed components
that oppose the old task's mean gradient; compatible directions remain
plastic. This combines task-repository geometry with low-rank conflict-aware
consolidation and is usable from `scripts/train.py --continual-method tgrsc`.

The subspace-projection part is related to existing methods such as
[EWC](https://doi.org/10.1073/pnas.1611835114),
[Orthogonal Gradient Descent](https://arxiv.org/abs/1910.07104), and
[Gradient Projection Memory](https://arxiv.org/abs/2103.09762). Therefore this
repository does **not** claim that low-rank gradient projection itself is new;
the proposed BONSAI contribution is the representation-only, signed-conflict,
OT/TDA-gated combination. A formal prior-art novelty search has not been done.

The second research iteration is **Adaptive Task-Graph Trust-Region (ATGTR)**
in `atgtr.py`. It treats gradient preservation as a small
inequality-constrained quadratic program. Let `g` be the current shared-
encoder gradient, `r(t,u)` the cached relation between the new task `t` and old
task `u`, and `a_u` the normalized mean old-task gradient. The proposed update
is `d = -g`; ATGTR protects the first-order old-task loss change with

```text
        a_u^T d >= -epsilon_u
```

and therefore solves the equivalent projection constraints `A_t g' <=
epsilon_t`. The rows of `A_t` contain `sqrt(r(t,u)) a_u` plus orthogonal
residual directions from the old low-rank SVD basis. The mean row uses trust
fraction `tau=0.5`; residual rows use `tau * beta * sqrt(s_u)` with
`beta=0.25`, so an arbitrary within-task basis axis cannot suppress learning
as strongly as the actual mean loss gradient. The active-set approximation is

```text
g' = g - A_t^T lambda
(A_t A_t^T + mu I) lambda = [A_t g - epsilon_t]_+
```

with `mu=1e-3` and at most 24 rows. The relation multiplies the row but not
the tolerance, so unrelated tasks become inactive instead of suffering a
similarity scale cancellation. This is a constant-size solve independent of
the number of model parameters; the method stores the same rank-limited task
anchors as TGRSC. The default CLI uses `compact_memory=True` and a four-task
reservoir: after extracting each mean-gradient trust direction it discards the
full displacement reference and residual basis, bounding the deployment memory
while retaining the richer basis form as `compact_memory=False`. ATGTR is a new BONSAI research method, not a claim that
trust-region or gradient-projection continual learning is new in isolation.
Its closest conceptual relatives remain
[EWC](https://doi.org/10.1073/pnas.1611835114),
[Orthogonal Gradient Descent](https://arxiv.org/abs/1910.07104), and
[Gradient Projection Memory](https://arxiv.org/abs/2103.09762).

The third iteration is **Adaptive Task-Graph Functional Replay (ATGFR)** in
`replay.py`. It addresses the failure that parameter protection alone does not
preserve the old input-to-output function. After each task, ATGFR keeps a
class-balanced training-only coreset and stores the task's logits and latent
features. During later tasks it adds

```text
L_ATGFR = Σ_u w(t,u) h_u [ CE(y_u, f_theta(x_u))
                           + T² KL(p_old^T || p_new^T)
                           + γ ||z_old - z_new||² ]
```

where `w(t,u)` is the task-graph relation and the adaptive thermostat is

```text
h_u = clip(1 + k [ drift_u / budget - 1 ]_+, 1, h_max).
```

Thus replay pressure increases only when the measured old-task feature drift
exceeds its budget. The default stores eight balanced examples per task and
uses temperature `T=2`, feature coefficient `gamma=0.25`, a 5% drift budget,
and a relation floor of `0.35`. It is a bounded coreset method, not test-data
rehearsal: all stored examples come from the completed task's training input.
Replay and distillation are established continual-learning ideas, so the
novel BONSAI contribution is their adaptive task-graph weighting and
drift-triggered thermostat. It is available from
`scripts/train.py --continual-method atgfr`; no claim of first prior art is
made.

## Tests

Command:

```text
pytest -q
```

Result on the local CPU environment: **94 passed**, one unrelated PyNVML
deprecation warning, about **65 seconds** in the final run. The added architecture suite is
18 tests and covers VIB stochasticity/KL, OT, TDA degeneracies, hierarchy
insertion/retrieval, repository serialization, SPD/conditioning, sheaf energy,
adapter overhead, composed routing, continual insertion/freezing, interference
penalty, RGSC/ATGTR/ATGFR projections, scaling, and router ablations. The pre-existing
legacy suite was
69/69 before the redesign and remains passing.

## Scaling measurements

Command:

```text
python scripts/benchmark_architecture.py --output results/architecture_metrics_full.json
```

These are one seeded CPU run (`seed=7`) on deterministic synthetic latent task
clouds. Candidate count and latency are per query; repository build is for all
tasks at that row. Routing is not encoder-training evidence.

| Tasks | Route acc. | Candidates | Reduction | Retrieval ms | Route ms | Build ms | Depth | ρ_N | m_N | r_N | m/(2r) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1.000 | 4.00 | 0.000 | 0.369 | 7.957 | 76.0 | 1 | 0.0245 | 2.884 | 0.673 | 2.143 |
| 8 | 0.875 | 4.50 | 0.438 | 0.487 | 5.666 | 117.6 | 3 | 0.0569 | 2.862 | 0.647 | 2.213 |
| 16 | 0.984 | 8.44 | 0.473 | 0.885 | 7.969 | 253.6 | 4 | 0.1138 | 2.468 | 0.698 | 1.769 |
| 32 | 0.867 | 15.77 | 0.507 | 1.742 | 9.496 | 494.6 | 5 | 0.2042 | 1.770 | 0.697 | 1.269 |
| 50 | 0.890 | 25.52 | 0.490 | 2.559 | 8.753 | 909.6 | 6 | 0.2801 | 1.484 | 0.687 | 1.080 |

The hierarchy reduces evaluated candidates once the repository exceeds its
four-task leaves, but the implementation does not claim O(log N) total routing:
the measured candidate work is approximately half of N at 50 tasks. The
conditional geometric criterion `2r_N < m_N` is satisfied in this seeded
latent-cloud diagnostic, but that is not a guarantee for learned data.
Maximum observed metric condition number was about 1.0124, and all tested
metrics were positive definite.

## Ablations

On the same 8-task latent-cloud episode, the flat prototype baseline routed at
1.000 accuracy with 8 candidates. The approximate hierarchy variants and the
full pipeline routed at 0.875 accuracy with 4.5 candidates per query. OT, TDA,
and sheaf scoring did not improve this single seed; TDA and sheaf added compute.
That is a measured negative result, not a claim that those components are
universally useless.

The matched 4-task comparison uses three fixed seeds (`7, 17, 27`), four tasks,
two classes per task, 16 examples per class, and three epochs. The same
generated episode and optimizer budget are reused for every variant:

| Variant | Task-aware accuracy | Interference drop | Interpretation |
|---|---:|---:|---|
| BONSAI + diagonal protection | 0.1901 ± 0.1624 | 0.0417 | Baseline |
| BONSAI + TGRSC | 0.1953 ± 0.1595 | 0.0347 | +0.0052 accuracy; lower mean interference |
| BONSAI + ATGTR | 0.1901 ± 0.1624 | 0.0417 | Matches baseline on this compact episode |

The TGRSC improvement is small relative to seed variance. ATGTR is stable here,
but does not beat TGRSC or diagonal protection. These are engineering signals,
not a solved continual-learning claim.

To test for cherry-picking, the fixed robustness command also crosses task count
and feature count over the same three seeds (`7, 17, 27`), with eight examples
per class and two epochs in every cell. The six methods in the harness are
diagonal protection, no VIB, no interference, TGRSC, ATGTR, and ATGFR; the table
below reports the four continual-learning protection variants as mean accuracy
± seed standard deviation. The corresponding raw records, including runtime,
overhead, and memory, are in `results/architecture_metrics_atgfr.json`.

| Cell | Tasks × input features | Diagonal | TGRSC | ATGTR | ATGFR |
|---|---:|---:|---:|---:|---:|
| Few/few | 2 × 8 | 0.2396 ± 0.0180 | 0.2396 ± 0.0180 | 0.2396 ± 0.0180 | 0.3333 ± 0.1443 |
| Many/few | 8 × 8 | 0.0885 ± 0.0861 | 0.0807 ± 0.0783 | 0.0885 ± 0.0861 | 0.1198 ± 0.0496 |
| Few/many | 2 × 128 | 0.2188 ± 0.0827 | 0.2188 ± 0.0827 | 0.2188 ± 0.0827 | 0.2604 ± 0.0955 |
| Many/many | 8 × 128 | 0.1250 ± 0.0435 | 0.1250 ± 0.0435 | 0.1250 ± 0.0435 | 0.2630 ± 0.0771 |

The grid is intentionally underpowered as a learning benchmark—two epochs and
small synthetic episodes—so the absolute accuracies are not real-world model
quality claims. Its value is that no cell was omitted after inspecting results:
ATGTR never catastrophically diverged after the trust-budget, relation-gate,
and memory-cap fixes, but it did not produce a broad accuracy win. TGRSC
reduced mean interference in the many-task/many-feature cell from 0.0268 to
0.0208, while its retained anchor state was 5.28× the model parameter count;
ATGTR was 0.42× in that same cell. ATGFR improved accuracy in all four cells
and produced negative mean interference in all four cells. After removing
disabled EWC state, its replay memory is lower than TGRSC in the few-task
cells but grows with input width and task count. The existing image
benchmark below remains the only comparison against the named baselines in
this report.

The same raw records measure consolidation state as float elements divided by
model parameters (the ratio is independent of the four-byte dtype assumption):

| Cell | Diagonal | TGRSC | ATGTR compact | ATGFR |
|---|---:|---:|---:|---:|
| Few/few | 1.845× | 4.015× | 1.003× | 0.421× |
| Many/few | 1.776× | 13.295× | 1.660× | 2.191× |
| Few/many | 1.966× | 0.871× | 0.218× | 0.613× |
| Many/many | 1.944× | 3.334× | 0.416× | 2.547× |

### Current-core classical comparison

The classical baselines now use the current VIB/MLP prediction core rather
than a separate architecture. The core has 199,148 parameters, hidden width
64, latent width 16, adapter rank 2, learning rate `3e-3`, five epochs, batch
size 32, and the same five seeds and structured-image tensors as the modular
BONSAI comparison. EWC and SI are online bounded-state variants; PackNet
freezes 50% of currently free weights after each task; PNN adds one frozen copy
of the same core per task.

| Method | Task-aware accuracy | Forgetting | Task-free accuracy | Parameter overhead | Memory / parameters |
|---|---:|---:|---:|---:|---:|
| EWC (current core) | 0.2500 ± 0.0000 | 1.0000 ± 0.0000 | 0.2500 ± 0.0000 | 0% | 2.000× |
| SI (current core) | 0.2500 ± 0.0000 | 1.0000 ± 0.0000 | 0.2500 ± 0.0000 | 0% | 2.000× |
| PackNet (current core) | 0.3333 ± 0.1054 | 0.0222 ± 0.0831 | 0.3333 ± 0.1054 | 0% | 1.000× |
| PNN (current core) | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | — | 300% | 0.000× |
| **ATGFR** | **0.6333 ± 0.1000** | **−0.2000 ± 0.1296** | **0.6333 ± 0.1000** | **0.014%** | **0.498×** |

PNN is strongest on task-aware accuracy but pays 300% column growth and has
no task-free path. ATGFR is lower than PNN on task-aware accuracy while
retaining task-free inference and negative mean forgetting. EWC and SI reach
chance-level final accuracy under this fixed low-data schedule; this is a
protocol result, not a universal claim about their method families. All 20
per-seed rows are retained in `results/current_backbone_baselines.json`.

### New modular BONSAI on the same image stream

The new version is now evaluated on the same structured image generator and
task order, using five seeds (`7, 17, 27, 37, 47`), four tasks, three classes per task,
24 training examples per class, 12 test examples per class, noise `0.1`, and
five epochs. The new modular model flattens the `3 × 32 × 32` image into the
same current VIB/MLP prediction core used by the matched EWC/SI/PackNet/PNN
benchmark above. The modular rows add BONSAI's repository, geometry, routing,
and task-adapter state; the classical rows are therefore capacity-matched at
the prediction-core level and differ only in continual-learning mechanism.

| New-version method | Task-aware accuracy | Forgetting | Task-free accuracy | Route accuracy | Parameter overhead | Consolidation memory |
|---|---:|---:|---:|---:|---:|---:|
| New modular BONSAI + diagonal | 0.1000 ± 0.0333 | 0.3778 ± 0.1133 | 0.1000 ± 0.0333 | 0.2639 ± 0.0739 | 0.014% | 1.999× |
| New modular BONSAI + TGRSC | 0.1181 ± 0.0426 | 0.3537 ± 0.0812 | 0.1125 ± 0.0363 | 0.2986 ± 0.0643 | 0.014% | 0.251× |
| New modular BONSAI + ATGTR | 0.1000 ± 0.0333 | 0.3778 ± 0.1133 | 0.1000 ± 0.0333 | 0.2639 ± 0.0739 | 0.014% | 0.042× |
| New modular BONSAI + ATGFR | 0.6333 ± 0.1000 | -0.2000 ± 0.1296 | 0.6333 ± 0.1000 | 0.5667 ± 0.0624 | 0.014% | 0.498× |

The new-version per-seed records are:

| Method | Seed | Task-aware | Forgetting | Task-free | Route accuracy |
|---|---:|---:|---:|---:|---:|
| New modular + diagonal | 7 | 0.0833 | 0.5556 | 0.0833 | 0.3958 |
| New modular + diagonal | 17 | 0.0833 | 0.3333 | 0.0833 | 0.2569 |
| New modular + diagonal | 27 | 0.1667 | 0.3333 | 0.1667 | 0.2500 |
| New modular + diagonal | 37 | 0.0833 | 0.2222 | 0.0833 | 0.2500 |
| New modular + diagonal | 47 | 0.0833 | 0.4444 | 0.0833 | 0.1667 |
| New modular + TGRSC | 7 | 0.1736 | 0.4352 | 0.1458 | 0.4097 |
| New modular + TGRSC | 17 | 0.0833 | 0.3333 | 0.0833 | 0.3333 |
| New modular + TGRSC | 27 | 0.1667 | 0.3333 | 0.1667 | 0.2500 |
| New modular + TGRSC | 37 | 0.0833 | 0.2222 | 0.0833 | 0.2500 |
| New modular + TGRSC | 47 | 0.0833 | 0.4444 | 0.0833 | 0.2500 |
| New modular + ATGTR | 7 | 0.0833 | 0.5556 | 0.0833 | 0.3958 |
| New modular + ATGTR | 17 | 0.0833 | 0.3333 | 0.0833 | 0.2569 |
| New modular + ATGTR | 27 | 0.1667 | 0.3333 | 0.1667 | 0.2500 |
| New modular + ATGTR | 37 | 0.0833 | 0.2222 | 0.0833 | 0.2500 |
| New modular + ATGTR | 47 | 0.0833 | 0.4444 | 0.0833 | 0.1667 |
| New modular + ATGFR | 7 | 0.6667 | -0.3333 | 0.6667 | 0.5000 |
| New modular + ATGFR | 17 | 0.7500 | -0.3333 | 0.7500 | 0.5833 |
| New modular + ATGFR | 27 | 0.5833 | -0.2222 | 0.5833 | 0.5833 |
| New modular + ATGFR | 37 | 0.5000 | 0.0000 | 0.5000 | 0.6667 |
| New modular + ATGFR | 47 | 0.5833 | -0.1111 | 0.5833 | 0.5000 |

ATGFR is the first new-version method in this report that materially reduces
forgetting: it moves task-aware forgetting from `0.3778` to `-0.2000` and raises
task-aware and task-free accuracy to `0.6333`. It uses eight training exemplars
per completed task plus stored logits/features, which accounts for its bounded
`0.498×`
consolidation memory after disabled diagonal state was removed. TGRSC is directionally better than the new
modular diagonal control, while ATGTR is neutral. The absolute scores are still
architecture-specific synthetic-image evidence, not production-level accuracy.
The raw records for this three-seed report are in
`results/new_version_image_comparison.json`. The primary five-seed artifact is
`results/new_version_image_comparison_5seed.json`; its run is reproducible with:

```text
python scripts/benchmark_new_version.py --output results/new_version_image_comparison_5seed.json --seeds 7 17 27 37 47 --num-tasks 4 --classes-per-task 3 --train-samples-per-class 24 --test-samples-per-class 12 --image-size 32 --noise 0.1 --epochs 5 --batch-size 32
```

The current-core baseline comparison can be rerun with:

```text
python scripts/benchmark_current_baselines.py --output results/current_backbone_baselines.json --seeds 7 17 27 37 47 --num-tasks 4 --classes-per-task 3 --train-samples-per-class 24 --test-samples-per-class 12 --image-size 32 --noise 0.1 --epochs 5 --batch-size 32
```

## Mathematical evaluation and deviations

- **VIB:** partially supported. The variational objective and stochastic path
  are genuine, but the local ablation did not separate VIB from no-VIB.
- **Hierarchy:** supported for candidate reduction after splits; retrieval is
  approximate because beam search can drop the globally nearest task.
- **OT:** implemented and cached; usefulness beyond prototypes was not
  supported by the one-seed ablation.
- **TDA:** partially supported as a mathematically valid H0 persistence
  descriptor. Higher-dimensional topology and Mapper are intentionally not in
  normal inference.
- **Riemannian geometry:** supported numerically for SPD and bounded
  conditioning. Global geodesic claims are not made; routing uses a local chart.
- **Margin criterion:** supported only as an empirical diagnostic in the
  latent-cloud run, not as an automatic correctness theorem.
- **Sheaf:** implemented as a sparse compatibility regularizer. Its transfer
  benefit is not supported by the current ablation.
- **Shared low-rank adapter:** parameter sharing and incremental coefficient
  insertion are supported by shape/count tests. The requested `ρ_N < 0.005`
  target was **not met** in the measured small model: it was 0.0245 at 4 tasks
  and 0.2801 at 50 tasks. The sheaf was revised from dense per-edge maps to a
  shared map plus scalar edge gates to reduce this growth, but the target still
  needs a larger backbone or further parameter-budget design.
- **Continual learning:** TGRSC is a usable low-rank, representation-only,
  conflict-aware extension. The three-seed matched comparison shows a small
  accuracy gain and lower mean interference, but high variance prevents a
  strong claim. The new modular image run improves from `0.1000` to
  `0.1181` with TGRSC over five seeds, while ATGTR remains neutral at
  `0.1000`; these models
  are compact current-core MLPs. ATGTR is a bounded, constant-size
  trust-region projection; after correcting its budget and relation gate it is
  stable across the fixed four-cell grid, but it does not beat the other
  methods there. After mass-preserving normalization, ATGFR moves new-version
  forgetting from `0.3778` to `-0.2000` and improves accuracy in every
  robustness-grid cell. The matched CIFAR-100 mechanism study now compares
  ATGFR against ER, DER++, and ER-ACE under the same compact backbone and
  exemplar count; it is a low-data pilot, not a universal accuracy claim.

## Compute and remaining issues

All reported runs used the reliable CPU path. CUDA was unavailable in the
installed PyTorch environment, and no additional GPU stack was introduced.
The 50-task repository benchmark took under a second to construct and roughly
single-digit milliseconds per four-query routing batch on this machine; the
full pytest run is reported above. The remaining research work is to train the
VIB on a non-leaking real or carefully designed continual dataset, compare the
current-core classical baselines and ATGFR on full-data real streams with
matched budgets, evaluate randomized task orders and longer streams, tune no thresholds
against the evaluation set, and measure whether OT/TDA/sheaf improve transfer or interference rather
than only adding latency, and whether either projection transfers to the
existing ResNet-18 image runner.

## Review expansion: real data, modern replay, ablations, orders, and cost

The critique identified five missing evidence classes. They are now covered by
two new artifacts plus a separate matched-current-core baseline artifact:

1. `results/current_backbone_baselines.json` contains 20 runs: EWC, SI,
   PackNet, and PNN on the current 199,148-parameter VIB/MLP core, using the
   same five seeds and structured image stream as the modular comparison.
2. `results/real_replay_cifar100_review.json` contains 24 runs: ER, DER++,
   ER-ACE, and ATGFR on ten Split-CIFAR-100 tasks, two seeds, and three task
   orders. All four use the same 122,884-parameter CompactCIFARNet, 20 images
   per task, two epochs, batch size 64, and Adam at `1e-3`.
3. `results/atgfr_component_ablation.json` contains eight replay variants over
   three seeds, including labels-only, labels+logits, labels+features, fixed
   full, no OT, no H0, Euclidean relation, and full ATGFR.

### Matched real-image results

| Method | Accuracy | Forgetting | Replay scalars | Wall time (s) |
|---|---:|---:|---:|---:|
| ER | 0.0321 ± 0.0033 | 0.0038 ± 0.0054 | 615880 | 17.44 ± 0.31 |
| DER++ | 0.0274 ± 0.0024 | 0.0049 ± 0.0049 | 635880 | 18.06 ± 1.25 |
| ER-ACE | 0.0311 ± 0.0032 | 0.0265 ± 0.0095 | 615880 | 17.80 ± 0.77 |
| ATGFR | **0.0358 ± 0.0052** | **0.0037 ± 0.0040** | 661480 | 18.91 ± 1.11 |

The absolute accuracy is low because the experiment uses eight training images
per class and two epochs. It resolves the real-image and matched-control gap,
not the stronger claim that ATGFR is a high-accuracy CIFAR-100 solution.
ATGFR stores 7.4% more scalar replay state than ER because logits and features
are included; the equal image count alone would hide that cost.

### Component and order findings

The three-seed component means are: labels-only 0.6667 accuracy and -0.3704
forgetting; fixed full 0.6782 and -0.3858; Euclidean relation 0.6111 and
-0.3333; and full ATGFR 0.5556 and -0.2593. The replay mechanism is supported,
but the complete OT/H0/thermostat stack is not the best variant in this stream.
The implementation now mass-normalizes relation weights so the graph allocates
replay without silently changing its total coefficient.

On the real-image orders, ATGFR accuracy is 0.0298, 0.0405, and 0.0370 for
the identity and two deterministic shuffles; forgetting is 0.0050, 0.0008,
and 0.0053. These values are retained in Figure 10 and the raw JSON.

### What remains genuinely open

Full-data and longer-schedule CIFAR-100/TinyImageNet studies, equal-byte
comparisons for the current-core classical baselines, compressed or private
replay, GPU energy, and out-of-distribution route abstention remain future
work. They are not described as solved by the current paper.
