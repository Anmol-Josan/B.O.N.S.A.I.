"""Lightweight, reproducible scaling and ablation experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter
from typing import Iterable

import torch
from torch import Tensor

from src.bonsai.continual import BONSAITrainer, InterferenceProtector, LossWeights
from src.bonsai.geometry import LowRankRiemannianMetric
from src.bonsai.metrics import (
    interference_drop,
    routing_margin,
    separation_ratio,
    within_task_radius,
)
from src.bonsai.repository import TaskRepository
from src.bonsai.atgtr import AdaptiveTaskGraphTrustRegion
from src.bonsai.replay import AdaptiveTaskGraphFunctionalReplay
from src.bonsai.rgsc import TopologyGatedRiemannianSubspaceConsolidator
from src.bonsai.router import TaskRouter
from src.bonsai.sheaf import SparseTaskSheaf
from src.bonsai.system import BONSAISystem
from src.data.synthetic_images import make_synthetic_image_tasks
from src.utils.metrics import average_accuracy, forgetting_measure


@dataclass
class ScalingRecord:
    task_count: int
    routing_accuracy: float
    candidate_count: float
    candidate_reduction: float
    retrieval_latency_ms: float
    routing_latency_ms: float
    repository_build_ms: float
    hierarchy_depth: int
    parameter_overhead: float
    routing_margin: float
    within_task_radius: float
    separation_ratio: float
    condition_number_max: float


def make_latent_task_clouds(
    num_tasks: int,
    latent_dim: int = 12,
    samples_per_task: int = 16,
    noise: float = 0.12,
    seed: int = 7,
) -> dict[int, Tensor]:
    """Create independent task clouds for repository/scaling mechanisms.

    This benchmark deliberately operates on latent clouds, not test labels or
    raw inputs. It measures repository/routing behavior without pretending to
    be evidence that an untrained encoder learned useful task structure.
    """

    if min(num_tasks, latent_dim, samples_per_task) < 1 or noise < 0.0:
        raise ValueError("invalid cloud dimensions or noise")
    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(num_tasks, latent_dim, generator=generator)
    centers = 2.5 * centers / centers.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return {
        task_id: centers[task_id].unsqueeze(0)
        + noise * torch.randn(samples_per_task, latent_dim, generator=generator)
        for task_id in range(num_tasks)
    }


def _build_components(latent_dim: int, top_k: int = 4) -> tuple[TaskRepository, LowRankRiemannianMetric, SparseTaskSheaf, TaskRouter]:
    repository = TaskRepository(
        latent_dim=latent_dim,
        coarse_dim=min(6, latent_dim),
        ot_projections=6,
        ot_samples=16,
        tda_bins=6,
        tda_samples=16,
        hierarchy_branching=4,
        hierarchy_leaf_capacity=4,
    )
    metric = LowRankRiemannianMetric(latent_dim=latent_dim, rank=min(2, latent_dim))
    sheaf = SparseTaskSheaf(latent_dim=latent_dim, stalk_dim=min(4, latent_dim))
    router = TaskRouter(repository, metric, sheaf, top_k=top_k, beam_width=2)
    return repository, metric, sheaf, router


def _flat_prototype_route(query: Tensor, clouds: dict[int, Tensor]) -> Tensor:
    task_ids = sorted(clouds)
    prototypes = torch.stack([clouds[task_id].mean(dim=0) for task_id in task_ids])
    distances = torch.cdist(query, prototypes)
    return torch.tensor(task_ids, dtype=torch.long)[distances.argmin(dim=1)]


def run_scaling_benchmark(
    task_counts: Iterable[int] = (4, 8, 16, 32, 50),
    latent_dim: int = 12,
    samples_per_task: int = 16,
    queries_per_task: int = 4,
    seed: int = 7,
) -> list[dict]:
    """Measure actual repository/routing quantities at increasing task counts."""

    records: list[dict] = []
    for task_count in task_counts:
        torch.manual_seed(seed + task_count)
        clouds = make_latent_task_clouds(
            task_count, latent_dim=latent_dim, samples_per_task=samples_per_task, seed=seed
        )
        repository, metric, sheaf, router = _build_components(latent_dim)
        system = BONSAISystem(
            input_dim=latent_dim,
            num_classes=max(1, task_count),
            hidden_dim=24,
            latent_dim=latent_dim,
            adapter_rank=min(2, latent_dim),
            metric_rank=min(2, latent_dim),
            sheaf_stalk_dim=min(4, latent_dim),
            repository_kwargs={
                "coarse_dim": min(6, latent_dim),
                "ot_projections": 6,
                "ot_samples": 16,
                "tda_bins": 6,
                "tda_samples": 16,
                "hierarchy_leaf_capacity": 4,
            },
            router_kwargs={"top_k": 4, "beam_width": 2},
        )
        build_start = perf_counter()
        for task_id, cloud in clouds.items():
            router.add_task(task_id, cloud)
            system.add_task(task_id, latent_samples=cloud)
        repository_build_ms = (perf_counter() - build_start) * 1000
        all_queries = []
        expected = []
        for task_id, cloud in clouds.items():
            all_queries.append(cloud[:queries_per_task])
            expected.extend([task_id] * min(queries_per_task, cloud.shape[0]))
        query = torch.cat(all_queries)
        expected_tensor = torch.tensor(expected)
        full_results = [router.route(cloud[:queries_per_task]) for cloud in clouds.values()]
        predictions = torch.cat([item.selected_task_ids.cpu() for item in full_results])
        candidate_counts = [item.candidate_comparisons / max(queries_per_task, 1) for item in full_results]
        retrieval_times = [item.retrieval_latency_ms / max(queries_per_task, 1) for item in full_results]
        total_times = [item.total_latency_ms / max(queries_per_task, 1) for item in full_results]
        task_latents = {task_id: cloud for task_id, cloud in clouds.items()}
        margin = routing_margin(system)
        radius = within_task_radius(system, task_latents)
        condition_numbers = metric.all_condition_numbers()
        record = ScalingRecord(
            task_count=task_count,
            routing_accuracy=float((predictions == expected_tensor).float().mean()),
            candidate_count=float(sum(candidate_counts) / len(candidate_counts)),
            candidate_reduction=1.0 - float(sum(candidate_counts) / len(candidate_counts)) / task_count,
            retrieval_latency_ms=float(sum(retrieval_times) / len(retrieval_times)),
            routing_latency_ms=float(sum(total_times) / len(total_times)),
            repository_build_ms=repository_build_ms,
            hierarchy_depth=repository.hierarchy.depth,
            parameter_overhead=system.parameter_overhead,
            routing_margin=margin,
            within_task_radius=radius,
            separation_ratio=separation_ratio(margin, radius),
            condition_number_max=max(condition_numbers.values(), default=0.0),
        )
        records.append(asdict(record))
    return records


def run_router_ablation(task_count: int = 8, latent_dim: int = 12, seed: int = 7) -> list[dict]:
    """Compare repository scoring components on the same held-out query clouds."""

    clouds = make_latent_task_clouds(task_count, latent_dim=latent_dim, samples_per_task=24, seed=seed)
    query = torch.cat([cloud[12:] for cloud in clouds.values()])
    expected = torch.tensor([task_id for task_id in clouds for _ in range(12)])
    variants = [
        ("flat_baseline", False, 0.0, 0.0, 0.0),
        ("prototype_router", True, 0.0, 0.0, 0.0),
        ("hierarchical_plus_ot", True, 0.2, 0.0, 0.0),
        ("hierarchical_plus_tda", True, 0.0, 0.2, 0.0),
        ("hierarchical_plus_sheaf", True, 0.0, 0.0, 0.05),
        ("full_bonsai", True, 0.2, 0.1, 0.05),
    ]
    records: list[dict] = []
    for name, hierarchical, ot_weight, tda_weight, sheaf_weight in variants:
        if not hierarchical:
            predictions = _flat_prototype_route(query, clouds)
            records.append(
                {"variant": name, "routing_accuracy": float((predictions == expected).float().mean()), "candidate_count": task_count}
            )
            continue
        torch.manual_seed(seed)
        repository, metric, sheaf, router = _build_components(latent_dim, top_k=4)
        router.ot_weight, router.tda_weight, router.sheaf_weight = ot_weight, tda_weight, sheaf_weight
        for task_id, cloud in clouds.items():
            router.add_task(task_id, cloud)
        results = [
            router.route(cloud[12:], context=cloud[:12]) for cloud in clouds.values()
        ]
        predictions = torch.cat([item.selected_task_ids.cpu() for item in results])
        records.append(
            {
                "variant": name,
                "routing_accuracy": float((predictions == expected).float().mean()),
                "candidate_count": sum(item.candidate_comparisons for item in results) / max(query.shape[0], 1),
                "hierarchy_depth": results[-1].hierarchy_depth,
                "routing_latency_ms": sum(item.total_latency_ms for item in results) / len(results),
            }
        )
    return records


def run_training_ablation(
    num_tasks: int = 4,
    input_dim: int = 8,
    classes_per_task: int = 2,
    samples_per_class: int = 16,
    epochs: int = 3,
    seed: int = 7,
    atgtr_trust_fraction: float = 0.5,
    atgtr_basis_weight: float = 0.25,
    atgtr_max_constraints: int = 24,
) -> list[dict]:
    """Matched continual-learning comparison on one fixed generated episode.

    Every variant sees the same task tensors, optimizer budget, and seed.  The
    optional protection methods differ only in their consolidation rule; this
    makes the result useful as an engineering comparison rather than a tuned
    leaderboard claim.
    """

    generator = torch.Generator().manual_seed(seed)
    class_centers = torch.randn(num_tasks * classes_per_task, input_dim, generator=generator)
    class_centers = 2.0 * class_centers / class_centers.norm(dim=1, keepdim=True).clamp_min(1e-8)
    tasks: list[tuple[int, Tensor, Tensor]] = []
    for task_id in range(num_tasks):
        task_inputs = []
        task_labels = []
        for local_class in range(classes_per_task):
            global_class = task_id * classes_per_task + local_class
            task_inputs.append(
                class_centers[global_class].unsqueeze(0)
                + 0.15 * torch.randn(samples_per_class, input_dim, generator=generator)
            )
            task_labels.append(torch.full((samples_per_class,), global_class, dtype=torch.long))
        task_inputs_tensor = torch.cat(task_inputs)
        task_labels_tensor = torch.cat(task_labels)
        permutation = torch.randperm(task_inputs_tensor.shape[0], generator=generator)
        tasks.append((task_id, task_inputs_tensor[permutation], task_labels_tensor[permutation]))
    records: list[dict] = []
    variants = (
        ("hierarchical_plus_vib", 1e-3, "diagonal"),
        ("hierarchical_no_vib", 0.0, "diagonal"),
        ("hierarchical_no_interference", 1e-3, "none"),
        ("hierarchical_tgrsc", 1e-3, "tgrsc"),
        ("hierarchical_atgtr", 1e-3, "atgtr"),
        ("hierarchical_atgfr", 1e-3, "atgfr"),
    )
    for variant, beta, protection_method in variants:
        torch.manual_seed(seed)
        system = BONSAISystem(
            input_dim=input_dim,
            num_classes=num_tasks * classes_per_task,
            hidden_dim=24,
            latent_dim=8,
            vib_beta=beta,
            adapter_rank=2,
            repository_kwargs={"coarse_dim": 5, "ot_projections": 4, "ot_samples": 12, "tda_bins": 5, "tda_samples": 12},
        )
        use_tgrsc = protection_method == "tgrsc"
        use_atgtr = protection_method == "atgtr"
        use_atgfr = protection_method == "atgfr"
        trainer = BONSAITrainer(
            system,
            learning_rate=3e-3,
            weights=LossWeights(
                interference=1.0
                if protection_method in {"tgrsc", "atgtr"}
                else (1.0 if protection_method == "diagonal" else 0.0),
                functional_replay=1.0 if use_atgfr else 0.0,
            ),
            protection=InterferenceProtector(
                0.0 if protection_method in {"none", "tgrsc", "atgtr", "atgfr"} else 1.0
            ),
            subspace_protection=(
                TopologyGatedRiemannianSubspaceConsolidator(
                    rank=4,
                    penalty_strength=1.0,
                    metric=system.metric,
                    shared_prefixes=("encoder.mu_layer.", "encoder.logvar_layer."),
                )
                if use_tgrsc
                else (
                    AdaptiveTaskGraphTrustRegion(
                        rank=4,
                        trust_fraction=atgtr_trust_fraction,
                        basis_weight=atgtr_basis_weight,
                        damping=1e-3,
                        max_constraints=atgtr_max_constraints,
                        metric=system.metric,
                        shared_prefixes=("encoder.mu_layer.", "encoder.logvar_layer."),
                    )
                    if use_atgtr
                    else None
                )
            ),
            functional_replay=(
                AdaptiveTaskGraphFunctionalReplay(
                    replay_per_task=8,
                    replay_strength=1.0,
                    distill_strength=1.0,
                    feature_strength=0.25,
                    temperature=2.0,
                    relation_floor=0.35,
                    drift_budget=0.05,
                    thermostat_gain=2.0,
                    max_multiplier=8.0,
                )
                if use_atgfr
                else None
            ),
            device="cpu",
        )
        start_time = perf_counter()
        accuracy_history: list[list[float]] = []
        for task_id, task_inputs, task_labels in tasks:
            trainer.fit_task(
                task_id,
                task_inputs,
                task_labels,
                epochs=epochs,
                batch_size=32,
            )
            current_accuracies = []
            with torch.no_grad():
                for seen_id, seen_inputs, seen_labels in tasks[: task_id + 1]:
                    output = system.model(seen_inputs, task_id=seen_id, sample=False)
                    current_accuracies.append(
                        float((output.logits.argmax(dim=1) == seen_labels).float().mean())
                    )
            accuracy_history.append(current_accuracies)
        final_accuracies = accuracy_history[-1]
        previous_best = [accuracy_history[index][index] for index in range(max(0, len(tasks) - 1))]
        previous_final = final_accuracies[: max(0, len(tasks) - 1)]
        records.append(
            {
                "variant": variant,
                "task_aware_average_accuracy": sum(final_accuracies) / len(final_accuracies),
                "interference_drop_percentage_points": interference_drop(previous_best, previous_final),
                "parameter_overhead": system.parameter_overhead,
                "max_condition_number": max(system.metric.all_condition_numbers().values()),
                "rgsc_anchor_count": len(trainer.subspace_protection.anchors)
                if trainer.subspace_protection is not None
                else 0,
                "consolidation_memory_elements": trainer.protection.stored_elements
                + (
                    trainer.subspace_protection.stored_elements
                    if trainer.subspace_protection is not None
                    else 0
                ),
                "replay_memory_elements": (
                    trainer.functional_replay.stored_elements
                    if trainer.functional_replay is not None
                    else 0
                ),
                "consolidation_memory_fraction": (
                    trainer.protection.stored_elements
                    + (
                        trainer.subspace_protection.stored_elements
                        if trainer.subspace_protection is not None
                        else 0
                    )
                    + (
                        trainer.functional_replay.stored_elements
                        if trainer.functional_replay is not None
                        else 0
                    )
                )
                / max(system.total_parameters, 1),
                "protection_method": protection_method,
                "input_dim": input_dim,
                "num_tasks": num_tasks,
                "classes_per_task": classes_per_task,
                "samples_per_class": samples_per_class,
                "epochs": epochs,
                "training_time_seconds": perf_counter() - start_time,
            }
        )
    return records


def run_continual_method_comparison(
    seeds: Iterable[int] = (7, 17, 27),
    num_tasks: int = 4,
    input_dim: int = 8,
    classes_per_task: int = 2,
    samples_per_class: int = 16,
    epochs: int = 3,
) -> list[dict]:
    """Run matched compact comparisons across seeds.

    The existing image benchmark remains the primary comparison against EWC,
    SI, PackNet, and PNN. This companion experiment isolates the new RGSC
    mechanism against BONSAI's diagonal-EWC protection under the same model,
    data, and optimizer budget.
    """

    records: list[dict] = []
    for seed in seeds:
        for record in run_training_ablation(
            num_tasks=num_tasks,
            input_dim=input_dim,
            classes_per_task=classes_per_task,
            samples_per_class=samples_per_class,
            epochs=epochs,
            seed=int(seed),
        ):
            records.append({"seed": int(seed), **record})
    return records


def run_continual_robustness_matrix(
    seeds: Iterable[int] = (7, 17, 27),
    samples_per_class: int = 8,
    epochs: int = 2,
) -> list[dict]:
    """Run a fixed, untuned grid spanning task and feature scales.

    The cells are chosen before execution and are deliberately crossed rather
    than selected after looking at results:

    * few tasks/few features: 2 tasks, 8 input features;
    * many tasks/few features: 8 tasks, 8 input features;
    * few tasks/many features: 2 tasks, 128 input features;
    * many tasks/many features: 8 tasks, 128 input features.

    All methods run on exactly the same data in each cell and use the same
    seeds.  This is a compact stress test, not a replacement for image or
    domain-specific benchmarks.
    """

    cells = (
        ("few_tasks_few_features", 2, 8),
        ("many_tasks_few_features", 8, 8),
        ("few_tasks_many_features", 2, 128),
        ("many_tasks_many_features", 8, 128),
    )
    records: list[dict] = []
    for cell, num_tasks, input_dim in cells:
        for record in run_continual_method_comparison(
            seeds=seeds,
            num_tasks=num_tasks,
            input_dim=input_dim,
            classes_per_task=2,
            samples_per_class=samples_per_class,
            epochs=epochs,
        ):
            records.append({"cell": cell, **record})
    return records


def run_modular_image_comparison(
    seeds: Iterable[int] = (7, 17, 27, 37, 47),
    num_tasks: int = 4,
    classes_per_task: int = 3,
    train_samples_per_class: int = 24,
    test_samples_per_class: int = 12,
    image_size: int = 32,
    noise: float = 0.1,
    epochs: int = 5,
    batch_size: int = 32,
    methods: Iterable[str] | None = None,
    atgfr_relation_mode: str = "full",
    atgfr_distill_strength: float = 1.0,
    atgfr_feature_strength: float = 0.25,
    atgfr_thermostat_gain: float = 2.0,
    atgfr_relation_floor: float = 0.35,
) -> list[dict]:
    """Evaluate the modular BONSAI variants on the structured image stream.

    The same generator is used by the matched current-core EWC, SI, PackNet,
    and PNN controls.  The modular system consumes flattened images with the
    current compact VIB/MLP encoder and adds its replay, routing, and adapter
    mechanisms on top.
    """

    if min(num_tasks, classes_per_task, train_samples_per_class, test_samples_per_class, epochs, batch_size) < 1:
        raise ValueError("task, sample, epoch, and batch dimensions must be positive")
    input_dim = 3 * image_size * image_size
    method_names = tuple(
        methods
        if methods is not None
        else (
            "new_bonsai_diagonal",
            "new_bonsai_tgrsc",
            "new_bonsai_atgtr",
            "new_bonsai_atgfr",
        )
    )
    valid_methods = {
        "new_bonsai_diagonal",
        "new_bonsai_tgrsc",
        "new_bonsai_atgtr",
        "new_bonsai_atgfr",
    }
    if not method_names or any(method not in valid_methods for method in method_names):
        raise ValueError(f"methods must be a non-empty subset of {sorted(valid_methods)}")
    if atgfr_relation_mode not in {"full", "no_ot", "no_tda", "euclidean", "uniform"}:
        raise ValueError("invalid ATGFR relation mode")
    records: list[dict] = []
    for seed in seeds:
        train_tasks, test_tasks = make_synthetic_image_tasks(
            num_tasks=num_tasks,
            classes_per_task=classes_per_task,
            train_samples_per_class=train_samples_per_class,
            test_samples_per_class=test_samples_per_class,
            image_size=image_size,
            noise=noise,
            seed=int(seed),
        )
        train_tensors = [
            (
                torch.stack([task[index][0] for index in range(len(task))]).flatten(start_dim=1),
                task.global_labels.clone(),
            )
            for task in train_tasks
        ]
        test_tensors = [
            (
                torch.stack([task[index][0] for index in range(len(task))]).flatten(start_dim=1),
                task.global_labels.clone(),
            )
            for task in test_tasks
        ]
        for method in method_names:
            torch.manual_seed(int(seed))
            system = BONSAISystem(
                input_dim=input_dim,
                num_classes=num_tasks * classes_per_task,
                hidden_dim=64,
                latent_dim=16,
                vib_beta=1e-3,
                adapter_rank=2,
                repository_kwargs={
                    "coarse_dim": 6,
                    "ot_projections": 4,
                    "ot_samples": 16,
                    "tda_bins": 6,
                    "tda_samples": 16,
                },
            )
            use_tgrsc = method == "new_bonsai_tgrsc"
            use_atgtr = method == "new_bonsai_atgtr"
            use_atgfr = method == "new_bonsai_atgfr"
            trainer = BONSAITrainer(
                system,
                learning_rate=3e-3,
                weights=LossWeights(
                    interference=1.0,
                    functional_replay=1.0 if use_atgfr else 0.0,
                ),
                protection=InterferenceProtector(
                    0.0 if use_tgrsc or use_atgtr or use_atgfr else 1.0
                ),
                subspace_protection=(
                    TopologyGatedRiemannianSubspaceConsolidator(
                        rank=4,
                        penalty_strength=1.0,
                        metric=system.metric,
                        shared_prefixes=("encoder.mu_layer.", "encoder.logvar_layer."),
                    )
                    if use_tgrsc
                    else (
                        AdaptiveTaskGraphTrustRegion(
                            rank=4,
                            trust_fraction=0.5,
                            basis_weight=0.25,
                            damping=1e-3,
                            max_constraints=24,
                            max_anchor_tasks=4,
                            metric=system.metric,
                            shared_prefixes=("encoder.mu_layer.", "encoder.logvar_layer."),
                        )
                        if use_atgtr
                        else None
                    )
                ),
                functional_replay=(
                    AdaptiveTaskGraphFunctionalReplay(
                        replay_per_task=8,
                        replay_strength=1.0,
                        temperature=2.0,
                        drift_budget=0.05,
                        thermostat_gain=atgfr_thermostat_gain,
                        max_multiplier=8.0,
                        relation_mode=atgfr_relation_mode,
                        distill_strength=atgfr_distill_strength,
                        feature_strength=atgfr_feature_strength,
                        relation_floor=atgfr_relation_floor,
                    )
                    if use_atgfr
                    else None
                ),
            )
            aware_history: list[list[float]] = []
            task_free_task_history: list[list[float]] = []
            route_history: list[float] = []
            start_time = perf_counter()
            for task_id, (inputs, labels) in enumerate(train_tensors):
                trainer.fit_task(task_id, inputs, labels, epochs=epochs, batch_size=batch_size)
                aware: list[float] = []
                routed_inputs: list[Tensor] = []
                routed_labels: list[Tensor] = []
                routed_tasks: list[Tensor] = []
                with torch.no_grad():
                    for seen_id, (test_inputs, test_labels) in enumerate(test_tensors[: task_id + 1]):
                        output = system.model(test_inputs, task_id=seen_id, sample=False)
                        aware.append(float((output.logits.argmax(dim=1) == test_labels).float().mean()))
                        routed_inputs.append(test_inputs)
                        routed_labels.append(test_labels)
                        routed_tasks.append(torch.full((test_inputs.shape[0],), seen_id, dtype=torch.long))
                    all_inputs = torch.cat(routed_inputs)
                    all_labels = torch.cat(routed_labels)
                    expected_tasks = torch.cat(routed_tasks)
                    route = system.route(all_inputs)
                    routed_predictions = all_labels.new_full(all_labels.shape, -1)
                    for selected_task in torch.unique(route.selected_task_ids).tolist():
                        positions = route.selected_task_ids == selected_task
                        output = system.model(all_inputs[positions], task_id=int(selected_task), sample=False)
                        routed_predictions[positions] = output.logits.argmax(dim=1)
                    task_free_task_history.append(
                        [
                            float(
                                (routed_predictions[expected_tasks == seen_id] == all_labels[expected_tasks == seen_id])
                                .float()
                                .mean()
                            )
                            for seen_id in range(task_id + 1)
                        ]
                    )
                    route_history.append(float((route.selected_task_ids == expected_tasks).float().mean()))
                aware_history.append(aware)
            records.append(
                {
                    "seed": int(seed),
                    "method": method,
                    "dataset": "synthetic_images",
                    "num_tasks": num_tasks,
                    "classes_per_task": classes_per_task,
                    "train_samples_per_class": train_samples_per_class,
                    "test_samples_per_class": test_samples_per_class,
                    "image_size": image_size,
                    "noise": noise,
                    "epochs": epochs,
                    "task_aware_average_accuracy": average_accuracy(aware_history[-1]),
                    "forgetting": forgetting_measure(aware_history),
                    "task_free_average_accuracy": average_accuracy(task_free_task_history[-1]),
                    "task_free_forgetting": forgetting_measure(task_free_task_history),
                    "task_free_route_accuracy": route_history[-1],
                    "parameter_overhead_percent": 100.0 * system.parameter_overhead,
                    "consolidation_memory_elements": trainer.protection.stored_elements
                    + (
                        trainer.subspace_protection.stored_elements
                        if trainer.subspace_protection is not None
                        else 0
                    ),
                    "replay_memory_elements": (
                        trainer.functional_replay.stored_elements
                        if trainer.functional_replay is not None
                        else 0
                    ),
                    "consolidation_memory_fraction": (
                        trainer.protection.stored_elements
                        + (
                            trainer.subspace_protection.stored_elements
                            if trainer.subspace_protection is not None
                            else 0
                        )
                        + (
                            trainer.functional_replay.stored_elements
                            if trainer.functional_replay is not None
                            else 0
                        )
                    )
                    / max(system.total_parameters, 1),
                    "training_time_seconds": perf_counter() - start_time,
                }
            )
    return records


def run_atgfr_component_ablation(
    seeds: Iterable[int] = (7, 17, 27),
    num_tasks: int = 4,
    classes_per_task: int = 3,
    train_samples_per_class: int = 24,
    test_samples_per_class: int = 12,
    image_size: int = 32,
    noise: float = 0.1,
    epochs: int = 5,
    batch_size: int = 32,
) -> list[dict]:
    """Run a predeclared factorial ablation of the functional replay terms.

    Every variant uses the same compact BONSAI model, task tensors, optimizer
    budget, and coreset size.  The descriptors are ablated through the
    repository relation mode, while replay terms and thermostat behavior are
    independently disabled.  This is intentionally separate from the main
    four-method comparison so a component claim cannot be inferred from a
    route-only ablation.
    """

    variants = (
        ("labels_only", "uniform", 0.0, 0.0, 0.0, 1.0),
        ("labels_logits", "uniform", 1.0, 0.0, 0.0, 1.0),
        ("labels_features", "uniform", 0.0, 0.25, 0.0, 1.0),
        ("fixed_full", "full", 1.0, 0.25, 0.0, 0.35),
        ("no_ot", "no_ot", 1.0, 0.25, 2.0, 0.35),
        ("no_tda", "no_tda", 1.0, 0.25, 2.0, 0.35),
        ("euclidean_relation", "euclidean", 1.0, 0.25, 2.0, 0.35),
        ("full_atgfr", "full", 1.0, 0.25, 2.0, 0.35),
    )
    records: list[dict] = []
    for name, relation_mode, distill, feature, thermostat, floor in variants:
        variant_records = run_modular_image_comparison(
            seeds=seeds,
            num_tasks=num_tasks,
            classes_per_task=classes_per_task,
            train_samples_per_class=train_samples_per_class,
            test_samples_per_class=test_samples_per_class,
            image_size=image_size,
            noise=noise,
            epochs=epochs,
            batch_size=batch_size,
            methods=("new_bonsai_atgfr",),
            atgfr_relation_mode=relation_mode,
            atgfr_distill_strength=distill,
            atgfr_feature_strength=feature,
            atgfr_thermostat_gain=thermostat,
            atgfr_relation_floor=floor,
        )
        for record in variant_records:
            records.append({**record, "variant": name})
    return records
