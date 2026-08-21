from __future__ import annotations

import torch

from src.bonsai.adapters import SharedLowRankAdapter
from src.bonsai.atgtr import AdaptiveTaskGraphTrustRegion
from src.bonsai.continual import BONSAITrainer, InterferenceProtector
from src.bonsai.evaluation import run_modular_image_comparison, run_router_ablation, run_scaling_benchmark
from src.bonsai.geometry import LowRankRiemannianMetric
from src.bonsai.hierarchy import TaskHierarchy
from src.bonsai.metrics import separation_ratio
from src.bonsai.ot import SlicedWasserstein
from src.bonsai.repository import TaskRepository
from src.bonsai.rgsc import TopologyGatedRiemannianSubspaceConsolidator
from src.bonsai.sheaf import SparseTaskSheaf
from src.bonsai.system import BONSAISystem
from src.bonsai.tda import ZeroDimensionalPersistence
from src.bonsai.vib import VIBEncoder


def test_vib_is_stochastic_in_training_and_finite() -> None:
    torch.manual_seed(3)
    encoder = VIBEncoder(input_dim=5, hidden_dim=12, latent_dim=3, beta=0.01)
    inputs = torch.randn(7, 5)
    encoder.train()
    first = encoder(inputs)
    second = encoder(inputs)
    assert first.z.shape == (7, 3)
    assert not torch.allclose(first.z, second.z)
    assert torch.isfinite(first.kl)
    assert encoder.information_loss(first) >= 0
    assert torch.allclose(encoder.deterministic(inputs), encoder.deterministic(inputs))


def test_sliced_wasserstein_is_zero_for_identical_descriptor_and_safe_for_one_point() -> None:
    scorer = SlicedWasserstein(latent_dim=3, projections=4, representative_samples=5)
    points = torch.randn(6, 3)
    descriptor = scorer.build(points)
    assert torch.allclose(scorer.distance(descriptor, descriptor), torch.zeros(()))
    assert torch.isfinite(scorer.prototype_distance(points[:1], descriptor))
    try:
        scorer.build(torch.tensor([[float("nan"), 0.0, 0.0]]))
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite OT inputs must be rejected")


def test_h0_persistence_descriptor_handles_degenerate_clouds() -> None:
    tda = ZeroDimensionalPersistence(bins=5, representative_samples=8)
    descriptor = tda.build(torch.zeros(1, 4))
    duplicate = tda.build(torch.zeros(8, 4))
    assert descriptor.vector.shape == (12,)
    assert torch.isfinite(duplicate.vector).all()
    assert tda.distance(duplicate, duplicate).item() == 0.0


def test_hierarchy_insertion_balances_and_reduces_candidates() -> None:
    hierarchy = TaskHierarchy(branching=3, leaf_capacity=4)
    for task_id in range(20):
        embedding = torch.zeros(5)
        embedding[task_id % 5] = float(task_id)
        hierarchy.insert(task_id, embedding)
    candidates, evaluated = hierarchy.retrieve(torch.zeros(5), top_k=3, beam_width=2)
    assert len(candidates) <= 3
    assert evaluated < 20
    assert hierarchy.depth >= 2
    assert hierarchy.node_count > hierarchy.task_count / 4


def test_repository_caches_descriptors_and_round_trips(tmp_path) -> None:
    repository = TaskRepository(latent_dim=4, coarse_dim=2, ot_projections=3, tda_bins=4)
    repository.add_task(0, torch.randn(6, 4))
    repository.add_task(1, torch.randn(6, 4) + 2.0)
    destination = tmp_path / "repository.pt"
    repository.save(destination)
    restored = TaskRepository.load(destination)
    assert restored.task_ids == (0, 1)
    assert torch.allclose(restored.get(1).prototype, repository.get(1).prototype)
    assert restored.hierarchy.depth == repository.hierarchy.depth


def test_riemannian_metrics_are_positive_definite_and_conditioned() -> None:
    metric = LowRankRiemannianMetric(latent_dim=6, rank=2, min_eigenvalue=0.2, max_update=0.4)
    metric.add_task(4)
    matrix = metric.matrix(4)
    eigenvalues = torch.linalg.eigvalsh(matrix)
    assert torch.all(eigenvalues > 0)
    assert torch.isfinite(metric.condition_number(4))
    point = torch.randn(3, 6)
    prototype = torch.zeros(6)
    assert torch.all(metric.distance_squared(point, prototype, 4) >= 0)
    assert torch.allclose(metric.log_map(point, prototype), point)


def test_sparse_sheaf_adds_local_edges_and_energy_is_finite() -> None:
    sheaf = SparseTaskSheaf(latent_dim=5, stalk_dim=2, max_edges_per_task=2)
    first = {0: torch.zeros(5), 1: torch.ones(5), 2: torch.full((5,), 2.0)}
    assert sheaf.add_task(0, first[0], {}) == []
    sheaf.add_task(1, first[1], {0: first[0]})
    sheaf.add_task(2, first[2], {0: first[0], 1: first[1]})
    assert sheaf.edge_count == 3
    assert torch.isfinite(sheaf.energy(first))


def test_shared_low_rank_adapter_has_rank_sized_task_overhead() -> None:
    adapter = SharedLowRankAdapter(8, 8, rank=2)
    shared = adapter.shared_parameter_count
    adapter.add_task(0)
    adapter.add_task(1)
    output = adapter(torch.randn(4, 8), task_id=1)
    assert output.shape == (4, 8)
    assert adapter.task_parameter_count == 4
    assert adapter.shared_parameter_count == shared
    adapter.freeze_task(0)
    assert not adapter.task_coefficients["0"].requires_grad


def test_composed_router_uses_candidates_and_single_query_does_not_claim_ot() -> None:
    system = BONSAISystem(
        input_dim=6,
        num_classes=4,
        hidden_dim=12,
        latent_dim=6,
        repository_kwargs={"coarse_dim": 3, "ot_projections": 3, "tda_bins": 4},
    )
    for task_id in range(8):
        cloud = torch.randn(8, 6) + task_id * 3.0
        system.add_task(task_id, cloud)
    query = torch.randn(2, 6) + 12.0
    result = system.router.route(query)
    assert result.selected_task_ids.shape == (2,)
    assert len(result.candidate_task_ids) < 8
    assert result.scores.shape[1] == len(result.candidate_task_ids)
    assert torch.isfinite(result.scores).all()


def test_continual_trainer_adds_tasks_and_freezes_old_coefficients() -> None:
    torch.manual_seed(5)
    system = BONSAISystem(
        input_dim=4,
        num_classes=4,
        hidden_dim=12,
        latent_dim=4,
        repository_kwargs={"coarse_dim": 2, "ot_projections": 3, "tda_bins": 4},
    )
    trainer = BONSAITrainer(system, learning_rate=1e-3)
    first_inputs = torch.randn(10, 4)
    second_inputs = torch.randn(10, 4) + 2.0
    trainer.fit_task(0, first_inputs, torch.zeros(10, dtype=torch.long), epochs=1, batch_size=5)
    trainer.fit_task(1, second_inputs, torch.ones(10, dtype=torch.long), epochs=1, batch_size=5)
    assert system.task_ids == (0, 1)
    assert not system.model.adapter.task_coefficients["0"].requires_grad
    assert system.model.adapter.task_coefficients["1"].requires_grad is False
    assert torch.isfinite(system.router.route(second_inputs[:2]).scores).all()


def test_continual_trainer_supports_topology_gated_subspace_consolidation() -> None:
    system = BONSAISystem(
        input_dim=4,
        num_classes=2,
        hidden_dim=10,
        latent_dim=4,
        repository_kwargs={"coarse_dim": 2, "ot_projections": 3, "tda_bins": 4},
    )
    subspace = TopologyGatedRiemannianSubspaceConsolidator(rank=2)
    trainer = BONSAITrainer(
        system,
        learning_rate=1e-3,
        protection=InterferenceProtector(0.0),
        subspace_protection=subspace,
    )
    trainer.fit_task(0, torch.randn(8, 4), torch.zeros(8, dtype=torch.long), epochs=1, batch_size=4)
    trainer.fit_task(1, torch.randn(8, 4) + 1.0, torch.ones(8, dtype=torch.long), epochs=1, batch_size=4)
    assert set(subspace.anchors) == {0, 1}
    assert subspace.diagnostics(system.repository, 1)[0] >= subspace.relation_floor


def test_interference_protection_and_ratio_edge_cases() -> None:
    module = torch.nn.Linear(3, 2)
    protector = InterferenceProtector()
    protector.consolidate(module)
    with torch.no_grad():
        module.weight.add_(0.1)
    assert protector.penalty(module).item() > 0.0
    assert separation_ratio(1.0, 0.0) == float("inf")


def test_disabled_interference_protection_does_not_retain_memory() -> None:
    module = torch.nn.Linear(3, 2)
    protector = InterferenceProtector(0.0)

    protector.consolidate(module)

    assert protector.stored_elements == 0


def test_rgsc_stores_gradient_subspace_and_projects_related_updates() -> None:
    torch.manual_seed(11)
    module = torch.nn.Linear(4, 2)
    repository = TaskRepository(latent_dim=4, coarse_dim=2, ot_projections=3, tda_bins=4)
    repository.add_task(0, torch.randn(8, 4))
    repository.add_task(1, torch.randn(8, 4) + 1.0)
    consolidator = TopologyGatedRiemannianSubspaceConsolidator(
        rank=2, penalty_strength=1.0, projection_strength=0.5
    )
    for direction in torch.eye(4)[:3]:
        module.zero_grad(set_to_none=True)
        module(direction.unsqueeze(0)).square().mean().backward()
        consolidator.record_gradient(module)
    anchor = consolidator.consolidate(0, module)
    assert anchor is not None
    assert anchor.basis.shape[1] <= 2
    with torch.no_grad():
        module.weight.add_(0.1)
    assert torch.isfinite(consolidator.penalty(module, repository, 1))
    module.zero_grad(set_to_none=True)
    module(torch.ones(1, 4)).sum().backward()
    before = torch.cat([parameter.grad.flatten() for parameter in module.parameters() if parameter.grad is not None]).norm()
    consolidator.project_gradients(module, repository, 1)
    after = torch.cat([parameter.grad.flatten() for parameter in module.parameters() if parameter.grad is not None]).norm()
    assert torch.isfinite(after)
    assert after <= before + 1e-6


def test_atgtr_solves_bounded_conflict_projection() -> None:
    torch.manual_seed(19)
    module = torch.nn.Linear(4, 1)
    repository = TaskRepository(latent_dim=4, coarse_dim=2, ot_projections=3, tda_bins=4)
    repository.add_task(0, torch.randn(8, 4))
    repository.add_task(1, torch.randn(8, 4) + 1.0)
    consolidator = AdaptiveTaskGraphTrustRegion(
        rank=2, trust_fraction=0.05, damping=1e-3, max_constraints=8
    )
    for direction in torch.eye(4)[:3]:
        module.zero_grad(set_to_none=True)
        module(direction.unsqueeze(0)).square().mean().backward()
        consolidator.record_gradient(module)
    anchor = consolidator.consolidate(0, module)
    assert anchor is not None
    gradient = anchor.mean_gradient
    offset = 0
    for parameter in module.parameters():
        size = parameter.numel()
        parameter.grad = gradient[offset : offset + size].view_as(parameter).clone()
        offset += size
    consolidator.project_gradients(module, repository, 1)
    assert consolidator.last_projection["constraint_count"] > 0.0
    assert consolidator.last_projection["violation_fraction"] > 0.0
    assert consolidator.last_projection["gradient_correction_norm"] > 0.0
    assert all(torch.isfinite(parameter.grad).all() for parameter in module.parameters())


def test_continual_trainer_supports_atgtr() -> None:
    torch.manual_seed(23)
    system = BONSAISystem(
        input_dim=4,
        num_classes=4,
        hidden_dim=10,
        latent_dim=4,
        repository_kwargs={"coarse_dim": 2, "ot_projections": 3, "tda_bins": 4},
    )
    subspace = AdaptiveTaskGraphTrustRegion(rank=2, damping=1e-3)
    trainer = BONSAITrainer(
        system,
        learning_rate=1e-3,
        protection=InterferenceProtector(0.0),
        subspace_protection=subspace,
    )
    trainer.fit_task(0, torch.randn(8, 4), torch.zeros(8, dtype=torch.long), epochs=1, batch_size=4)
    trainer.fit_task(1, torch.randn(8, 4) + 1.0, torch.ones(8, dtype=torch.long), epochs=1, batch_size=4)
    assert set(subspace.anchors) == {0, 1}
    assert subspace.last_projection["constraint_count"] > 0.0


def test_scaling_benchmark_reports_measured_hierarchy_and_geometry() -> None:
    records = run_scaling_benchmark(task_counts=(4, 8), latent_dim=6, samples_per_task=8, queries_per_task=2)
    assert [record["task_count"] for record in records] == [4, 8]
    assert all(0.0 <= record["routing_accuracy"] <= 1.0 for record in records)
    assert all(record["candidate_count"] <= record["task_count"] for record in records)
    assert records[-1]["candidate_count"] < records[-1]["task_count"]
    assert all(record["hierarchy_depth"] >= 1 for record in records)
    assert all(record["condition_number_max"] > 0.0 for record in records)


def test_ablation_harness_returns_real_component_variants() -> None:
    records = run_router_ablation(task_count=4, latent_dim=6)
    assert {record["variant"] for record in records} == {
        "flat_baseline",
        "prototype_router",
        "hierarchical_plus_ot",
        "hierarchical_plus_tda",
        "hierarchical_plus_sheaf",
        "full_bonsai",
    }
    assert all(0.0 <= record["routing_accuracy"] <= 1.0 for record in records)


def test_new_modular_image_comparison_smoke() -> None:
    records = run_modular_image_comparison(
        seeds=(7,),
        num_tasks=2,
        classes_per_task=2,
        train_samples_per_class=2,
        test_samples_per_class=2,
        image_size=16,
        noise=0.1,
        epochs=1,
        batch_size=4,
    )
    assert {record["method"] for record in records} == {
        "new_bonsai_diagonal",
        "new_bonsai_tgrsc",
        "new_bonsai_atgtr",
        "new_bonsai_atgfr",
    }
    atgfr = next(record for record in records if record["method"] == "new_bonsai_atgfr")
    assert atgfr["replay_memory_elements"] > 0
    assert all(torch.isfinite(torch.tensor(record["task_aware_average_accuracy"])) for record in records)


def test_task_similarity_ablation_modes_are_explicit() -> None:
    system = BONSAISystem(
        input_dim=4,
        num_classes=2,
        hidden_dim=8,
        latent_dim=4,
        repository_kwargs={
            "coarse_dim": 3,
            "ot_projections": 2,
            "ot_samples": 4,
            "tda_bins": 3,
            "tda_samples": 4,
        },
    )
    system.register_task(0, torch.randn(6, 4))
    system.register_task(1, torch.randn(6, 4) + 0.5)
    values = {
        mode: system.repository.task_similarity(0, 1, mode=mode)
        for mode in ("full", "no_ot", "no_tda", "euclidean", "uniform")
    }
    assert all(0.0 < value <= 1.0 for value in values.values())
    assert values["uniform"] == 1.0
