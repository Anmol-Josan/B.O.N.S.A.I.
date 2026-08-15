from __future__ import annotations

import torch

from src.algorithms.continual import ToyContinualLearner
from src.algorithms.mask_manager import MaskManager
from src.algorithms.rewire import RewireEngine
from src.data.synthetic import make_synthetic_tasks


def test_two_task_sequence_retains_task_one_and_respects_capacity_budget() -> None:
    torch.manual_seed(7)
    tasks = make_synthetic_tasks(
        num_tasks=2, classes_per_task=2, samples_per_class=64, input_dim=2, seed=7
    )
    learner = ToyContinualLearner(input_dim=2, hidden_dim=16, classes_per_task=2, num_tasks=2)
    initial_parameter_count = learner.total_parameters

    learner.train_task(0, tasks[0], epochs=5, batch_size=16, learning_rate=0.2)
    task_one_before = learner.accuracy(0, tasks[0])
    assert task_one_before >= 0.90

    manager = MaskManager(saliency_quantile=0.8)
    saliency = manager.compute_saliency(learner, learner.loss_on_task(0, tasks[0]))
    masks = manager.build_critical_masks(saliency)
    masks["heads.0.weight"] = torch.ones_like(learner.heads[0].weight, dtype=torch.bool)
    masks["heads.0.bias"] = torch.ones_like(learner.heads[0].bias, dtype=torch.bool)
    manager.freeze_critical(learner, masks)
    frozen_snapshot = {
        name: parameter.detach().clone()
        for name, parameter in learner.named_parameters()
        if name in manager.critical_masks
    }

    # The next task receives a newly allocated path; the old path is not rewired.
    RewireEngine(strategy="orthogonal", seed=8).rewire(
        learner.heads[1],
        {"weight": torch.zeros_like(learner.heads[1].weight, dtype=torch.bool), "bias": torch.zeros_like(learner.heads[1].bias, dtype=torch.bool)},
    )
    learner.train_task(1, tasks[1], epochs=5, batch_size=16, learning_rate=0.2)
    task_one_after = learner.accuracy(0, tasks[0])

    assert task_one_before - task_one_after < 0.02
    for name, parameter in learner.named_parameters():
        mask = manager.critical_masks.get(name)
        if mask is not None:
            assert torch.equal(parameter.detach()[mask], frozen_snapshot[name][mask])
    assert learner.total_parameters <= initial_parameter_count * 1.25

    predictions, selected_tasks, entropies = learner.predict_with_entropy(
        torch.cat([tasks[0].inputs[:4], tasks[1].inputs[:4]], dim=0)
    )
    assert predictions.shape == (8,)
    assert selected_tasks.shape == (8,)
    assert entropies.shape == (8, 2)
    assert torch.isfinite(entropies).all()


def test_toy_learner_can_explicitly_update_shared_encoder_on_later_tasks() -> None:
    tasks = make_synthetic_tasks(num_tasks=2, classes_per_task=2, samples_per_class=8, seed=9)
    learner = ToyContinualLearner(input_dim=2, hidden_dim=8, classes_per_task=2, num_tasks=2)
    learner.train_task(0, tasks[0], epochs=1, batch_size=8)
    before = learner.encoder.mu_layer.weight.detach().clone()
    learner.train_task(1, tasks[1], epochs=1, batch_size=8, update_encoder=True)

    assert not torch.equal(before, learner.encoder.mu_layer.weight.detach())
