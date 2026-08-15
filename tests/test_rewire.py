from __future__ import annotations

import torch
from torch import nn

from src.algorithms.rewire import RewireEngine


def test_orthogonal_rewire_preserves_frozen_entries_and_changes_unfrozen_entries() -> None:
    torch.manual_seed(0)
    model = nn.Linear(4, 4, bias=False)
    before = model.weight.detach().clone()
    frozen = torch.zeros_like(model.weight, dtype=torch.bool)
    frozen[0, :2] = True
    frozen[2, 2:] = True

    RewireEngine(strategy="orthogonal", seed=11).rewire(model, {"weight": frozen})

    after = model.weight.detach()
    assert torch.equal(after[frozen], before[frozen])
    assert torch.any(after[~frozen] != before[~frozen])


def test_fully_unfrozen_square_matrix_is_orthogonal_after_rewire() -> None:
    model = nn.Linear(5, 5, bias=False)
    RewireEngine(strategy="orthogonal", seed=12).rewire(
        model, {"weight": torch.zeros_like(model.weight, dtype=torch.bool)}
    )

    gram = model.weight @ model.weight.T
    assert torch.allclose(gram, torch.eye(5), atol=1e-5, rtol=1e-5)


def test_gaussian_strategy_is_available_for_rewiring_ablation() -> None:
    model = nn.Linear(3, 3, bias=False)
    before = model.weight.detach().clone()
    RewireEngine(strategy="gaussian", seed=13).rewire(
        model, {"weight": torch.zeros_like(model.weight, dtype=torch.bool)}
    )

    assert not torch.equal(model.weight, before)


def test_residual_rewire_preserves_bias_and_moves_available_weights_partway() -> None:
    torch.manual_seed(4)
    model = nn.Linear(4, 4, bias=True)
    before_weight = model.weight.detach().clone()
    before_bias = model.bias.detach().clone()
    RewireEngine(strategy="orthogonal", seed=14, strength=0.2).rewire(
        model, {"weight": torch.zeros_like(model.weight, dtype=torch.bool)}
    )

    assert torch.equal(model.bias.detach(), before_bias)
    assert torch.any(model.weight.detach() != before_weight)
    assert torch.linalg.vector_norm(model.weight - before_weight) < 2.0
