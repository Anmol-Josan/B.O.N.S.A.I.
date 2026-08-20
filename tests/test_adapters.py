from __future__ import annotations

import torch

from src.models.adapters import BottleneckAdapter, ConvBottleneckAdapter


def test_bottleneck_adapter_starts_as_identity_and_has_low_rank_capacity() -> None:
    adapter = BottleneckAdapter(hidden_dim=12, bottleneck_dim=3)
    features = torch.randn(5, 12)

    assert torch.allclose(adapter(features), features)
    assert adapter.parameter_count == 12 * 3 * 2


def test_convolutional_adapter_starts_as_identity() -> None:
    adapter = ConvBottleneckAdapter(channels=8, bottleneck_dim=2)
    features = torch.randn(3, 8, 6, 6)

    assert torch.allclose(adapter(features), features)
    assert adapter.parameter_count == 8 * 2 * 2


def test_nonzero_residual_initialization_activates_both_low_rank_factors() -> None:
    torch.manual_seed(4)
    adapter = BottleneckAdapter(hidden_dim=12, bottleneck_dim=3, up_init_std=0.02)
    features = torch.randn(5, 12)
    loss = adapter(features).square().mean()
    loss.backward()

    assert not torch.allclose(adapter(features.detach()), features.detach())
    assert adapter.down.weight.grad is not None
    assert adapter.down.weight.grad.abs().sum() > 0
