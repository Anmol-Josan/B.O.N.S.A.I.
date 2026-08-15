from __future__ import annotations

import torch

from src.models.adapters import BottleneckAdapter


def test_bottleneck_adapter_starts_as_identity_and_has_low_rank_capacity() -> None:
    adapter = BottleneckAdapter(hidden_dim=12, bottleneck_dim=3)
    features = torch.randn(5, 12)

    assert torch.allclose(adapter(features), features)
    assert adapter.parameter_count == 12 * 3 * 2

