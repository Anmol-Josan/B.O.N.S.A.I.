from __future__ import annotations

import torch

from src.models.vib_layers import VIBConv2d, VIBLinear


def test_vib_linear_returns_expected_shape_and_nonnegative_kl() -> None:
    torch.manual_seed(0)
    layer = VIBLinear(5, 3)
    output = layer(torch.randn(4, 5))

    assert output.shape == (4, 3)
    assert layer.kl_loss.ndim == 0
    assert layer.kl_loss.item() >= 0.0


def test_vib_conv_returns_expected_shape_and_nonnegative_kl() -> None:
    torch.manual_seed(0)
    layer = VIBConv2d(3, 4, kernel_size=3, padding=1)
    output = layer(torch.randn(2, 3, 8, 8))

    assert output.shape == (2, 4, 8, 8)
    assert layer.kl_loss.ndim == 0
    assert layer.kl_loss.item() >= 0.0


def test_reparameterized_vib_backpropagates_to_mean_and_variance() -> None:
    torch.manual_seed(1)
    layer = VIBLinear(4, 2)
    output = layer(torch.randn(6, 4))
    loss = output.square().mean() + 0.2 * layer.kl_loss
    loss.backward()

    assert layer.mu_layer.weight.grad is not None
    assert layer.logvar_layer.weight.grad is not None
    assert torch.isfinite(layer.mu_layer.weight.grad).all()
    assert torch.isfinite(layer.logvar_layer.weight.grad).all()


def test_forward_with_kl_is_consistent_with_tensor_forward_contract() -> None:
    torch.manual_seed(2)
    layer = VIBLinear(3, 2)
    layer.eval()
    output, kl = layer.forward_with_kl(torch.ones(2, 3))

    assert output.shape == (2, 2)
    assert kl.ndim == 0
    assert torch.allclose(output, layer.mu_layer(torch.ones(2, 3)))

