"""Variational information-bottleneck layers.

The layers keep the tensor-only PyTorch ``forward`` contract and expose the
regularization term through ``kl_loss``. ``forward_with_kl`` is provided for
call sites that want to collect both values explicitly.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


def _standard_normal_kl(mu: Tensor, logvar: Tensor) -> Tensor:
    """Return the minibatch-mean KL(q || N(0, I)) for diagonal Gaussians."""

    elementwise = -0.5 * (1.0 + logvar - mu.square() - logvar.exp())
    # The analytic expression is nonnegative, but a clamp protects the scalar
    # from tiny negative values caused by floating-point cancellation.
    return elementwise.flatten(start_dim=1).sum(dim=1).mean().clamp_min(0.0)


class _VIBBase(nn.Module):
    """Shared state and API for VIB layers."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_kl_loss", torch.zeros(()), persistent=False)

    @property
    def kl_loss(self) -> Tensor:
        """Most recent minibatch KL term, differentiable until backward."""

        return self._kl_loss

    def _sample(self, mu: Tensor, logvar: Tensor) -> Tensor:
        self._kl_loss = _standard_normal_kl(mu, logvar)
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward_with_kl(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        output = self.forward(x)
        return output, self.kl_loss


class VIBLinear(_VIBBase):
    """A stochastic linear bottleneck with a standard-normal prior."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.mu_layer = nn.Linear(in_features, out_features, bias=bias)
        self.logvar_layer = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x).clamp(min=-12.0, max=8.0)
        return self._sample(mu, logvar)


class VIBConv2d(_VIBBase):
    """A stochastic convolutional bottleneck with a standard-normal prior."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        conv_kwargs = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
        }
        self.mu_layer = nn.Conv2d(in_channels, out_channels, **conv_kwargs)
        self.logvar_layer = nn.Conv2d(in_channels, out_channels, **conv_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x).clamp(min=-12.0, max=8.0)
        return self._sample(mu, logvar)

