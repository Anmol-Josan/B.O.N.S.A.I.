"""Compact variational information bottleneck representation.

For a diagonal Gaussian posterior and standard-normal prior, the KL term is
the practical variational upper bound used for ``I(X; Z)``. A supervised
task/classification loss supplies the tractable negative ``I(Z; T)`` surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class VIBOutput:
    """Output of a stochastic bottleneck."""

    z: Tensor
    mu: Tensor
    logvar: Tensor

    @property
    def kl(self) -> Tensor:
        return diagonal_gaussian_kl(self.mu, self.logvar)


def diagonal_gaussian_kl(mu: Tensor, logvar: Tensor) -> Tensor:
    """Return a finite minibatch mean KL for diagonal Gaussians."""

    if mu.shape != logvar.shape or mu.ndim < 2:
        raise ValueError("mu and logvar must have the same shape and a batch dimension")
    elementwise = -0.5 * (1.0 + logvar - mu.square() - logvar.exp())
    return elementwise.flatten(start_dim=1).sum(dim=1).mean().clamp_min(0.0)


class VIBEncoder(nn.Module):
    """A small MLP VIB encoder for vector or flattened image observations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        beta: float = 1e-3,
        logvar_min: float = -10.0,
        logvar_max: float = 4.0,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, latent_dim) < 1:
            raise ValueError("encoder dimensions must be positive")
        if beta < 0.0:
            raise ValueError("beta must be nonnegative")
        if logvar_min >= logvar_max:
            raise ValueError("logvar_min must be smaller than logvar_max")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.feature_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

    def _flatten_input(self, inputs: Tensor) -> Tensor:
        if inputs.ndim < 2:
            raise ValueError("inputs must have a batch dimension")
        flattened = inputs.flatten(start_dim=1)
        if flattened.shape[1] != self.input_dim:
            raise ValueError(
                f"expected flattened input width {self.input_dim}, got {flattened.shape[1]}"
            )
        return flattened

    def forward(self, inputs: Tensor, sample: bool | None = None) -> VIBOutput:
        features = self.feature_network(self._flatten_input(inputs))
        mu = self.mu_layer(features)
        logvar = self.logvar_layer(features).clamp(self.logvar_min, self.logvar_max)
        should_sample = self.training if sample is None else sample
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if should_sample else mu
        return VIBOutput(z=z, mu=mu, logvar=logvar)

    def information_loss(
        self, output: VIBOutput, predictive_loss: Tensor | None = None
    ) -> Tensor:
        """Return ``beta * KL`` plus an optional predictive surrogate."""

        loss = self.beta * output.kl
        if predictive_loss is not None:
            loss = loss + predictive_loss
        return loss

    def deterministic(self, inputs: Tensor) -> Tensor:
        """Encode without changing module state or sampling noise."""

        features = self.feature_network(self._flatten_input(inputs))
        return self.mu_layer(features)
