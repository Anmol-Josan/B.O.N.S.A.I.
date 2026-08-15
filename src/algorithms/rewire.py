"""Reinitialization of non-critical parameters between continual tasks."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Collection

import torch
from torch import Tensor, nn


def _random_tensor_like(parameter: Tensor, generator: torch.Generator | None) -> Tensor:
    return torch.randn(
        parameter.shape,
        generator=generator,
        device=parameter.device,
        dtype=parameter.dtype,
    )


def _orthogonal_tensor_like(parameter: Tensor, generator: torch.Generator | None) -> Tensor:
    """Generate a (possibly semi-)orthogonal tensor with the same shape."""

    if parameter.ndim < 2:
        return _random_tensor_like(parameter, generator)
    rows = parameter.shape[0]
    cols = parameter.numel() // rows
    tall_shape = (max(rows, cols), min(rows, cols))
    random_matrix = torch.randn(
        tall_shape,
        generator=generator,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    q, r = torch.linalg.qr(random_matrix, mode="reduced")
    signs = torch.sign(torch.diagonal(r, 0))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(0)
    matrix = q if rows >= cols else q.T
    return matrix.reshape(parameter.shape)


class RewireEngine:
    """Rewire only non-critical entries while preserving frozen values."""

    def __init__(
        self,
        strategy: str = "orthogonal",
        seed: int | None = None,
        strength: float = 1.0,
        rewire_bias: bool = False,
    ) -> None:
        if strategy not in {"orthogonal", "gaussian"}:
            raise ValueError("strategy must be 'orthogonal' or 'gaussian'")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        self.strategy = strategy
        self.seed = seed
        self.strength = strength
        self.rewire_bias = rewire_bias

    def rewire(
        self,
        model: nn.Module,
        frozen_masks: Mapping[str, Tensor],
        exclude_names: Collection[str] | None = None,
    ) -> list[str]:
        """Reinitialize available entries and return modified parameter names.

        A full-strength rewire retains the original behavior.  A smaller
        ``strength`` performs a residual move toward the orthogonal candidate,
        preserving useful representations while still changing the available
        subgraph.  One-dimensional tensors (typically biases and VIB log-scale
        terms) are preserved by default because reinitializing them can create
        large, non-geometric shifts in activation statistics.
        """

        changed: list[str] = []
        excluded = set(exclude_names or ())
        for index, (name, parameter) in enumerate(model.named_parameters()):
            if not parameter.requires_grad:
                continue
            if name in excluded or (parameter.ndim < 2 and not self.rewire_bias):
                continue
            frozen = frozen_masks.get(name)
            if frozen is None:
                frozen = torch.zeros_like(parameter, dtype=torch.bool)
            if frozen.shape != parameter.shape:
                raise ValueError(f"mask shape does not match parameter {name}")
            generator = None
            if self.seed is not None:
                generator = torch.Generator(device=parameter.device)
                generator.manual_seed(self.seed + index)
            if self.strategy == "orthogonal":
                candidate = _orthogonal_tensor_like(parameter, generator)
            else:
                candidate = _random_tensor_like(parameter, generator)
            with torch.no_grad():
                original = parameter.detach().clone()
                if self.strength < 1.0:
                    candidate = original + self.strength * (candidate - original)
                parameter.copy_(torch.where(frozen.to(parameter.device), original, candidate))
            if torch.any(~frozen):
                changed.append(name)
        return changed
