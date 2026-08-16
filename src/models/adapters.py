"""Small task-local residual modules for parameter-efficient specialization."""

from __future__ import annotations

from torch import Tensor, nn


class BottleneckAdapter(nn.Module):
    """A zero-initialized low-rank residual adapter.

    The adapter starts as an exact identity map, so allocating a new task path
    does not invalidate the shared representation.  Only ``2 * hidden_dim *
    bottleneck_dim`` weights are task-specific, which is substantially smaller
    than copying a full backbone column.
    """

    def __init__(self, hidden_dim: int, bottleneck_dim: int, scale: float = 1.0) -> None:
        super().__init__()
        if min(hidden_dim, bottleneck_dim) < 1:
            raise ValueError("adapter dimensions must be positive")
        if scale < 0.0:
            raise ValueError("scale must be nonnegative")
        self.down = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        self.scale = scale
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, features: Tensor) -> Tensor:
        residual = self.up(self.down(features).tanh())
        return features + self.scale * residual


class IdentityAdapter(nn.Module):
    """Parameter-free adapter used when the rank is disabled."""

    def forward(self, features: Tensor) -> Tensor:
        return features


class ConvBottleneckAdapter(nn.Module):
    """A zero-initialized 1x1 convolutional residual adapter."""

    def __init__(self, channels: int, bottleneck_dim: int, scale: float = 1.0) -> None:
        super().__init__()
        if min(channels, bottleneck_dim) < 1:
            raise ValueError("adapter dimensions must be positive")
        if scale < 0.0:
            raise ValueError("scale must be nonnegative")
        self.down = nn.Conv2d(channels, bottleneck_dim, kernel_size=1, bias=False)
        self.up = nn.Conv2d(bottleneck_dim, channels, kernel_size=1, bias=False)
        self.scale = scale
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, features: Tensor) -> Tensor:
        residual = self.up(self.down(features).tanh())
        return features + self.scale * residual
