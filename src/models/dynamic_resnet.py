"""Compact dynamic backbone used by the BONSAI research pipeline.

The stem intentionally stays stable while growth is localized to residual
junctions after global pooling. This makes capacity changes explicit and easy
to account for in experiments.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ResidualJunction(nn.Module):
    """A bottleneck residual adapter over pooled feature vectors."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.down = nn.Linear(feature_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.up = nn.Linear(hidden_dim, feature_dim)

    def forward(self, features: Tensor) -> Tensor:
        return features + self.up(self.activation(self.down(features)))


class DynamicBackbone(nn.Module):
    """Small convolutional classifier with validation-driven junction growth."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 16,
        junction_growth_ratio: float = 0.04,
        plateau_patience: int = 5,
        plateau_min_improvement: float = 0.01,
    ) -> None:
        super().__init__()
        if not 0.0 < junction_growth_ratio:
            raise ValueError("junction_growth_ratio must be positive")
        if plateau_patience < 1:
            raise ValueError("plateau_patience must be at least 1")
        if plateau_min_improvement < 0.0:
            raise ValueError("plateau_min_improvement must be nonnegative")
        self.junction_growth_ratio = junction_growth_ratio
        self.plateau_patience = plateau_patience
        self.plateau_min_improvement = plateau_min_improvement
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feature_dim = base_channels * 2
        self.junctions = nn.ModuleList()
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.initial_parameter_count = self.total_parameters
        self.best_validation_loss = math.inf
        self.plateau_epochs = 0
        self.expansion_events = 0

    @property
    def total_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def parameter_overhead(self) -> float:
        return (self.total_parameters - self.initial_parameter_count) / self.initial_parameter_count

    def _select_junction_hidden_dim(self) -> int:
        target = self.initial_parameter_count * self.junction_growth_ratio
        # A bottleneck with width h has 2*d*h + h + d parameters.
        approximate = max(1, round((target - self.feature_dim) / (2 * self.feature_dim + 1)))
        candidates = range(max(1, approximate - 3), approximate + 4)
        return min(
            candidates,
            key=lambda hidden: abs(
                (2 * self.feature_dim * hidden + hidden + self.feature_dim) - target
            ),
        )

    def expand(self) -> ResidualJunction:
        """Insert one local residual junction and return it."""

        hidden_dim = self._select_junction_hidden_dim()
        junction = ResidualJunction(self.feature_dim, hidden_dim)
        self.junctions.append(junction)
        self.expansion_events += 1
        return junction

    def record_validation_loss(self, validation_loss: float) -> bool:
        """Track plateau progress and expand after ``plateau_patience`` misses."""

        if not math.isfinite(validation_loss) or validation_loss < 0.0:
            raise ValueError("validation_loss must be a finite nonnegative number")
        if math.isinf(self.best_validation_loss):
            self.best_validation_loss = validation_loss
            return False
        required_loss = self.best_validation_loss * (1.0 - self.plateau_min_improvement)
        if validation_loss < required_loss:
            self.best_validation_loss = validation_loss
            self.plateau_epochs = 0
            return False
        self.plateau_epochs += 1
        if self.plateau_epochs < self.plateau_patience:
            return False
        self.expand()
        self.best_validation_loss = validation_loss
        self.plateau_epochs = 0
        return True

    def forward_features(self, inputs: Tensor) -> Tensor:
        features = self.stem(inputs).flatten(1)
        for junction in self.junctions:
            features = junction(features)
        return features

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.forward_features(inputs))

