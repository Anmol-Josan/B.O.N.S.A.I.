from __future__ import annotations

import torch
from torch import nn

from src.algorithms.mask_manager import MaskManager


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 3)
        self.second = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(torch.tanh(self.first(x)))


def test_saliency_is_absolute_loss_gradient_and_masks_are_binary() -> None:
    torch.manual_seed(0)
    model = TinyClassifier()
    inputs = torch.randn(8, 4)
    labels = torch.randint(0, 2, (8,))
    loss = nn.functional.cross_entropy(model(inputs), labels)

    manager = MaskManager(saliency_quantile=0.8)
    saliency = manager.compute_saliency(model, loss)

    assert set(saliency) == {"first.weight", "first.bias", "second.weight", "second.bias"}
    assert all(torch.all(value >= 0) for value in saliency.values())
    assert torch.allclose(saliency["first.weight"], model.first.weight.grad.abs())

    masks = manager.build_critical_masks(saliency)
    assert all(mask.dtype == torch.bool for mask in masks.values())
    assert sum(mask.numel() for mask in masks.values()) == sum(
        mask.numel() for mask in masks.values()
    )
    assert any(mask.any() for mask in masks.values())


def test_critical_mask_zeroes_only_frozen_gradients_on_future_backward() -> None:
    torch.manual_seed(1)
    model = TinyClassifier()
    manager = MaskManager(saliency_quantile=0.5)
    masks = {
        "first.weight": torch.tensor(
            [[True, False, False, False], [False, True, False, False], [False, False, False, True]]
        ),
        "first.bias": torch.tensor([True, False, False]),
        "second.weight": torch.zeros_like(model.second.weight, dtype=torch.bool),
        "second.bias": torch.tensor([False, True]),
    }
    manager.freeze_critical(model, masks)

    model.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(torch.randn(10, 4)), torch.randint(0, 2, (10,)))
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None
        assert torch.all(parameter.grad[masks[name]] == 0)
        if (~masks[name]).any():
            assert torch.any(parameter.grad[~masks[name]] != 0)


def test_masks_can_be_accumulated_without_unfreezing_previous_tasks() -> None:
    manager = MaskManager(saliency_quantile=0.5)
    first = {"w": torch.tensor([True, False, False])}
    second = {"w": torch.tensor([False, True, False])}

    manager.update_critical_masks(first)
    manager.update_critical_masks(second)

    assert torch.equal(manager.critical_masks["w"], torch.tensor([True, True, False]))

