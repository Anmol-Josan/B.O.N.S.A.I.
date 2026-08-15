"""Critical-subgraph extraction and elementwise gradient freezing."""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import Tensor, nn


class MaskManager:
    """Maintain task masks and prevent updates to accumulated critical weights.

    A mask value of ``True`` means that the corresponding parameter element is
    critical/frozen. Masks are accumulated with logical OR so learning a later
    task cannot unfreeze an earlier task's subgraph.
    """

    def __init__(self, saliency_quantile: float = 0.8) -> None:
        if not 0.0 <= saliency_quantile <= 1.0:
            raise ValueError("saliency_quantile must be in [0, 1]")
        self.saliency_quantile = saliency_quantile
        self.critical_masks: dict[str, Tensor] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []

    @staticmethod
    def compute_saliency(model: nn.Module, loss: Tensor) -> dict[str, Tensor]:
        """Backpropagate ``loss`` and return |d loss / d parameter| tensors."""

        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        return {
            name: parameter.grad.detach().abs().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        }

    def build_critical_masks(
        self, saliency: Mapping[str, Tensor], quantile: float | None = None
    ) -> dict[str, Tensor]:
        """Threshold saliency globally and select the requested top quantile."""

        quantile = self.saliency_quantile if quantile is None else quantile
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be in [0, 1]")
        if not saliency:
            return {}
        flattened = torch.cat([value.detach().reshape(-1) for value in saliency.values()])
        keep_count = max(1, math.ceil((1.0 - quantile) * flattened.numel()))
        selected = torch.zeros(flattened.numel(), dtype=torch.bool, device=flattened.device)
        selected[torch.topk(flattened, k=keep_count, largest=True, sorted=False).indices] = True
        masks: dict[str, Tensor] = {}
        offset = 0
        for name, value in saliency.items():
            size = value.numel()
            masks[name] = selected[offset : offset + size].reshape(value.shape)
            offset += size
        return masks

    def update_critical_masks(self, masks: Mapping[str, Tensor]) -> None:
        """Accumulate masks from a new task without changing prior masks."""

        for name, mask in masks.items():
            boolean_mask = mask.detach().to(dtype=torch.bool)
            previous = self.critical_masks.get(name)
            if previous is not None and previous.shape != boolean_mask.shape:
                raise ValueError(f"mask shape changed for parameter {name}")
            self.critical_masks[name] = (
                boolean_mask if previous is None else previous.to(boolean_mask.device) | boolean_mask
            ).clone()

    def freeze_critical(self, model: nn.Module, masks: Mapping[str, Tensor] | None = None) -> None:
        """Install gradient hooks that zero critical entries on every backward."""

        if masks is not None:
            self.update_critical_masks(masks)
        self.remove_hooks()
        parameters = dict(model.named_parameters())
        for name, mask in self.critical_masks.items():
            parameter = parameters.get(name)
            if parameter is None:
                raise KeyError(f"no model parameter named {name}")
            if parameter.shape != mask.shape:
                raise ValueError(f"mask shape does not match parameter {name}")
            frozen_mask = mask.detach().clone()

            def mask_gradient(gradient: Tensor, frozen_mask: Tensor = frozen_mask) -> Tensor:
                return gradient.masked_fill(frozen_mask.to(device=gradient.device), 0.0)

            self._hook_handles.append(parameter.register_hook(mask_gradient))

    def apply_gradient_masks(self, model: nn.Module) -> None:
        """Apply masks to already-computed gradients (useful before an optimizer step)."""

        for name, parameter in model.named_parameters():
            if parameter.grad is not None and name in self.critical_masks:
                parameter.grad.mul_(~self.critical_masks[name].to(parameter.grad.device))

    def remove_hooks(self) -> None:
        """Remove all installed gradient hooks."""

        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    @property
    def frozen_parameter_count(self) -> int:
        return sum(mask.sum().item() for mask in self.critical_masks.values())
