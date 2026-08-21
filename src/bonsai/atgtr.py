"""Adaptive Task-Graph Trust-Region Consolidation (ATGTR).

ATGTR turns gradient preservation into a small quadratic projection problem.
For an old-task descent direction ``a_u`` and proposed update ``d=-g``, it
approximately enforces

    a_u^T d >= -epsilon_u.

The closed-form active-set approximation is ``d' = d + A^T lambda`` with
``(A A^T + mu I) lambda = [-epsilon - A d]_+``. Unlike null-space GPM/OGD,
ATGTR only corrects violated constraints, so compatible transfer remains
plastic. ``A`` contains low-rank signed directions from the old task gradient
subspaces and is relation-weighted by BONSAI's OT/TDA/Riemannian repository;
the default compact mode retains only the mean-gradient row for a bounded
recent-task reservoir.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.bonsai.repository import TaskRepository
from src.bonsai.rgsc import TopologyGatedRiemannianSubspaceConsolidator


class AdaptiveTaskGraphTrustRegion(TopologyGatedRiemannianSubspaceConsolidator):
    """Low-rank inequality-constrained gradient projection for BONSAI."""

    def __init__(
        self,
        rank: int = 4,
        trust_fraction: float = 0.5,
        basis_weight: float = 0.25,
        damping: float = 1e-4,
        max_constraints: int = 32,
        max_anchor_tasks: int = 4,
        compact_memory: bool = True,
        **kwargs: object,
    ) -> None:
        if trust_fraction < 0.0 or basis_weight < 0.0:
            raise ValueError("trust_fraction and basis_weight must be nonnegative")
        if damping <= 0.0 or max_constraints < 1 or max_anchor_tasks < 1:
            raise ValueError("damping, max_constraints, and max_anchor_tasks must be positive")
        super().__init__(rank=rank, penalty_strength=0.0, projection_strength=0.0, **kwargs)
        self.trust_fraction = trust_fraction
        self.basis_weight = basis_weight
        self.damping = damping
        self.max_constraints = max_constraints
        self.max_anchor_tasks = max_anchor_tasks
        self.compact_memory = compact_memory
        self.last_projection: dict[str, float] = {}

    @torch.no_grad()
    def consolidate(self, task_id: int, module: nn.Module):
        """Store a bounded-memory anchor for the completed task.

        ATGTR's default deployment mode does not need a displacement penalty,
        so it can discard the full parameter reference and optional residual
        basis after extracting the mean-gradient trust direction.  Keeping a
        fixed recent-task reservoir makes memory independent of the lifetime
        task count; ``compact_memory=False`` exposes the richer research form.
        """

        anchor = super().consolidate(task_id, module)
        if anchor is None:
            return None
        if self.compact_memory:
            anchor.reference = anchor.reference.new_empty(0)
            anchor.basis = anchor.basis[:, :0]
            anchor.spectrum = anchor.spectrum[:0]
        while len(self.anchors) > self.max_anchor_tasks:
            oldest_task_id = next(iter(self.anchors))
            del self.anchors[oldest_task_id]
        return anchor

    def penalty(
        self,
        module: nn.Module,
        repository: TaskRepository,
        new_task_id: int,
    ) -> Tensor:
        """ATGTR protects through the projected step, not displacement loss."""

        return next(module.parameters()).new_zeros(())

    def _constraint_rows(
        self,
        repository: TaskRepository,
        new_task_id: int,
        anchor_task_id: int,
        anchor,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        relation = self._relation(repository, new_task_id, anchor_task_id)
        mean = anchor.mean_gradient.to(device)
        mean = mean / mean.norm().clamp_min(1e-8)
        basis = anchor.basis.to(device)
        # The mean gradient is the actual first-order old-task direction.  The
        # SVD basis also contains within-task variation, but individual basis
        # axes are not themselves loss gradients.  Make those residual axes
        # orthogonal to the mean and protect them with a softer budget.
        residual_basis = basis - mean.unsqueeze(1) @ (mean.unsqueeze(0) @ basis)
        residual_norms = residual_basis.norm(dim=0)
        valid = residual_norms > 1e-6
        residual_basis = residual_basis[:, valid]
        if residual_basis.shape[1] > 0:
            residual_basis = residual_basis / residual_norms[valid].unsqueeze(0)
            directions = torch.cat((mean.unsqueeze(0), residual_basis.T), dim=0)
            residual_spectrum = anchor.spectrum.to(device)[valid].clamp_min(1e-6).sqrt()
        else:
            directions = mean.unsqueeze(0)
            residual_spectrum = torch.empty(0, device=device)
        weights = torch.cat(
            (
                torch.ones(1, device=device),
                self.basis_weight * residual_spectrum,
            )
        )
        directions = directions * weights.unsqueeze(1)
        scale = torch.tensor(relation, device=device).sqrt()
        rows = directions * scale
        return rows, torch.full(
            (rows.shape[0],), self.trust_fraction, device=device
        ) * weights

    @torch.no_grad()
    def project_gradients(
        self,
        module: nn.Module,
        repository: TaskRepository,
        new_task_id: int,
    ) -> None:
        if not self.anchors:
            self.last_projection = {"constraint_count": 0.0, "violation_fraction": 0.0}
            return
        ordered = sorted(
            self.anchors.items(),
            key=lambda item: self._relation(repository, new_task_id, item[0]),
            reverse=True,
        )
        rows: list[Tensor] = []
        epsilons: list[Tensor] = []
        names = next(iter(self.anchors.values())).names
        gradient, layout = self._gradient_by_name(module, names)
        device = gradient.device
        for anchor_task_id, anchor in ordered:
            if anchor_task_id not in repository.records or new_task_id not in repository.records:
                continue
            anchor_rows, anchor_epsilons = self._constraint_rows(
                repository, new_task_id, anchor_task_id, anchor, device
            )
            rows.append(anchor_rows)
            epsilons.append(anchor_epsilons)
            if sum(item.shape[0] for item in rows) >= self.max_constraints:
                break
        if not rows:
            self.last_projection = {"constraint_count": 0.0, "violation_fraction": 0.0}
            return
        matrix = torch.cat(rows, dim=0)[: self.max_constraints]
        epsilon = torch.cat(epsilons, dim=0)[: self.max_constraints] * gradient.norm().clamp_min(1e-8)
        gram = matrix @ matrix.T
        violations = (matrix @ gradient - epsilon).clamp_min(0.0)
        active = violations > 0.0
        if active.any():
            active_matrix = gram[active][:, active]
            rhs = violations[active]
            regularized = active_matrix + self.damping * torch.eye(
                active_matrix.shape[0], device=device, dtype=active_matrix.dtype
            )
            multipliers = torch.linalg.solve(regularized, rhs).clamp_min(0.0)
            correction = matrix[active].T @ multipliers
            updated_gradient = gradient - correction
        else:
            updated_gradient = gradient
        for parameter, start, end in layout:
            if parameter.grad is not None:
                parameter.grad.copy_(updated_gradient[start:end].view_as(parameter).to(parameter.grad.dtype))
        self.last_projection = {
            "constraint_count": float(matrix.shape[0]),
            "violation_fraction": float(active.float().mean()),
            "gradient_correction_norm": float((gradient - updated_gradient).norm()),
        }
