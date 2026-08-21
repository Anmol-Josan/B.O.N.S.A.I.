"""Topology-gated low-rank subspace consolidation.

This is the research extension built on top of BONSAI's repository.  Diagonal
EWC treats every parameter direction independently.  RGSC instead stores a
small basis for the shared-parameter directions actually used by each task and
protects those directions with a relation-dependent quadratic penalty and a
soft gradient projection.

For task ``u`` with gradient matrix ``G_u`` and compact right-singular basis
``Q_u``, the retention term is

    L_RGSC = sum_u w(t, u) || diag(s_u)^(1/2) Q_u^T (theta-theta_u) ||^2.

The gate ``w(t,u)`` is computed from cached sliced-Wasserstein and persistence
distances, with a nonzero floor so an unrelated task cannot disable retention.
This is a practical low-rank approximation, not a claim that the full Fisher
matrix is known.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.bonsai.geometry import LowRankRiemannianMetric
from src.bonsai.repository import TaskRepository


@dataclass
class SubspaceAnchor:
    task_id: int
    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    reference: Tensor
    basis: Tensor
    spectrum: Tensor
    mean_gradient: Tensor


class TopologyGatedRiemannianSubspaceConsolidator:
    """Incremental low-rank protection for shared BONSAI parameters."""

    def __init__(
        self,
        rank: int = 4,
        penalty_strength: float = 1.0,
        projection_strength: float = 0.35,
        relation_floor: float = 0.1,
        max_parameters: int = 100_000,
        metric: LowRankRiemannianMetric | None = None,
        geometry_temperature: float = 5.0,
        shared_prefixes: tuple[str, ...] | None = None,
    ) -> None:
        if min(rank, max_parameters) < 1:
            raise ValueError("rank and max_parameters must be positive")
        if penalty_strength < 0.0 or projection_strength < 0.0:
            raise ValueError("consolidation strengths must be nonnegative")
        if not 0.0 <= relation_floor <= 1.0:
            raise ValueError("relation_floor must be in [0, 1]")
        if geometry_temperature <= 0.0:
            raise ValueError("geometry_temperature must be positive")
        self.rank = rank
        self.penalty_strength = penalty_strength
        self.projection_strength = projection_strength
        self.relation_floor = relation_floor
        self.max_parameters = max_parameters
        self.metric = metric
        self.geometry_temperature = geometry_temperature
        self.shared_prefixes = shared_prefixes
        self.anchors: dict[int, SubspaceAnchor] = {}
        self._gradient_history: list[Tensor] = []
        self._parameter_signature: tuple[str, ...] | None = None

    @property
    def stored_elements(self) -> int:
        """Number of float elements retained by task anchors."""

        return sum(
            anchor.reference.numel()
            + anchor.basis.numel()
            + anchor.spectrum.numel()
            + anchor.mean_gradient.numel()
            for anchor in self.anchors.values()
        )

    @staticmethod
    def _is_shared_parameter(name: str) -> bool:
        """Select the shared representation, excluding task-specific outputs.

        BONSAI's compact global classifier contains rows for classes that have
        not arrived yet. Protecting its full gradient subspace would suppress
        legitimate plasticity for those new rows, so RGSC is intentionally
        applied to the VIB encoder chart only. The classifier remains trainable
        while the representation geometry is consolidated.
        """

        return name.startswith("encoder.")

    def _parameter_items(self, module: nn.Module) -> list[tuple[str, nn.Parameter]]:
        all_items = [
            (name, parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad and not name.startswith("adapter.task_coefficients.")
        ]
        items = [item for item in all_items if self._is_shared_parameter(item[0])]
        if self.shared_prefixes is not None:
            items = [
                item for item in items
                if item[0].startswith(self.shared_prefixes)
            ]
        # Keep the consolidator independently usable for a generic module
        # (e.g. a linear research probe) that has no ``encoder.`` namespace.
        if not items:
            items = all_items
        if not items:
            raise ValueError("module has no eligible shared parameters")
        total = sum(parameter.numel() for _, parameter in items)
        if total > self.max_parameters:
            # For large backbones, protect the compact representation and
            # classifier first. This keeps memory O(rank * selected_parameters)
            # and does not silently pretend to protect the whole network.
            preferred = [
                item for item in items if item[0].startswith(("encoder.", "classifier."))
            ]
            if preferred and sum(parameter.numel() for _, parameter in preferred) <= self.max_parameters:
                items = preferred
            else:
                raise ValueError(
                    "eligible shared parameters exceed max_parameters; pass a larger budget "
                    "or restrict the module before RGSC consolidation"
                )
        signature = tuple(name for name, _ in items)
        if self._parameter_signature is None:
            self._parameter_signature = signature
        elif signature != self._parameter_signature:
            raise ValueError("shared parameter set changed after RGSC consolidation")
        return items

    def _pack_parameters(self, module: nn.Module) -> tuple[Tensor, list[tuple[str, tuple[int, ...], int, int]]]:
        items = self._parameter_items(module)
        pieces: list[Tensor] = []
        layout: list[tuple[str, tuple[int, ...], int, int]] = []
        offset = 0
        for name, parameter in items:
            flattened = parameter.detach().float().flatten()
            end = offset + flattened.numel()
            pieces.append(flattened)
            layout.append((name, tuple(parameter.shape), offset, end))
            offset = end
        return torch.cat(pieces), layout

    def _pack_gradients(self, module: nn.Module) -> Tensor:
        items = self._parameter_items(module)
        pieces = []
        for _, parameter in items:
            if parameter.grad is None:
                pieces.append(torch.zeros_like(parameter, dtype=torch.float32).flatten())
            else:
                pieces.append(parameter.grad.detach().float().flatten())
        return torch.cat(pieces)

    def record_gradient(self, module: nn.Module) -> None:
        """Store one detached shared gradient vector for the current task."""

        self._gradient_history.append(self._pack_gradients(module).cpu())

    def consolidate(self, task_id: int, module: nn.Module) -> SubspaceAnchor | None:
        """Fit and store a rank-limited task gradient subspace."""

        if not self._gradient_history:
            return None
        packed, layout = self._pack_parameters(module)
        gradients = torch.stack(self._gradient_history)
        self._gradient_history.clear()
        finite_rows = torch.isfinite(gradients).all(dim=1)
        gradients = gradients[finite_rows]
        if gradients.numel() == 0:
            return None
        # Use the second-moment task-gradient subspace rather than centering
        # it; with one minibatch the task still contributes a valid direction.
        _, singular_values, right_vectors = torch.linalg.svd(gradients, full_matrices=False)
        rank = min(self.rank, right_vectors.shape[0], right_vectors.shape[1])
        if rank < 1:
            return None
        basis = right_vectors[:rank].T.contiguous()
        spectrum = singular_values[:rank].square()
        spectrum = spectrum / spectrum.max().clamp_min(1e-8)
        names = tuple(item[0] for item in layout)
        shapes = tuple(item[1] for item in layout)
        anchor = SubspaceAnchor(
            task_id=int(task_id),
            names=names,
            shapes=shapes,
            reference=packed.cpu(),
            basis=basis.cpu(),
            spectrum=spectrum.cpu(),
            mean_gradient=gradients.mean(dim=0).cpu(),
        )
        self.anchors[int(task_id)] = anchor
        return anchor

    def _relation(
        self,
        repository: TaskRepository,
        new_task_id: int,
        old_task_id: int,
    ) -> float:
        similarity = repository.task_similarity(new_task_id, old_task_id)
        if self.metric is not None:
            new_prototype = repository.get(new_task_id).prototype
            old_prototype = repository.get(old_task_id).prototype
            geometric_distance = float(
                self.metric.distance(new_prototype, old_prototype, old_task_id).detach()
            )
            similarity *= float(
                torch.exp(torch.tensor(-geometric_distance / self.geometry_temperature))
            )
        return self.relation_floor + (1.0 - self.relation_floor) * similarity

    @staticmethod
    def _current_by_name(module: nn.Module, names: tuple[str, ...]) -> Tensor:
        parameters = dict(module.named_parameters())
        missing = [name for name in names if name not in parameters]
        if missing:
            raise ValueError(f"module is missing RGSC parameters: {missing[:3]}")
        return torch.cat([parameters[name].detach().float().flatten() for name in names])

    @staticmethod
    def _gradient_by_name(module: nn.Module, names: tuple[str, ...]) -> tuple[Tensor, list[tuple[nn.Parameter, int, int]]]:
        parameters = dict(module.named_parameters())
        pieces: list[Tensor] = []
        layout: list[tuple[nn.Parameter, int, int]] = []
        offset = 0
        for name in names:
            parameter = parameters[name]
            gradient = (
                torch.zeros_like(parameter, dtype=torch.float32)
                if parameter.grad is None
                else parameter.grad.detach().float()
            )
            end = offset + gradient.numel()
            pieces.append(gradient.flatten())
            layout.append((parameter, offset, end))
            offset = end
        return torch.cat(pieces), layout

    def penalty(
        self,
        module: nn.Module,
        repository: TaskRepository,
        new_task_id: int,
    ) -> Tensor:
        """Return the relation-gated low-rank displacement penalty."""

        if not self.anchors:
            return next(module.parameters()).new_zeros(())
        terms: list[Tensor] = []
        for old_task_id, anchor in self.anchors.items():
            if old_task_id not in repository.records or new_task_id not in repository.records:
                continue
            current = self._current_by_name(module, anchor.names).to(anchor.basis.device)
            displacement = current - anchor.reference.to(current.device)
            basis = anchor.basis.to(current.device)
            spectrum = anchor.spectrum.to(current.device)
            coordinates = basis.T @ displacement
            relation = self._relation(repository, new_task_id, old_task_id)
            terms.append(relation * (spectrum * coordinates.square()).mean())
        if not terms:
            return next(module.parameters()).new_zeros(())
        return self.penalty_strength * torch.stack(terms).mean()

    @torch.no_grad()
    def project_gradients(
        self,
        module: nn.Module,
        repository: TaskRepository,
        new_task_id: int,
    ) -> None:
        """Softly remove gradient components in related old-task subspaces."""

        if not self.anchors:
            return
        for old_task_id, anchor in self.anchors.items():
            if old_task_id not in repository.records or new_task_id not in repository.records:
                continue
            gradient, layout = self._gradient_by_name(module, anchor.names)
            basis = anchor.basis.to(gradient.device)
            relation = self._relation(repository, new_task_id, old_task_id)
            old_coordinates = basis.T @ anchor.mean_gradient.to(gradient.device)
            new_coordinates = basis.T @ gradient
            # Only opposing signed components are conflicts. A compatible
            # update in a retained direction is useful transfer and should
            # remain trainable.
            conflict = (old_coordinates * new_coordinates < 0.0).to(gradient.dtype)
            projected = basis @ (new_coordinates * conflict)
            updated = gradient - self.projection_strength * relation * projected
            for parameter, start, end in layout:
                if parameter.grad is not None:
                    parameter.grad.copy_(updated[start:end].view_as(parameter).to(parameter.grad.dtype))

    def diagnostics(self, repository: TaskRepository, new_task_id: int) -> dict[int, float]:
        return {
            old_task_id: self._relation(repository, new_task_id, old_task_id)
            for old_task_id in self.anchors
            if old_task_id in repository.records and new_task_id in repository.records
        }
