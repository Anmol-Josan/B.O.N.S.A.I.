"""Continual-learning protection and lightweight BONSAI training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from src.bonsai.system import BONSAISystem
from src.bonsai.replay import AdaptiveTaskGraphFunctionalReplay
from src.bonsai.rgsc import TopologyGatedRiemannianSubspaceConsolidator


@dataclass
class LossWeights:
    """Independently ablatable terms of the combined objective."""

    route: float = 0.25
    vib: float = 1.0
    geometry: float = 0.15
    ot: float = 0.05
    sheaf: float = 0.02
    interference: float = 0.5
    functional_replay: float = 1.0


class InterferenceProtector:
    """EWC-style parameter protection for shared BONSAI parameters."""

    def __init__(self, strength: float = 1.0) -> None:
        if strength < 0.0:
            raise ValueError("strength must be nonnegative")
        self.strength = strength
        self.references: dict[str, Tensor] = {}
        self.importance: dict[str, Tensor] = {}

    def consolidate(self, module: torch.nn.Module, gradients: dict[str, Tensor] | None = None) -> None:
        """Snapshot shared parameters and squared gradients after a task."""

        # A disabled protector is used by the projection and replay variants.
        # Retaining EWC references in that case silently consumes O(P) memory
        # without contributing any loss, which is especially wasteful for
        # image inputs and obscures the true memory cost of the selected method.
        if self.strength == 0.0:
            return
        gradients = gradients or {}
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad or not parameter.is_floating_point():
                continue
            self.references[name] = parameter.detach().clone()
            gradient = gradients.get(name)
            if gradient is None:
                old = self.importance.get(name)
                self.importance[name] = torch.ones_like(parameter) if old is None else old
            else:
                value = gradient.detach().square()
                old = self.importance.get(name)
                self.importance[name] = value if old is None else 0.5 * old + 0.5 * value

    def penalty(self, module: torch.nn.Module) -> Tensor:
        terms: list[Tensor] = []
        for name, parameter in module.named_parameters():
            reference = self.references.get(name)
            importance = self.importance.get(name)
            if reference is None or importance is None:
                continue
            terms.append((importance.to(parameter.device) * (parameter - reference.to(parameter.device)).square()).mean())
        if not terms:
            try:
                return next(module.parameters()).new_zeros(())
            except StopIteration:
                return torch.zeros(())
        return self.strength * torch.stack(terms).mean()

    def snapshot(self, module: torch.nn.Module) -> dict[str, Tensor]:
        return {name: parameter.detach().clone() for name, parameter in module.named_parameters()}

    @property
    def stored_elements(self) -> int:
        """Number of float elements retained by diagonal protection."""

        return sum(value.numel() for value in self.references.values()) + sum(
            value.numel() for value in self.importance.values()
        )

    @staticmethod
    def parameter_drift(before: dict[str, Tensor], after: dict[str, Tensor]) -> float:
        numerator = 0.0
        denominator = 0.0
        for name, previous in before.items():
            if name not in after:
                continue
            numerator += float((after[name] - previous).square().sum())
            denominator += float(previous.square().sum())
        return numerator**0.5 / max(denominator**0.5, 1e-12)


def _task_route_loss(system: BONSAISystem, z: Tensor, task_id: int, margin: float = 1.0) -> Tensor:
    if system.repository.task_count <= 1:
        return z.new_zeros(())
    record = system.repository.get(task_id)
    others = [other for other in system.repository.task_ids if other != task_id]
    prototypes = system.repository.prototypes(others).to(z.device, z.dtype)
    own = system.metric.distance(z, record.prototype.to(z.device, z.dtype), task_id)
    other_distances = torch.stack(
        [system.metric.distance(record.prototype.to(z.device, z.dtype), prototype, task_id) for prototype in prototypes]
    )
    return torch.relu(margin + own.mean() - other_distances.min()).square()


class BONSAITrainer:
    """Minibatch trainer for the full modular architecture.

    Descriptors are built once when a task enters the repository and refreshed
    after its local training. OT/TDA do not run on each single-example route.
    """

    def __init__(
        self,
        system: BONSAISystem,
        learning_rate: float = 2e-3,
        weights: LossWeights | None = None,
        protection: InterferenceProtector | None = None,
        subspace_protection: TopologyGatedRiemannianSubspaceConsolidator | None = None,
        functional_replay: AdaptiveTaskGraphFunctionalReplay | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.system = system.to(device)
        self.learning_rate = learning_rate
        self.weights = weights or LossWeights()
        self.protection = protection or InterferenceProtector()
        self.subspace_protection = subspace_protection
        self.functional_replay = functional_replay
        self.device = torch.device(device)

    def _ensure_task(self, task_id: int, inputs: Tensor) -> None:
        if task_id not in self.system.model.adapter.task_ids:
            self.system.add_task(task_id)
        if task_id not in self.system.repository.records:
            with torch.no_grad():
                latents = self.system.model.deterministic_features(inputs.to(self.device))
            self.system.register_task(task_id, latents.detach())

    def fit_task(
        self,
        task_id: int,
        inputs: Tensor,
        labels: Tensor,
        epochs: int = 1,
        batch_size: int = 32,
    ) -> list[dict[str, float]]:
        if inputs.shape[0] != labels.shape[0] or inputs.shape[0] < 1:
            raise ValueError("inputs and labels must contain the same non-empty number of samples")
        if min(epochs, batch_size) < 1:
            raise ValueError("epochs and batch_size must be positive")
        self._ensure_task(task_id, inputs)
        self.system.model.adapter.train_only_task(task_id)
        optimizer = torch.optim.Adam(
            [parameter for parameter in self.system.parameters() if parameter.requires_grad],
            lr=self.learning_rate,
        )
        inputs = inputs.to(self.device)
        labels = labels.to(self.device).long()
        history: list[dict[str, float]] = []
        task_record = self.system.repository.get(task_id)
        reference_descriptor = task_record.distribution
        for _ in range(epochs):
            permutation = torch.randperm(inputs.shape[0], device=self.device)
            running: list[Tensor] = []
            gradient_accumulator: dict[str, Tensor] = {}
            self.system.train()
            for start in range(0, inputs.shape[0], batch_size):
                batch_indices = permutation[start : start + batch_size]
                batch_inputs = inputs[batch_indices]
                batch_labels = labels[batch_indices]
                output = self.system.model(batch_inputs, task_id=task_id, sample=True)
                task_loss = F.cross_entropy(output.logits, batch_labels)
                vib_loss = output.vib.kl * self.system.model.encoder.beta
                # Retrieval/repository geometry belongs to the shared VIB
                # chart. The task adapter is deliberately applied only to the
                # prediction path so routing cannot depend on a task id that
                # is unavailable at inference time.
                routing_latents = output.vib.z
                prototype = task_record.prototype.to(self.device, routing_latents.dtype)
                others = self.system.repository.prototypes(
                    [other for other in self.system.repository.task_ids if other != task_id]
                ).to(self.device, routing_latents.dtype)
                _, _, geom_loss = self.system.metric.geometry_loss(
                    routing_latents, task_id, prototype, others, margin=1.0
                )
                route_loss = _task_route_loss(self.system, routing_latents, task_id)
                if routing_latents.shape[0] >= self.system.router.minimum_episode_size:
                    query_descriptor = self.system.repository.ot.build(routing_latents)
                    ot_loss = self.system.repository.ot.distance(query_descriptor, reference_descriptor)
                else:
                    ot_loss = routing_latents.new_zeros(())
                sheaf_loss = self.system.sheaf.local_compatibility(
                    task_id,
                    routing_latents.mean(dim=0),
                    {
                        other: self.system.repository.get(other).prototype.to(self.device, routing_latents.dtype)
                        for other in self.system.repository.task_ids
                        if other != task_id
                    },
                ).mean()
                interference_loss = self.protection.penalty(self.system.model)
                subspace_loss = (
                    self.subspace_protection.penalty(
                        self.system.model, self.system.repository, task_id
                    )
                    if self.subspace_protection is not None
                    else output.z.new_zeros(())
                )
                functional_replay_loss = (
                    self.functional_replay.loss(self.system, task_id)
                    if self.functional_replay is not None
                    else output.z.new_zeros(())
                )
                loss = (
                    task_loss
                    + self.weights.route * route_loss
                    + self.weights.vib * vib_loss
                    + self.weights.geometry * geom_loss
                    + self.weights.ot * ot_loss
                    + self.weights.sheaf * sheaf_loss
                    + self.weights.interference * (interference_loss + subspace_loss)
                    + self.weights.functional_replay * functional_replay_loss
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("BONSAI objective became non-finite")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.subspace_protection is not None:
                    self.subspace_protection.record_gradient(self.system.model)
                    self.subspace_protection.project_gradients(
                        self.system.model, self.system.repository, task_id
                    )
                for name, parameter in self.system.model.named_parameters():
                    if parameter.grad is not None:
                        gradient_accumulator[name] = parameter.grad.detach().clone()
                torch.nn.utils.clip_grad_norm_(self.system.parameters(), max_norm=5.0)
                optimizer.step()
                running.append(loss.detach())
            self.protection.consolidate(self.system.model, gradient_accumulator)
            history.append(
                {"loss": float(torch.stack(running).mean()), "task_loss": float(task_loss.detach())}
            )
        if self.subspace_protection is not None:
            self.subspace_protection.consolidate(task_id, self.system.model)
        self.system.model.adapter.freeze_task(task_id)
        with torch.no_grad():
            refreshed = self.system.model.deterministic_features(inputs)
        self.system.router.update_task(task_id, refreshed.detach())
        if self.functional_replay is not None:
            self.functional_replay.consolidate_task(task_id, self.system, inputs, labels)
        return history
