"""Adaptive task-graph functional replay for the modular BONSAI trainer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from src.bonsai.system import BONSAISystem


@dataclass
class FunctionalMemory:
    """A small frozen input/output anchor set for one completed task."""

    task_id: int
    inputs: Tensor
    labels: Tensor
    logits: Tensor
    features: Tensor


class AdaptiveTaskGraphFunctionalReplay:
    """Coreset replay with relation gating and an output-drift thermostat.

    For old task ``u`` and current task ``t``, the replay loss is

    ``w(t,u) * (CE(y_u, f_theta(x_u)) +
    tau^2 KL(p_old^tau || p_new^tau) + gamma ||z_old-z_new||^2)``.

    ``w(t,u)`` comes from BONSAI's cached task graph.  A measured feature drift
    above ``drift_budget`` multiplies the weight, so the method spends more
    replay budget only when old behavior is actually moving.  The coreset is
    class-balanced and bounded per task; no old task is regenerated from test
    data.
    """

    def __init__(
        self,
        replay_per_task: int = 8,
        replay_strength: float = 1.0,
        distill_strength: float = 1.0,
        feature_strength: float = 0.25,
        temperature: float = 2.0,
        relation_floor: float = 0.25,
        drift_budget: float = 0.05,
        thermostat_gain: float = 2.0,
        max_multiplier: float = 8.0,
        relation_mode: str = "full",
    ) -> None:
        if replay_per_task < 1:
            raise ValueError("replay_per_task must be positive")
        if min(replay_strength, distill_strength, feature_strength) < 0.0:
            raise ValueError("replay strengths must be nonnegative")
        if temperature <= 0.0 or drift_budget <= 0.0 or thermostat_gain < 0.0:
            raise ValueError("temperature and drift budget must be positive")
        if not 0.0 <= relation_floor <= 1.0 or max_multiplier < 1.0:
            raise ValueError("invalid relation floor or multiplier")
        if relation_mode not in {"full", "no_ot", "no_tda", "euclidean", "uniform"}:
            raise ValueError(f"unknown relation mode: {relation_mode}")
        self.replay_per_task = replay_per_task
        self.replay_strength = replay_strength
        self.distill_strength = distill_strength
        self.feature_strength = feature_strength
        self.temperature = temperature
        self.relation_floor = relation_floor
        self.drift_budget = drift_budget
        self.thermostat_gain = thermostat_gain
        self.max_multiplier = max_multiplier
        self.relation_mode = relation_mode
        self.memory: dict[int, FunctionalMemory] = {}
        self.last_diagnostics: dict[int, dict[str, float]] = {}

    def _balanced_indices(self, labels: Tensor) -> Tensor:
        selected: list[int] = []
        positions = {
            int(label): (labels == label).nonzero(as_tuple=False).flatten().tolist()
            for label in torch.unique(labels, sorted=True).tolist()
        }
        offset = 0
        while len(selected) < min(self.replay_per_task, labels.numel()):
            added = False
            for class_positions in positions.values():
                if offset < len(class_positions) and len(selected) < self.replay_per_task:
                    selected.append(class_positions[offset])
                    added = True
            if not added:
                break
            offset += 1
        return torch.tensor(selected, dtype=torch.long)

    @torch.no_grad()
    def consolidate_task(
        self,
        task_id: int,
        system: BONSAISystem,
        inputs: Tensor,
        labels: Tensor,
    ) -> None:
        """Store detached current-task outputs after its training completes."""

        indices = self._balanced_indices(labels.detach().cpu())
        selected_inputs = inputs.detach().cpu()[indices]
        selected_labels = labels.detach().cpu()[indices].long()
        output = system.model(
            selected_inputs.to(next(system.parameters()).device),
            task_id=task_id,
            sample=False,
        )
        self.memory[int(task_id)] = FunctionalMemory(
            task_id=int(task_id),
            inputs=selected_inputs,
            labels=selected_labels,
            logits=output.logits.detach().cpu(),
            features=output.z.detach().cpu(),
        )

    def _relation(self, system: BONSAISystem, new_task_id: int, old_task_id: int) -> float:
        if new_task_id not in system.repository.records or old_task_id not in system.repository.records:
            return self.relation_floor
        similarity = system.repository.task_similarity(
            new_task_id, old_task_id, mode=self.relation_mode
        )
        return self.relation_floor + (1.0 - self.relation_floor) * similarity

    def loss(self, system: BONSAISystem, new_task_id: int) -> Tensor:
        """Return differentiable replay/distillation loss for old tasks."""

        if not self.memory:
            return next(system.parameters()).new_zeros(())
        device = next(system.parameters()).device
        terms: list[Tensor] = []
        weights: list[float] = []
        diagnostics: dict[int, dict[str, float]] = {}
        temperature = self.temperature
        for old_task_id, record in self.memory.items():
            inputs = record.inputs.to(device)
            labels = record.labels.to(device)
            old_logits = record.logits.to(device)
            old_features = record.features.to(device)
            output = system.model(inputs, task_id=old_task_id, sample=False)
            replay_loss = F.cross_entropy(output.logits, labels)
            distillation = F.kl_div(
                F.log_softmax(output.logits / temperature, dim=-1),
                F.softmax(old_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * temperature**2
            feature_loss = F.mse_loss(output.z, old_features)
            relative_drift = float(
                (output.z.detach() - old_features).square().mean()
                / old_features.square().mean().clamp_min(1e-6)
            )
            thermostat = min(
                self.max_multiplier,
                1.0
                + self.thermostat_gain
                * max(0.0, relative_drift / self.drift_budget - 1.0),
            )
            relation = self._relation(system, new_task_id, old_task_id)
            weight = relation * thermostat
            terms.append(
                self.replay_strength * replay_loss
                + self.distill_strength * distillation
                + self.feature_strength * feature_loss
            )
            weights.append(weight)
            diagnostics[old_task_id] = {
                "relation": relation,
                "relative_drift": relative_drift,
                "thermostat": thermostat,
            }
        self.last_diagnostics = diagnostics
        if not terms:
            return next(system.parameters()).new_zeros(())
        # Preserve the total replay mass while changing its allocation across
        # old tasks.  Without this normalization, a low graph similarity
        # silently turns relation gating into a smaller replay coefficient,
        # confounding "which task" with "how much replay".
        mean_weight = max(sum(weights) / len(weights), 1e-6)
        normalized = [term * (weight / mean_weight) for term, weight in zip(terms, weights)]
        return torch.stack(normalized).mean()

    @property
    def stored_elements(self) -> int:
        return sum(
            record.inputs.numel()
            + record.labels.numel()
            + record.logits.numel()
            + record.features.numel()
            for record in self.memory.values()
        )
