"""Small BONSAI prediction model composed around the VIB and shared adapter."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from src.bonsai.adapters import SharedLowRankAdapter
from src.bonsai.vib import VIBEncoder, VIBOutput


@dataclass
class BONSAIModelOutput:
    logits: Tensor
    z: Tensor
    vib: VIBOutput


class BONSAIModel(nn.Module):
    """Compact shared classifier; task specificity lives in adapter coefficients."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        vib_beta: float = 1e-3,
        adapter_rank: int = 2,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        self.encoder = VIBEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            beta=vib_beta,
        )
        self.adapter = SharedLowRankAdapter(latent_dim, latent_dim, rank=adapter_rank)
        self.classifier = nn.Linear(latent_dim, num_classes)
        self.initial_parameter_count = self.total_parameters

    @property
    def total_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def parameter_overhead(self) -> float:
        return (self.total_parameters - self.initial_parameter_count) / self.initial_parameter_count

    def add_task(self, task_id: int) -> None:
        self.adapter.add_task(task_id)

    def encode(self, inputs: Tensor, sample: bool = True) -> VIBOutput:
        return self.encoder(inputs, sample=sample)

    def deterministic_features(self, inputs: Tensor, task_id: int | None = None) -> Tensor:
        z = self.encoder.deterministic(inputs)
        return self.adapter(z, task_id=task_id)

    def forward(self, inputs: Tensor, task_id: int | None = None, sample: bool | None = None) -> BONSAIModelOutput:
        vib = self.encoder(inputs, sample=sample)
        adapted = self.adapter(vib.z, task_id=task_id)
        return BONSAIModelOutput(logits=self.classifier(adapted), z=adapted, vib=vib)
