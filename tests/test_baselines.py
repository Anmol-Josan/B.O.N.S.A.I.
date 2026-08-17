from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.baselines import EWC, PNN, SI, PackNet, build_resnet18


def test_all_baselines_share_resnet18_forward_contract() -> None:
    inputs = torch.randn(2, 3, 32, 32)
    models = [
        EWC(num_classes=5),
        SI(num_classes=5),
        PackNet(num_classes=5),
        PNN(num_classes=5),
    ]

    for model in models:
        assert model(inputs).shape == (2, 5)
    assert sum(parameter.numel() for parameter in build_resnet18(5).parameters()) == sum(
        parameter.numel() for parameter in models[0].backbone.parameters()
    )


def test_ewc_consolidates_fisher_and_penalty_is_nonnegative() -> None:
    model = EWC(num_classes=3)
    loader = DataLoader(TensorDataset(torch.randn(2, 3, 32, 32), torch.tensor([0, 1])), batch_size=2)
    model.consolidate(loader, max_batches=1)

    assert model.ewc_penalty().item() >= 0.0
    assert model.fisher


def test_si_and_packnet_record_task_state() -> None:
    si = SI(num_classes=3)
    before = {name: parameter.detach().clone() for name, parameter in si.named_parameters()}
    si.update_synaptic_importance(before)
    si.consolidate_task()
    assert si.si_penalty().item() >= 0.0

    packnet = PackNet(num_classes=3)
    masks = packnet.prune_by_magnitude(0.5)
    assert masks
    assert packnet.frozen_parameter_count > 0


def test_pnn_adds_a_new_column_for_each_task() -> None:
    model = PNN(num_classes=3)
    initial = model.total_parameters
    model.add_task_column()

    assert len(model.columns) == 2
    assert model.total_parameters > initial
    assert model(torch.randn(2, 3, 32, 32), task_id=1).shape == (2, 3)


def test_progressive_columns_can_use_task_local_heads() -> None:
    model = PNN(num_classes=6, task_classes=3)
    model.add_task_column()

    assert model(torch.randn(2, 3, 32, 32), task_id=1).shape == (2, 3)
