from __future__ import annotations

from src.bonsai.matched_baselines import run_current_backbone_comparison


def test_current_backbone_controls_share_stream_and_account_task_awareness() -> None:
    records = run_current_backbone_comparison(
        seeds=(7,),
        num_tasks=2,
        classes_per_task=2,
        train_samples_per_class=4,
        test_samples_per_class=4,
        image_size=16,
        noise=0.1,
        epochs=1,
        batch_size=8,
    )

    assert {record["method"] for record in records} == {
        "current_ewc",
        "current_si",
        "current_packnet",
        "current_pnn",
    }
    by_method = {record["method"]: record for record in records}
    assert by_method["current_ewc"]["backbone"] == "current_bonsai_vib_mlp"
    assert by_method["current_si"]["consolidation_memory_elements"] == 2 * by_method["current_si"]["parameter_count"]
    assert by_method["current_packnet"]["consolidation_memory_elements"] == by_method["current_packnet"]["parameter_count"]
    assert by_method["current_pnn"]["task_free_supported"] is False
    assert by_method["current_pnn"]["parameter_overhead_percent"] == 100.0
