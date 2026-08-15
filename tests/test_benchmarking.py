from __future__ import annotations

from src.utils.benchmarking import ToyBenchmarkConfig, run_toy_suite


def test_toy_suite_writes_runs_summaries_and_plots(tmp_path) -> None:
    records, summaries = run_toy_suite(
        ToyBenchmarkConfig(seeds=(3,), epochs_per_task=1, samples_per_class=8, output_dir=tmp_path)
    )

    assert len(records) == 5
    assert len(summaries) == 5
    assert (tmp_path / "runs.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "accuracy_curves.png").exists()
    assert (tmp_path / "mask_sparsity.png").exists()
