from __future__ import annotations

import csv
import json

from src.utils.results import save_records_csv, summarize_records, write_summary_artifacts


def test_results_are_aggregated_by_method_and_dataset(tmp_path) -> None:
    records = [
        {"dataset": "synthetic", "method": "BONSAI", "seed": 1, "average_accuracy": 0.8},
        {"dataset": "synthetic", "method": "BONSAI", "seed": 2, "average_accuracy": 0.6},
    ]
    summary = summarize_records(records)

    assert summary[0]["average_accuracy_mean"] == 0.7
    assert summary[0]["average_accuracy_std"] == 0.1

    csv_path = tmp_path / "runs.csv"
    json_path = tmp_path / "summary.json"
    save_records_csv(csv_path, records)
    write_summary_artifacts(json_path, summary)
    assert len(list(csv.DictReader(csv_path.open(newline="")))) == 2
    assert json.loads(json_path.read_text())[0]["method"] == "BONSAI"

