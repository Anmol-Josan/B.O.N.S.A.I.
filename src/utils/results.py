"""Structured experiment result persistence and seed aggregation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Sequence


def _ensure_parent(destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def save_records_csv(destination: str | Path, records: Sequence[dict[str, Any]]) -> None:
    destination = _ensure_parent(destination)
    fieldnames = sorted({key for record in records for key in record})
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def summarize_records(
    records: Sequence[dict[str, Any]], group_keys: tuple[str, ...] = ("dataset", "method")
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = tuple(record.get(group_key) for group_key in group_keys)
        groups.setdefault(key, []).append(record)
    summaries: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        summary = {group_key: value for group_key, value in zip(group_keys, key)}
        numeric_keys = sorted(
            key_name
            for key_name, value in group[0].items()
            if key_name not in group_keys
            and key_name != "seed"
            and isinstance(value, (int, float))
        )
        for numeric_key in numeric_keys:
            values = [float(record[numeric_key]) for record in group if numeric_key in record]
            summary[f"{numeric_key}_mean"] = fmean(values)
            summary[f"{numeric_key}_std"] = round(pstdev(values), 12) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def write_summary_artifacts(destination: str | Path, summaries: Sequence[dict[str, Any]]) -> None:
    destination = _ensure_parent(destination)
    destination.write_text(json.dumps(list(summaries), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_destination = destination.with_suffix(".csv")
    save_records_csv(csv_destination, summaries)
