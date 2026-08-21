from __future__ import annotations

import json

from src.utils.metrics import (
    average_accuracy,
    forgetting_measure,
    parameter_overhead,
    save_metrics_json,
)


def test_continual_metrics_match_published_definitions() -> None:
    history = [[0.8], [0.75, 0.9], [0.7, 0.88, 0.95]]

    assert average_accuracy(history[-1]) == (0.7 + 0.88 + 0.95) / 3
    assert forgetting_measure(history) == ((0.8 - 0.7) + (0.9 - 0.88)) / 2
    assert parameter_overhead(initial_parameters=100, total_parameters=125) == 25.0


def test_forgetting_uses_peak_before_final() -> None:
    history = [[0.6], [0.8, 0.7], [0.75, 0.72, 0.9]]

    assert forgetting_measure(history) == ((0.8 - 0.75) + (0.7 - 0.72)) / 2


def test_metrics_json_is_structured_and_round_trips(tmp_path) -> None:
    destination = tmp_path / "metrics.json"
    payload = {"seed": 7, "average_accuracy": 0.8, "forgetting": 0.1}

    save_metrics_json(destination, payload)

    assert json.loads(destination.read_text()) == payload
