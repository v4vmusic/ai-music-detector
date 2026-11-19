from __future__ import annotations

import math

import pytest

from aidetect.calibration import (
    calibration_fallback,
    label_for_probability,
    temperature_scale_scores,
)
from aidetect.constants import AI_GENERATED_LABEL, HUMAN_LABEL, UNCERTAIN_LABEL


def test_temperature_scale_scores_matches_reference() -> None:
    scores = [0.1, 0.5, 0.9]
    scaled = temperature_scale_scores(scores, temperature=2.0)
    assert len(scaled) == len(scores)
    for value in scaled:
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0
    assert scaled[1] == pytest.approx(0.5)


def test_temperature_scale_scores_rejects_non_positive_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        temperature_scale_scores([0.5], temperature=0.0)


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, HUMAN_LABEL),
        (0.34, HUMAN_LABEL),
        (0.35, UNCERTAIN_LABEL),
        (0.5, UNCERTAIN_LABEL),
        (0.65, UNCERTAIN_LABEL),
        (0.66, AI_GENERATED_LABEL),
        (1.0, AI_GENERATED_LABEL),
    ],
)
def test_label_for_probability_uses_inclusive_uncertain_edges(
    probability: float, expected: str
) -> None:
    assert label_for_probability(probability, low=0.35, high=0.65) == expected


def test_label_for_probability_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="probability"):
        label_for_probability(1.1, low=0.35, high=0.65)


def test_calibration_fallback_uses_fixed_thresholds_and_warns() -> None:
    fallback = calibration_fallback(reason="degenerate dev scores")
    assert fallback.temperature == 1.0
    assert fallback.low_threshold == 0.35
    assert fallback.high_threshold == 0.65
    assert fallback.warnings and "calibration_fallback" in fallback.warnings[0]
