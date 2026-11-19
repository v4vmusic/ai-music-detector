from __future__ import annotations

import math

import pytest

from aidetect.errors import ModelLoadError
from aidetect.probability import as_probability, sigmoid


@pytest.mark.parametrize(
    "logit,expected",
    [(0.0, 0.5), (2.0, 1.0 / (1.0 + math.exp(-2.0))), (-3.0, 1.0 / (1.0 + math.exp(3.0)))],
)
def test_sigmoid_matches_reference(logit: float, expected: float) -> None:
    assert sigmoid(logit) == pytest.approx(expected)


@pytest.mark.parametrize("logit", [-1_000.0, 1_000.0])
def test_sigmoid_is_finite_for_extreme_logits(logit: float) -> None:
    probability = sigmoid(logit)
    assert math.isfinite(probability)
    assert 0.0 <= probability <= 1.0


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_as_probability_accepts_valid_values(value: float) -> None:
    assert as_probability(value) == value


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf, -math.inf])
def test_as_probability_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ModelLoadError, match="probability"):
        as_probability(value)
