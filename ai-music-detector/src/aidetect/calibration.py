from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from aidetect.constants import (
    AI_GENERATED_LABEL,
    FALLBACK_HIGH_THRESHOLD,
    FALLBACK_LOW_THRESHOLD,
    HUMAN_LABEL,
    UNCERTAIN_LABEL,
    TriStateLabel,
)
from aidetect.probability import sigmoid

__all__ = [
    "Calibration",
    "calibration_fallback",
    "label_for_probability",
    "temperature_scale_scores",
]


@dataclass(frozen=True, slots=True)
class Calibration:
    """Packaged or fallback temperature and tri-state thresholds for one backend."""

    temperature: float
    low_threshold: float
    high_threshold: float
    warnings: tuple[str, ...] = ()
    model_id: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be positive and finite")
        for name in ("low_threshold", "high_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite probability")
        if self.low_threshold > self.high_threshold:
            raise ValueError("low_threshold must not exceed high_threshold")


def _raw_score_to_logit(raw_score: float) -> float:
    if not math.isfinite(raw_score):
        raise ValueError("raw_score must be finite")
    if 0.0 <= raw_score <= 1.0:
        clipped = min(max(raw_score, 1e-7), 1.0 - 1e-7)
        return math.log(clipped / (1.0 - clipped))
    return raw_score


def temperature_scale_scores(raw_scores: Sequence[float], temperature: float) -> list[float]:
    """Apply the bench temperature transform to a batch of raw backend scores.

    Scores already in ``[0, 1]`` are logit-transformed first; logits outside that range are
    scaled directly. This preserves the semantics of ``benches/_shared/eval.py`` so packaged
    calibrations are applied exactly as they were fitted.
    """

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return [sigmoid(_raw_score_to_logit(float(score)) / temperature) for score in raw_scores]


def label_for_probability(probability: float, *, low: float, high: float) -> TriStateLabel:
    """Return the conservative tri-state label, with both thresholds inside ``uncertain``."""

    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be a finite value in [0.0, 1.0]")
    if not math.isfinite(low) or not math.isfinite(high) or not 0.0 <= low <= high <= 1.0:
        raise ValueError("thresholds must satisfy 0.0 <= low <= high <= 1.0")
    if probability < low:
        return HUMAN_LABEL
    if probability > high:
        return AI_GENERATED_LABEL
    return UNCERTAIN_LABEL


def calibration_fallback(*, reason: str) -> Calibration:
    """Return the fixed conservative fallback and an actionable warning."""

    return Calibration(
        temperature=1.0,
        low_threshold=FALLBACK_LOW_THRESHOLD,
        high_threshold=FALLBACK_HIGH_THRESHOLD,
        warnings=(f"calibration_fallback: {reason}",),
    )
