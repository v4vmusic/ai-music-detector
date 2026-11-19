from __future__ import annotations

import math

from aidetect.errors import ModelLoadError

__all__ = ["as_probability", "sigmoid"]


def sigmoid(logit: float) -> float:
    """Numerically stable sigmoid of a single logit, with no overflow for extremes."""

    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def as_probability(value: float) -> float:
    """Validate a model-supplied probability without transforming it."""

    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ModelLoadError(f"Model output is not a valid probability: {value!r}")
    return float(value)
