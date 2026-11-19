from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aidetect.constants import (
    AI_GENERATED_LABEL,
    HUMAN_LABEL,
    SAMPLE_RATE,
    TRI_STATE_LABELS,
    UNCERTAIN_LABEL,
    Placement,
    TriStateLabel,
)
from aidetect.errors import AudioDecodeError

__all__ = [
    "AudioWindow",
    "CascadeResult",
    "ModelMetadata",
    "ModelOutput",
    "Result",
    "Timings",
    "validate_audio_window",
]


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _require_non_negative(value: float, name: str) -> None:
    _require_finite(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _require_non_empty(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def validate_audio_window(window: AudioWindow) -> None:
    """Check the shared audio contract for a window before a backend scores it."""

    if window.sample_rate != SAMPLE_RATE:
        raise AudioDecodeError(f"Expected {SAMPLE_RATE} Hz audio, got {window.sample_rate} Hz")
    if window.samples.ndim != 1:
        raise AudioDecodeError(f"Expected mono audio, got shape {window.samples.shape}")
    if window.samples.dtype != np.float32:
        raise AudioDecodeError(f"Expected float32 samples, got {window.samples.dtype}")
    if not bool(np.all(np.isfinite(window.samples))):
        raise AudioDecodeError("Audio samples must be finite")
    if bool(np.any(np.abs(window.samples) > 1.0)):
        raise AudioDecodeError("Audio samples must be within [-1.0, 1.0]")
    _require_non_negative(window.offset_seconds, "offset_seconds")
    if window.window_seconds is not None:
        _require_finite(window.window_seconds, "window_seconds")
        if window.window_seconds <= 0.0:
            raise ValueError("window_seconds must be positive")


@dataclass(frozen=True, slots=True)
class AudioWindow:
    """One 16 kHz mono float32 window ready for a model backend."""

    samples: NDArray[np.float32]
    sample_rate: int = SAMPLE_RATE
    offset_seconds: float = 0.0
    window_seconds: float | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_audio_window(self)


@dataclass(frozen=True, slots=True)
class Timings:
    """Wall-clock timings, in seconds, for one score."""

    score_seconds: float
    load_seconds: float | None = None
    gpu_seconds: float | None = None

    def __post_init__(self) -> None:
        _require_non_negative(self.score_seconds, "score_seconds")
        if self.load_seconds is not None:
            _require_non_negative(self.load_seconds, "load_seconds")
        if self.gpu_seconds is not None:
            _require_non_negative(self.gpu_seconds, "gpu_seconds")


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """A backend's score, in the shared shape used by JSONL runners."""

    raw_score: float
    calibrated_probability: float | None
    model_id: str
    model_revision: str
    window: str
    warnings: tuple[str, ...]
    timings: Timings

    def __post_init__(self) -> None:
        _require_finite(self.raw_score, "raw_score")
        if self.calibrated_probability is not None:
            _require_finite(self.calibrated_probability, "calibrated_probability")
            if not 0.0 <= self.calibrated_probability <= 1.0:
                raise ValueError(
                    "calibrated_probability must be in [0.0, 1.0], "
                    f"got {self.calibrated_probability!r}"
                )
        _require_non_empty(self.model_id, "model_id")
        _require_non_empty(self.model_revision, "model_revision")
        _require_non_empty(self.window, "window")


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Registered facts about a backend, without model ids or revisions duplicated."""

    model_id: str
    revision: str
    window_seconds: float
    sample_rate: int = SAMPLE_RATE
    preprocessing_version: str = "1"

    def __post_init__(self) -> None:
        _require_non_empty(self.model_id, "model_id")
        _require_non_empty(self.revision, "revision")
        _require_finite(self.window_seconds, "window_seconds")
        if self.window_seconds <= 0.0:
            raise ValueError("window_seconds must be positive")
        if self.sample_rate != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {self.sample_rate} Hz")
        _require_non_empty(self.preprocessing_version, "preprocessing_version")


@dataclass(frozen=True, slots=True)
class Result:
    """A calibrated probability and its derived tri-state label for one track."""

    probability: float
    label: TriStateLabel
    model_id: str
    model_revision: str
    window: str
    placement: Placement
    raw_score: float
    warnings: tuple[str, ...]
    timings: Timings

    def __post_init__(self) -> None:
        _require_finite(self.probability, "probability")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0.0, 1.0], got {self.probability!r}")
        if self.label not in TRI_STATE_LABELS:
            raise ValueError(f"label must be one of {TRI_STATE_LABELS}, got {self.label!r}")
        _require_non_empty(self.model_id, "model_id")
        _require_non_empty(self.model_revision, "model_revision")
        _require_non_empty(self.window, "window")


@dataclass(frozen=True, slots=True)
class CascadeResult:
    """Conservative two-stage outcome: only stage one may auto-label ``ai_generated``."""

    label: TriStateLabel
    primary: Result
    second_opinion: Result | None
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.label not in (HUMAN_LABEL, UNCERTAIN_LABEL, AI_GENERATED_LABEL):
            raise ValueError(f"invalid cascade label {self.label!r}")
        if self.label == AI_GENERATED_LABEL and self.second_opinion is not None:
            raise ValueError("an auto-AI cascade result must not carry a second-opinion stage")
