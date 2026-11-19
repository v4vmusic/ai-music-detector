from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from benches._shared.constants import SAMPLE_RATE
from benches._shared.errors import AudioDecodeError

__all__ = ["AudioWindow", "ModelMetadata", "ModelOutput", "Timings"]


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


@dataclass(frozen=True, slots=True)
class AudioWindow:
    """One 16 kHz mono float32 window ready for a model backend."""

    samples: NDArray[np.float32]
    sample_rate: int = SAMPLE_RATE
    offset_seconds: float = 0.0
    window_seconds: float | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.sample_rate != SAMPLE_RATE:
            raise AudioDecodeError(f"Expected {SAMPLE_RATE} Hz audio, got {self.sample_rate} Hz")
        if self.samples.ndim != 1:
            raise AudioDecodeError(f"Expected mono audio, got shape {self.samples.shape}")
        if self.samples.dtype != np.float32:
            raise AudioDecodeError(f"Expected float32 samples, got {self.samples.dtype}")
        if not bool(np.all(np.isfinite(self.samples))):
            raise AudioDecodeError("Audio samples must be finite")
        if bool(np.any(np.abs(self.samples) > 1.0)):
            raise AudioDecodeError("Audio samples must be within [-1.0, 1.0]")
        _require_non_negative(self.offset_seconds, "offset_seconds")
        if self.window_seconds is not None:
            _require_finite(self.window_seconds, "window_seconds")
            if self.window_seconds <= 0.0:
                raise ValueError("window_seconds must be positive")


@dataclass(frozen=True, slots=True)
class Timings:
    """Wall-clock timings, in seconds, for one model score."""

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
