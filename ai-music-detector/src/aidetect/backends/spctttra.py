from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from aidetect.constants import SAMPLE_RATE
from aidetect.errors import AudioDecodeError, ModelLoadError
from aidetect.probability import sigmoid
from aidetect.registry import SpecTTTraSpec, get_specttra_spec
from aidetect.types import (
    AudioWindow,
    ModelMetadata,
    ModelOutput,
    Timings,
    validate_audio_window,
)

__all__ = ["Classifier", "ClassifierLoader", "SpecTTTraBackend", "load", "sigmoid"]


class Classifier(Protocol):
    """The narrow classifier surface the backend needs; fakes fit without torch."""

    def __call__(self, samples: NDArray[np.float32]) -> float:
        """Return the single AI-generated logit for one 120 s window."""

    def unload(self) -> None:
        """Release any model handle held by the classifier."""


class ClassifierLoader(Protocol):
    """Builds a classifier for a registered variant (runner or test fake)."""

    def __call__(
        self, spec: SpecTTTraSpec, *, cache_dir: Path | None, device: str
    ) -> Classifier: ...


class SpecTTTraBackend:
    """Scores shared 120 s windows with an injected classifier (no file I/O here)."""

    def __init__(self, spec: SpecTTTraSpec, classifier: Classifier | None) -> None:
        self._spec = spec
        self._classifier = classifier

    def score(self, window: AudioWindow) -> ModelOutput:
        validate_audio_window(window)
        expected_samples = int(round(self._spec.window_seconds * SAMPLE_RATE))
        if window.samples.size != expected_samples:
            raise AudioDecodeError(
                f"Expected {expected_samples} samples for {self._spec.key}, "
                f"got {window.samples.size}"
            )
        classifier = self._classifier
        if classifier is None:
            raise ModelLoadError(f"SpecTTTra backend for {self._spec.key} is unloaded")

        started = perf_counter()
        logit = classifier(window.samples)
        if not math.isfinite(logit):
            raise ModelLoadError(f"Classifier for {self._spec.key} returned a non-finite logit")
        probability = sigmoid(logit)
        window_description = (
            f"{self._spec.window_seconds:g}s@{self._spec.sample_rate}Hz@{window.offset_seconds:g}s"
        )
        return ModelOutput(
            raw_score=logit,
            calibrated_probability=probability,
            model_id=self._spec.model_id,
            model_revision=self._spec.revision,
            window=window_description,
            warnings=window.warnings,
            timings=Timings(score_seconds=perf_counter() - started),
        )

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self._spec.model_id,
            revision=self._spec.revision,
            window_seconds=self._spec.window_seconds,
            sample_rate=self._spec.sample_rate,
            preprocessing_version=self._spec.preprocessing_version,
        )

    def unload(self) -> None:
        if self._classifier is not None:
            self._classifier.unload()
        self._classifier = None


def load(
    key: str,
    *,
    classifier: Classifier | None = None,
    classifier_loader: ClassifierLoader | None = None,
    cache_dir: Path | None = None,
    device: str = "cuda",
) -> SpecTTTraBackend:
    """Build a backend from a classifier; weights and I/O belong to the supplied loader."""

    spec = get_specttra_spec(key)
    if classifier is None:
        if classifier_loader is None:
            raise ModelLoadError(
                "SpecTTTra needs an injected classifier; the runner owns weight loading"
            )
        classifier = classifier_loader(spec, cache_dir=cache_dir, device=device)
    return SpecTTTraBackend(spec=spec, classifier=classifier)
