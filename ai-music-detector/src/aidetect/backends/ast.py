from __future__ import annotations

from collections.abc import Hashable, Mapping
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from aidetect.constants import SAMPLE_RATE
from aidetect.errors import AudioDecodeError, ModelLoadError
from aidetect.probability import as_probability
from aidetect.registry import AstSpec, get_ast_spec
from aidetect.types import (
    AudioWindow,
    ModelMetadata,
    ModelOutput,
    Timings,
    validate_audio_window,
)

__all__ = ["ASTBackend", "Pipeline", "PipelineLoader", "ai_probability", "load"]


class Pipeline(Protocol):
    """Narrow model surface: probabilities keyed by class id, plus id2label and unload."""

    def __call__(self, samples: NDArray[np.float32], *, sampling_rate: int) -> Mapping[int, float]:
        """Return one probability per class id."""

    def id2label(self) -> Mapping[int, str]:
        """Return the checkpoint's class id to label mapping."""

    def unload(self) -> None:
        """Release any model handle held by the pipeline."""


class PipelineLoader(Protocol, Hashable):
    """Builds a pipeline for the AST spec (runner or test fake)."""

    def __call__(self, spec: AstSpec, *, cache_dir: Path | None, device: str) -> Pipeline: ...


def ai_probability(
    predictions: Mapping[int, float], id2label: Mapping[int, str], positive_label: str
) -> float:
    """Resolve the AI class through id2label, never by assuming its class index."""

    labels = {int(index): label for index, label in id2label.items()}
    matching = [index for index, label in labels.items() if label == positive_label]
    if len(matching) != 1:
        raise ModelLoadError(
            f"Expected exactly one class labelled {positive_label!r} in id2label, got {labels!r}"
        )
    index = matching[0]
    if index not in predictions:
        raise ModelLoadError(f"missing score for class {index} ({positive_label!r})")
    return as_probability(float(predictions[index]))


class ASTBackend:
    """Scores shared 60 s windows with an injected pipeline (no file I/O here)."""

    def __init__(self, spec: AstSpec, pipeline: Pipeline | None) -> None:
        self._spec = spec
        self._pipeline = pipeline

    def score(self, window: AudioWindow) -> ModelOutput:
        validate_audio_window(window)
        expected_samples = int(round(self._spec.window_seconds * SAMPLE_RATE))
        if window.samples.size != expected_samples:
            raise AudioDecodeError(
                f"Expected {expected_samples} samples for {self._spec.key}, "
                f"got {window.samples.size}"
            )
        pipeline = self._pipeline
        if pipeline is None:
            raise ModelLoadError(f"AST backend for {self._spec.key} is unloaded")

        started = perf_counter()
        predictions = pipeline(window.samples, sampling_rate=SAMPLE_RATE)
        probability = ai_probability(predictions, pipeline.id2label(), self._spec.positive_label)
        window_description = (
            f"{self._spec.window_seconds:g}s@{self._spec.sample_rate}Hz@{window.offset_seconds:g}s"
        )
        return ModelOutput(
            raw_score=probability,
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
        if self._pipeline is not None:
            self._pipeline.unload()
        self._pipeline = None
        _build_pipeline.cache_clear()


@lru_cache(maxsize=1)
def _build_pipeline(
    key: str, cache_dir: Path | None, device: str, loader: PipelineLoader
) -> Pipeline:
    """Create at most one pipeline per (key, cache dir, device, loader) tuple."""

    return loader(get_ast_spec(key), cache_dir=cache_dir, device=device)


def load(
    key: str,
    *,
    pipeline: Pipeline | None = None,
    pipeline_loader: PipelineLoader | None = None,
    cache_dir: Path | None = None,
    device: str = "cpu",
) -> ASTBackend:
    """Build a backend from a pipeline; weights and I/O belong to the supplied loader."""

    spec = get_ast_spec(key)
    if pipeline is None:
        if pipeline_loader is None:
            raise ModelLoadError("AST needs an injected pipeline; the runner owns weight loading")
        pipeline = _build_pipeline(key, cache_dir, device, pipeline_loader)
    return ASTBackend(spec=spec, pipeline=pipeline)
