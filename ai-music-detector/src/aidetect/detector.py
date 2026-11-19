from __future__ import annotations

from pathlib import Path
from typing import Protocol

from aidetect.audio import decode_window, decode_window_bytes
from aidetect.backends import ast as ast_backend
from aidetect.backends import fakeprint as fakeprint_backend
from aidetect.backends import loaders
from aidetect.backends import spctttra as spctttra_backend
from aidetect.calibration import (
    Calibration,
    calibration_fallback,
    label_for_probability,
    temperature_scale_scores,
)
from aidetect.config import (
    DEFAULT_CONFIG_NAME,
    SECOND_OPINION_CONFIG_NAME,
    BackendConfig,
    get_backend_config,
    get_calibration,
    get_config_for_model,
)
from aidetect.errors import ModelLoadError
from aidetect.types import AudioWindow, CascadeResult, ModelOutput, Result

__all__ = ["CascadeDetector", "Detector", "Scorer"]


class Scorer(Protocol):
    """The minimal backend surface the detector needs."""

    def score(self, window: AudioWindow) -> ModelOutput:
        """Score one preprocessed audio window."""

    def metadata(self) -> object:
        """Return pinned backend metadata."""


def _merge_warnings(*groups: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    for group in groups:
        for warning in group:
            if warning not in merged:
                merged.append(warning)
    return tuple(merged)


def _resolve_config(*, config_name: str | None, model_id: str | None) -> BackendConfig:
    if config_name is not None and model_id is not None:
        config = get_backend_config(config_name)
        if config.model_id != model_id and config.model_key != model_id:
            raise ModelLoadError(f"Config {config_name!r} does not support model_id {model_id!r}")
        return config
    if config_name is not None:
        return get_backend_config(config_name)
    if model_id is not None:
        return get_config_for_model(model_id)
    return get_backend_config(DEFAULT_CONFIG_NAME)


def _resolve_calibration(config: BackendConfig, override: Calibration | None) -> Calibration:
    if override is not None:
        return override
    try:
        return get_calibration(config.model_id)
    except (KeyError, TypeError, ValueError, ModelLoadError) as exc:
        return calibration_fallback(reason=f"missing or invalid calibration: {exc}")


def _load_backend(config: BackendConfig, *, cache_dir: Path | None, device: str) -> Scorer:
    if config.backend == "fakeprint":
        return fakeprint_backend.load(
            weights_loader=loaders.load_fakeprint_weights, cache_dir=cache_dir, device=device
        )
    if config.backend == "spectttra":
        return spctttra_backend.load(
            config.model_key,
            classifier_loader=loaders.load_spctttra_classifier,
            cache_dir=cache_dir,
            device=device,
        )
    if config.backend == "ast":
        return ast_backend.load(
            config.model_key,
            pipeline_loader=loaders.load_ast_pipeline,
            cache_dir=cache_dir,
            device=device,
        )
    raise ModelLoadError(f"Unsupported backend {config.backend!r} in config {config.config_name!r}")


class Detector:
    """Single-backend detector with packaged temperature and tri-state thresholds."""

    def __init__(
        self,
        *,
        config: BackendConfig,
        calibration: Calibration,
        backend: Scorer,
    ) -> None:
        self.config = config
        self.calibration = calibration
        self._backend = backend
        self.model_id = config.model_id

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | None = None,
        *,
        config_name: str | None = None,
        backend: Scorer | str | None = None,
        calibration: Calibration | None = None,
        cache_dir: Path | None = None,
        device: str = "cpu",
    ) -> Detector:
        """Build a detector from packaged config and calibration, loading weights lazily."""

        config = _resolve_config(config_name=config_name, model_id=model_id)
        resolved_calibration = _resolve_calibration(config, calibration)
        if backend is None:
            scorer = _load_backend(config, cache_dir=cache_dir, device=device)
        elif isinstance(backend, str):
            raise ModelLoadError(
                "Select a backend with model_id or config_name; backend injection needs a scorer"
            )
        else:
            scorer = backend
        return cls(config=config, calibration=resolved_calibration, backend=scorer)

    def score_file(self, path: Path | str) -> Result:
        """Decode one configured window from a local audio file and score it."""

        window = decode_window(
            path,
            window_seconds=self.config.window_seconds,
            placement=self.config.placement,
            fill_mode=self.config.fill_mode,
        )
        return self._score_window(window)

    def score_bytes(self, data: bytes) -> Result:
        """Decode one configured window from in-memory audio bytes and score it."""

        if not data:
            raise ValueError("audio bytes must not be empty")
        window = decode_window_bytes(
            data,
            window_seconds=self.config.window_seconds,
            placement=self.config.placement,
            fill_mode=self.config.fill_mode,
        )
        return self._score_window(window)

    def _score_window(self, window: AudioWindow) -> Result:
        output = self._backend.score(window)
        probability = temperature_scale_scores(
            [output.raw_score], temperature=self.calibration.temperature
        )[0]
        label = label_for_probability(
            probability,
            low=self.calibration.low_threshold,
            high=self.calibration.high_threshold,
        )
        return Result(
            probability=probability,
            label=label,
            model_id=self.model_id,
            model_revision=output.model_revision,
            window=output.window,
            placement=self.config.placement,
            raw_score=output.raw_score,
            warnings=_merge_warnings(window.warnings, output.warnings, self.calibration.warnings),
            timings=output.timings,
        )

    @property
    def flag_threshold(self) -> float:
        """The dev-selected raw-score cutoff for this config."""

        return self.config.flag_threshold


class CascadeDetector:
    """Conservative two-stage detector: only stage one can auto-label ``ai_generated``."""

    def __init__(self, *, primary: Detector, second_opinion: Detector) -> None:
        self.primary = primary
        self.second_opinion = second_opinion

    @classmethod
    def from_pretrained(
        cls,
        *,
        primary: Scorer | str | None = None,
        second_opinion: Scorer | str | None = None,
        primary_config_name: str = DEFAULT_CONFIG_NAME,
        second_opinion_config_name: str = SECOND_OPINION_CONFIG_NAME,
        cache_dir: Path | None = None,
        device: str = "cpu",
    ) -> CascadeDetector:
        """Build the D1 primary and recommended second-opinion detectors."""

        primary_detector = Detector.from_pretrained(
            config_name=primary_config_name,
            backend=primary,
            cache_dir=cache_dir,
            device=device,
        )
        second_detector = Detector.from_pretrained(
            config_name=second_opinion_config_name,
            backend=second_opinion,
            cache_dir=cache_dir,
            device=device,
        )
        return cls(primary=primary_detector, second_opinion=second_detector)

    def score_file(self, path: Path | str) -> CascadeResult:
        """Run stage one on a local file, then stage two only when stage one did not flag."""

        primary_result = self.primary.score_file(path)
        if primary_result.raw_score >= self.primary.flag_threshold:
            return CascadeResult(
                label="ai_generated",
                primary=primary_result,
                second_opinion=None,
                warnings=primary_result.warnings,
            )
        second_result = self.second_opinion.score_file(path)
        return self._combine(primary_result, second_result)

    def score_bytes(self, data: bytes) -> CascadeResult:
        """Run the conservative cascade on in-memory audio bytes."""

        if not data:
            raise ValueError("audio bytes must not be empty")
        primary_result = self.primary.score_bytes(data)
        if primary_result.raw_score >= self.primary.flag_threshold:
            return CascadeResult(
                label="ai_generated",
                primary=primary_result,
                second_opinion=None,
                warnings=primary_result.warnings,
            )
        second_result = self.second_opinion.score_bytes(data)
        return self._combine(primary_result, second_result)

    def _combine(self, primary_result: Result, second_result: Result) -> CascadeResult:
        flagged = second_result.raw_score >= self.second_opinion.flag_threshold
        return CascadeResult(
            label="uncertain" if flagged else "human",
            primary=primary_result,
            second_opinion=second_result,
            warnings=_merge_warnings(primary_result.warnings, second_result.warnings),
        )
