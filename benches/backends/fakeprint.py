from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, cast

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]  # no published stubs or py.typed
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from benches._shared.errors import ModelLoadError
from benches._shared.registry import FAKEPRINT_SPEC
from benches._shared.types import (
    AudioWindow,
    ModelMetadata,
    ModelOutput,
    Timings,
    validate_audio_window,
)

__all__ = ["FakeprintBackend", "extract_fakeprint", "load"]

INPUT_NAME = "fakeprint"


class _SessionInput(Protocol):
    name: str


class OnnxSession(Protocol):
    """The narrow ONNX surface the backend needs; real and fake sessions both fit."""

    def get_inputs(self) -> Sequence[_SessionInput]:
        """Return the model's declared inputs."""

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, NDArray[np.float32]],
    ) -> Sequence[NDArray[np.float32]]:
        """Run inference on the supplied feature batch."""


def _lower_hull(values: NDArray[np.float32], size: int) -> NDArray[np.float32]:
    """Minimum filter with ``nearest`` edge handling, matching the reference extractor."""

    left = size // 2
    right = size - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return cast(NDArray[np.float32], np.min(sliding_window_view(padded, size), axis=-1))


def extract_fakeprint(samples: NDArray[np.float32]) -> NDArray[np.float32]:
    """Turn one 16 kHz mono window into the model's 3585-value fakeprint.

    Mirrors the released model's reference preprocessing: Hann-windowed power spectrum
    with 50% overlap, dB conversion, time averaging, the 1-8 kHz band, lower-envelope
    subtraction and min-max normalisation. The ONNX model applies its own sigmoid.
    """

    spec = FAKEPRINT_SPEC
    n_fft = spec.n_fft
    hop = n_fft // 2
    window = (
        0.5 - 0.5 * np.cos(2.0 * math.pi * np.arange(n_fft, dtype=np.float32) / n_fft)
    ).astype(np.float32)

    padded = np.pad(samples, (n_fft // 2, n_fft // 2), mode="reflect")
    frames = sliding_window_view(padded, n_fft)[::hop]
    spectrum = np.fft.rfft(frames * window, n=n_fft, axis=-1)
    power_db = (10.0 * np.log10(np.clip(np.abs(spectrum) ** 2, 1e-10, 1e6))).astype(np.float32)
    mean_db = power_db.mean(axis=0)

    freq_bins = np.linspace(
        0.0,
        FAKEPRINT_SPEC.sample_rate / 2.0,
        num=n_fft // 2 + 1,
        dtype=np.float32,
    )
    band = mean_db[(freq_bins >= spec.freq_min_hz) & (freq_bins <= spec.freq_max_hz)]
    hull = np.clip(_lower_hull(band, spec.hull_area_bins), spec.min_db, None)
    residue = np.clip(band - hull, 0.0, spec.max_db)
    return cast(NDArray[np.float32], (residue / (residue.max() + 1e-6)).astype(np.float32))


@dataclass(frozen=True, slots=True)
class FakeprintBackend:
    """Scores shared audio windows with an injected ONNX session (no I/O here)."""

    session: OnnxSession

    def score(self, window: AudioWindow) -> ModelOutput:
        validate_audio_window(window)
        started = perf_counter()
        features = extract_fakeprint(window.samples)
        result = self.session.run(None, {INPUT_NAME: features.reshape(1, -1)})
        probability = float(np.asarray(result[0]).reshape(-1)[0])

        return ModelOutput(
            raw_score=probability,
            calibrated_probability=probability,
            model_id=FAKEPRINT_SPEC.model_id,
            model_revision=FAKEPRINT_SPEC.revision,
            window=f"{FAKEPRINT_SPEC.window_seconds:g}s@{FAKEPRINT_SPEC.sample_rate}Hz",
            warnings=window.warnings,
            timings=Timings(score_seconds=perf_counter() - started),
        )

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=FAKEPRINT_SPEC.model_id,
            revision=FAKEPRINT_SPEC.revision,
            window_seconds=FAKEPRINT_SPEC.window_seconds,
            sample_rate=FAKEPRINT_SPEC.sample_rate,
            preprocessing_version=FAKEPRINT_SPEC.preprocessing_version,
        )


def load(
    session: OnnxSession | None = None,
    *,
    model_bytes: bytes | None = None,
) -> FakeprintBackend:
    """Build a backend from a session (tests) or serialized weights (runner)."""

    if session is None:
        if model_bytes is None:
            raise ValueError("load() needs either session or model_bytes")
        session = cast(
            OnnxSession,
            ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"]),
        )

    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].name != INPUT_NAME:
        names = [declared.name for declared in inputs]
        raise ModelLoadError(f"Expected one ONNX input named {INPUT_NAME!r}, got {names!r}")
    return FakeprintBackend(session=session)
