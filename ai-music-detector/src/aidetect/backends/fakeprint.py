from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from aidetect.errors import ModelLoadError
from aidetect.probability import as_probability
from aidetect.registry import FAKEPRINT_SPEC, ModelSpec
from aidetect.types import (
    AudioWindow,
    ModelMetadata,
    ModelOutput,
    Timings,
    validate_audio_window,
)

__all__ = ["FakeprintBackend", "OnnxSession", "extract_fakeprint", "load"]

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


class WeightsLoader(Protocol):
    """Fetches pinned model bytes (runner or test fake)."""

    def __call__(self, spec: ModelSpec, *, cache_dir: Path | None, device: str) -> bytes: ...


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
        0.0, FAKEPRINT_SPEC.sample_rate / 2.0, num=n_fft // 2 + 1, dtype=np.float32
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
        probability = as_probability(float(np.asarray(result[0]).reshape(-1)[0]))

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
    weights_loader: WeightsLoader | None = None,
    cache_dir: Path | None = None,
    device: str = "cpu",
) -> FakeprintBackend:
    """Build a backend from a session, model bytes or an injected weights loader."""

    if session is None:
        if model_bytes is None:
            if weights_loader is None:
                raise ModelLoadError(
                    "Fakeprint needs an ONNX session, model_bytes or an injected weights loader"
                )
            model_bytes = weights_loader(FAKEPRINT_SPEC, cache_dir=cache_dir, device=device)
        # Import here so plain `import aidetect` never loads ONNX runtime (CC §7).
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]  # no published stubs/py.typed
        except ImportError as exc:
            raise ModelLoadError(
                "Fakeprint needs onnxruntime; install the `onnx` or `full` extra"
            ) from exc

        session = cast(
            OnnxSession,
            ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"]),
        )

    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].name != INPUT_NAME:
        names = [declared.name for declared in inputs]
        raise ModelLoadError(f"Expected one ONNX input named {INPUT_NAME!r}, got {names!r}")
    return FakeprintBackend(session=session)
