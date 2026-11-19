from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
import pytest
from numpy.typing import NDArray

from benches._shared.constants import WARNING_SHORT_TRACK_PADDED
from benches._shared.errors import AudioDecodeError
from benches._shared.types import AudioWindow, ModelMetadata, ModelOutput, Timings


def _window(samples: NDArray[np.float32] | None = None) -> AudioWindow:
    return AudioWindow(samples=np.zeros(16_000, dtype=np.float32) if samples is None else samples)


def _output(
    *,
    raw_score: float = -1.25,
    calibrated_probability: float | None = 0.42,
    model_id: str = "fakeprint",
    model_revision: str = "abc123",
    window: str = "3s@2s",
    warnings: tuple[str, ...] = (),
    timings: Timings | None = None,
) -> ModelOutput:
    return ModelOutput(
        raw_score=raw_score,
        calibrated_probability=calibrated_probability,
        model_id=model_id,
        model_revision=model_revision,
        window=window,
        warnings=warnings,
        timings=Timings(score_seconds=0.01) if timings is None else timings,
    )


def test_audio_window_accepts_valid_window() -> None:
    window = _window()
    assert window.sample_rate == 16_000
    assert window.samples.shape == (16_000,)
    assert window.samples.dtype == np.float32
    assert window.offset_seconds == 0.0
    assert window.warnings == ()


def test_audio_window_rejects_wrong_sample_rate() -> None:
    with pytest.raises(AudioDecodeError, match="16000"):
        AudioWindow(samples=np.zeros(16, dtype=np.float32), sample_rate=8_000)


def test_audio_window_rejects_stereo() -> None:
    with pytest.raises(AudioDecodeError, match="mono"):
        _window(np.zeros((2, 16), dtype=np.float32))


def test_audio_window_rejects_non_float32() -> None:
    with pytest.raises(AudioDecodeError, match="float32"):
        _window(np.zeros(16, dtype=np.float64))


def test_audio_window_rejects_out_of_range_values() -> None:
    with pytest.raises(AudioDecodeError, match=r"\[-1\.0, 1\.0\]"):
        _window(np.array([0.0, 1.5], dtype=np.float32))


def test_audio_window_rejects_non_finite_values() -> None:
    with pytest.raises(AudioDecodeError, match="finite"):
        _window(np.array([0.0, math.nan], dtype=np.float32))


def test_audio_window_keeps_short_track_warning() -> None:
    window = AudioWindow(
        samples=np.zeros(16, dtype=np.float32),
        warnings=(WARNING_SHORT_TRACK_PADDED,),
    )
    assert window.warnings == (WARNING_SHORT_TRACK_PADDED,)


def test_model_output_serialises_to_plain_types() -> None:
    payload = asdict(_output())
    assert payload["raw_score"] == -1.25
    assert payload["calibrated_probability"] == 0.42
    assert payload["model_id"] == "fakeprint"
    assert payload["model_revision"] == "abc123"
    assert payload["timings"]["score_seconds"] == 0.01


def test_model_output_rejects_probability_above_one() -> None:
    with pytest.raises(ValueError, match="calibrated_probability"):
        _output(calibrated_probability=1.01)


def test_model_output_rejects_nan_probability() -> None:
    with pytest.raises(ValueError, match="calibrated_probability"):
        _output(calibrated_probability=math.nan)


def test_model_output_allows_missing_calibration() -> None:
    assert _output(calibrated_probability=None).calibrated_probability is None


def test_model_output_rejects_non_finite_raw_score() -> None:
    with pytest.raises(ValueError, match="raw_score"):
        _output(raw_score=math.inf)


def test_model_output_rejects_empty_model_id() -> None:
    with pytest.raises(ValueError, match="model_id"):
        _output(model_id="")


def test_model_output_rejects_missing_required_fields() -> None:
    # A dataclass raises TypeError at construction when a required field is omitted.
    with pytest.raises(TypeError):
        ModelOutput(raw_score=0.0)  # type: ignore[call-arg]


def test_timings_rejects_negative_seconds() -> None:
    with pytest.raises(ValueError, match="score_seconds"):
        Timings(score_seconds=-0.1)


def test_model_metadata_defaults() -> None:
    metadata = ModelMetadata(model_id="fakeprint", revision="abc123", window_seconds=3.0)
    assert metadata.sample_rate == 16_000
    assert metadata.preprocessing_version == "1"


def test_model_metadata_requires_window_seconds() -> None:
    # A dataclass raises TypeError at construction when a required field is omitted.
    with pytest.raises(TypeError):
        ModelMetadata(model_id="fakeprint", revision="abc123")  # type: ignore[call-arg]
