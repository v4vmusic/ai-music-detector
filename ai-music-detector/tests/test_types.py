from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
import pytest
from numpy.typing import NDArray

from aidetect.constants import AI_GENERATED_LABEL, WARNING_SHORT_TRACK_PADDED
from aidetect.errors import AudioDecodeError
from aidetect.types import AudioWindow, ModelMetadata, ModelOutput, Result, Timings


def _window(samples: NDArray[np.float32] | None = None) -> AudioWindow:
    return AudioWindow(samples=np.zeros(16_000, dtype=np.float32) if samples is None else samples)


def _output() -> ModelOutput:
    return ModelOutput(
        raw_score=-1.25,
        calibrated_probability=0.42,
        model_id="fakeprint",
        model_revision="abc123",
        window="3s@2s",
        warnings=(),
        timings=Timings(score_seconds=0.01),
    )


def test_audio_window_enforces_shared_contract() -> None:
    window = _window()
    assert window.sample_rate == 16_000
    assert window.samples.shape == (16_000,)
    assert window.samples.dtype == np.float32
    assert window.offset_seconds == 0.0


@pytest.mark.parametrize(
    ("samples", "rate", "message"),
    [
        (np.zeros(16, dtype=np.float32), 8_000, "16000"),
        (np.zeros((2, 16), dtype=np.float32), 16_000, "mono"),
        (np.zeros(16, dtype=np.float64), 16_000, "float32"),
        (np.array([0.0, 1.5], dtype=np.float32), 16_000, r"\[-1\.0, 1\.0\]"),
        (np.array([0.0, math.nan], dtype=np.float32), 16_000, "finite"),
    ],
)
def test_audio_window_rejects_invalid_windows(
    samples: NDArray[np.float32], rate: int, message: str
) -> None:
    with pytest.raises(AudioDecodeError, match=message):
        AudioWindow(samples=samples, sample_rate=rate)


def test_audio_window_keeps_short_track_warning() -> None:
    window = AudioWindow(
        samples=np.zeros(16, dtype=np.float32), warnings=(WARNING_SHORT_TRACK_PADDED,)
    )
    assert window.warnings == (WARNING_SHORT_TRACK_PADDED,)


def test_model_output_serialises_to_plain_types() -> None:
    payload = asdict(_output())
    assert payload["raw_score"] == -1.25
    assert payload["calibrated_probability"] == 0.42
    assert payload["timings"]["score_seconds"] == 0.01


@pytest.mark.parametrize("probability", [1.01, math.nan, -0.1])
def test_model_output_rejects_invalid_probability(probability: float) -> None:
    with pytest.raises(ValueError, match="calibrated_probability"):
        ModelOutput(
            raw_score=0.0,
            calibrated_probability=probability,
            model_id="fakeprint",
            model_revision="abc123",
            window="3s@2s",
            warnings=(),
            timings=Timings(score_seconds=0.01),
        )


def test_model_output_allows_missing_calibration() -> None:
    output = ModelOutput(
        raw_score=0.0,
        calibrated_probability=None,
        model_id="fakeprint",
        model_revision="abc123",
        window="3s@2s",
        warnings=(),
        timings=Timings(score_seconds=0.01),
    )
    assert output.calibrated_probability is None


def test_timings_rejects_negative_seconds() -> None:
    with pytest.raises(ValueError, match="score_seconds"):
        Timings(score_seconds=-0.1)


def test_model_metadata_defaults() -> None:
    metadata = ModelMetadata(model_id="fakeprint", revision="abc123", window_seconds=3.0)
    assert metadata.sample_rate == 16_000
    assert metadata.preprocessing_version == "1"


def test_result_validates_probability_and_label() -> None:
    result = Result(
        probability=0.5,
        label=AI_GENERATED_LABEL,
        model_id="fakeprint",
        model_revision="abc123",
        window="10s@16000Hz",
        placement="middle",
        raw_score=0.5,
        warnings=(),
        timings=Timings(score_seconds=0.01),
    )
    assert result.probability == 0.5
    with pytest.raises(ValueError, match="probability"):
        Result(
            probability=1.5,
            label=AI_GENERATED_LABEL,
            model_id="fakeprint",
            model_revision="abc123",
            window="10s@16000Hz",
            placement="middle",
            raw_score=0.5,
            warnings=(),
            timings=Timings(score_seconds=0.01),
        )
