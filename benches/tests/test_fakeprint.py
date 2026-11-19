from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from benches._shared.constants import SAMPLE_RATE
from benches._shared.errors import AudioDecodeError, ModelLoadError
from benches._shared.registry import FAKEPRINT_SPEC
from benches._shared.types import AudioWindow
from benches.backends.fakeprint import extract_fakeprint, load


class _FakeOnnxInput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    """ONNX-session double: no weights, no network, fixed output."""

    def __init__(self, output: float = 0.73, input_name: str = "fakeprint") -> None:
        self.output = output
        self.input_name = input_name
        self.input_feed: Mapping[str, NDArray[np.float32]] = {}

    def get_inputs(self) -> list[_FakeOnnxInput]:
        return [_FakeOnnxInput(self.input_name)]

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, NDArray[np.float32]],
    ) -> list[NDArray[np.float32]]:
        self.input_feed = input_feed
        return [np.asarray([[self.output]], dtype=np.float32)]


def _tone(frequency: float = 2_000.0, seconds: float = 10.0) -> NDArray[np.float32]:
    count = int(round(seconds * SAMPLE_RATE))
    return cast(
        NDArray[np.float32],
        (0.5 * np.sin(2.0 * math.pi * frequency * np.arange(count) / SAMPLE_RATE)).astype(
            np.float32
        ),
    )


def test_extract_fakeprint_is_deterministic_and_bounded() -> None:
    features = extract_fakeprint(_tone())
    again = extract_fakeprint(_tone())

    assert features.shape == (FAKEPRINT_SPEC.feature_count,)
    assert features.dtype == np.float32
    assert bool(np.all(np.isfinite(features)))
    assert float(features.min()) >= 0.0
    assert float(features.max()) <= 1.0
    np.testing.assert_array_equal(features, again)


def test_silence_and_in_band_tone_have_different_valid_features() -> None:
    silence = np.zeros(int(10.0 * SAMPLE_RATE), dtype=np.float32)
    silent_features = extract_fakeprint(silence)
    tone_features = extract_fakeprint(_tone())

    assert not np.array_equal(silent_features, tone_features)
    assert float(tone_features.max()) > 0.0
    for features in (silent_features, tone_features):
        assert features.shape == (FAKEPRINT_SPEC.feature_count,)
        assert features.dtype == np.float32
        assert bool(np.all(np.isfinite(features)))
        assert 0.0 <= float(features.min()) <= float(features.max()) <= 1.0


def test_fake_session_probability_is_passed_through_unchanged() -> None:
    session = _FakeSession(output=0.73)
    backend = load(session=session)
    window = AudioWindow(samples=_tone(), window_seconds=10.0)

    result = backend.score(window)

    assert result.raw_score == pytest.approx(0.73)
    assert result.calibrated_probability == pytest.approx(0.73)
    assert result.model_id == FAKEPRINT_SPEC.model_id
    assert result.model_revision == FAKEPRINT_SPEC.revision
    assert result.window == "10s@16000Hz"
    assert session.input_feed["fakeprint"].shape == (1, FAKEPRINT_SPEC.feature_count)

    metadata = backend.metadata()
    assert metadata.model_id == FAKEPRINT_SPEC.model_id
    assert metadata.revision == FAKEPRINT_SPEC.revision
    assert metadata.window_seconds == FAKEPRINT_SPEC.window_seconds
    assert metadata.sample_rate == SAMPLE_RATE


def test_load_rejects_a_session_with_the_wrong_input_name() -> None:
    with pytest.raises(ModelLoadError, match="fakeprint"):
        load(session=_FakeSession(input_name="audio"))


@pytest.mark.parametrize(
    "bad_samples,bad_rate",
    [
        (_tone(), 8_000),
        (np.stack((_tone(), _tone())), SAMPLE_RATE),
    ],
)
def test_score_rejects_wrong_sample_rate_or_stereo_with_shared_error(
    bad_samples: NDArray[np.float32],
    bad_rate: int,
) -> None:
    # Bypass the dataclass constructor to prove the backend also guards its input.
    bad_window = cast(
        AudioWindow,
        SimpleNamespace(samples=bad_samples, sample_rate=bad_rate, warnings=()),
    )

    with pytest.raises(AudioDecodeError):
        load(session=_FakeSession()).score(bad_window)
