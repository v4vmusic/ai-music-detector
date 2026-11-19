from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from aidetect.calibration import calibration_fallback
from aidetect.config import DEFAULT_CONFIG_NAME
from aidetect.constants import AI_GENERATED_LABEL, HUMAN_LABEL, UNCERTAIN_LABEL
from aidetect.detector import Detector
from aidetect.errors import ModelLoadError
from aidetect.types import Result

from .conftest import FixedScorer, WriteWav


def test_detector_default_is_d1_fakeprint_middle_window() -> None:
    detector = Detector.from_pretrained(backend=FixedScorer(raw_score=0.9, probability=0.9))
    assert detector.model_id == "lofcz/ai-music-detector"
    assert detector.config.config_name == DEFAULT_CONFIG_NAME
    assert detector.config.window_seconds == 10.0
    assert detector.config.placement == "middle"


def test_score_file_returns_full_result_shape(tmp_path: Path, write_wav: WriteWav) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    detector = Detector.from_pretrained(backend=FixedScorer(raw_score=0.5, probability=0.5))
    result = detector.score_file(audio)

    assert isinstance(result, Result)
    assert 0.0 <= result.probability <= 1.0
    assert result.label in (HUMAN_LABEL, UNCERTAIN_LABEL, AI_GENERATED_LABEL)
    assert result.model_id == "lofcz/ai-music-detector"
    assert result.model_revision == "test-revision"
    assert result.window == "10s@16000Hz"
    assert result.placement == "middle"
    assert isinstance(result.warnings, tuple)
    assert result.timings.score_seconds >= 0.0


def test_score_bytes_matches_score_file(tmp_path: Path, write_wav: WriteWav) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    backend = FixedScorer(raw_score=0.2, probability=0.2)
    from_file = Detector.from_pretrained(backend=backend).score_file(audio)

    backend_bytes = FixedScorer(raw_score=0.2, probability=0.2)
    from_bytes = Detector.from_pretrained(backend=backend_bytes).score_bytes(audio.read_bytes())

    assert from_file.probability == pytest.approx(from_bytes.probability)
    assert from_file.label == from_bytes.label
    assert backend.calls[0][0] == backend_bytes.calls[0][0]


def test_threshold_boundaries_are_uncertain(tmp_path: Path, write_wav: WriteWav) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    calibration = calibration_fallback(reason="test")
    low, high = calibration.low_threshold, calibration.high_threshold

    expected = {
        low - 0.01: HUMAN_LABEL,
        low: UNCERTAIN_LABEL,
        (low + high) / 2: UNCERTAIN_LABEL,
        high: UNCERTAIN_LABEL,
        high + 0.01: AI_GENERATED_LABEL,
    }
    for probability, label in expected.items():
        detector = Detector.from_pretrained(
            backend=FixedScorer(raw_score=probability, probability=probability),
            calibration=calibration,
        )
        assert detector.score_file(audio).label == label


def test_packaged_calibration_is_applied_without_dev_data(
    tmp_path: Path, write_wav: WriteWav
) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    # Raw 0.5 becomes ~0.5 under temperature scaling; verify the transform is applied
    # by using a backend whose calibrated_probability deliberately disagrees.
    detector = Detector.from_pretrained(
        backend=FixedScorer(raw_score=0.02, probability=0.99),
    )
    result = detector.score_file(audio)
    assert result.probability != pytest.approx(0.99)
    assert 0.0 <= result.probability <= 1.0


def test_degenerate_calibration_uses_fallback_with_warning(
    tmp_path: Path, write_wav: WriteWav
) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    detector = Detector.from_pretrained(
        backend=FixedScorer(raw_score=0.5, probability=0.5),
        calibration=calibration_fallback(reason="degenerate dev scores"),
    )
    result = detector.score_file(audio)
    assert any("calibration_fallback" in warning for warning in result.warnings)


def test_unsupported_model_id_raises_actionable_error() -> None:
    with pytest.raises(ModelLoadError, match="model_id"):
        Detector.from_pretrained(model_id="not-a-model", backend=FixedScorer(0.5, 0.5))


def test_score_bytes_rejects_empty_input() -> None:
    detector = Detector.from_pretrained(backend=FixedScorer(0.5, 0.5))
    with pytest.raises(ValueError, match="empty"):
        detector.score_bytes(b"")


def test_backend_names_are_selected_by_config_not_a_string_argument() -> None:
    with pytest.raises(ModelLoadError, match="backend"):
        Detector.from_pretrained(backend="ast")


def test_score_url_matches_score_bytes_with_injected_fetcher(
    tmp_path: Path, write_wav: WriteWav
) -> None:
    payload = write_wav(tmp_path / "ten.wav", 10.0).read_bytes()
    urls: list[str] = []

    def fetch(url: str) -> bytes:
        urls.append(url)
        return payload

    from_url = Detector.from_pretrained(backend=FixedScorer(0.2, 0.2)).score_url(
        "https://example.com/track.mp3", fetcher=fetch
    )
    from_bytes = Detector.from_pretrained(backend=FixedScorer(0.2, 0.2)).score_bytes(payload)

    assert urls == ["https://example.com/track.mp3"]
    assert asdict(from_url) == asdict(from_bytes)
