from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from aidetect.config import SECOND_OPINION_CONFIG_NAME, get_backend_config
from aidetect.constants import AI_GENERATED_LABEL, HUMAN_LABEL, UNCERTAIN_LABEL
from aidetect.detector import CascadeDetector
from aidetect.types import CascadeResult

from .conftest import FixedScorer, WriteWav


def _detector(primary_raw: float, second_raw: float) -> CascadeDetector:
    return CascadeDetector.from_pretrained(
        primary=FixedScorer(raw_score=primary_raw, probability=0.5),
        second_opinion=FixedScorer(raw_score=second_raw, probability=0.5, model_id="alpha"),
    )


def test_stage_one_high_confidence_flags_ai_and_skips_second_opinion(
    tmp_path: Path, write_wav: WriteWav
) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    detector = _detector(primary_raw=1.0, second_raw=100.0)

    result = detector.score_file(audio)

    assert isinstance(result, CascadeResult)
    assert result.label == AI_GENERATED_LABEL
    assert result.second_opinion is None
    assert result.primary.placement == "middle"


def test_stage_two_flag_routes_to_uncertain_not_ai(tmp_path: Path, write_wav: WriteWav) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    # Primary below its auto-AI cutoff; second opinion strongly flags AI.
    detector = _detector(primary_raw=0.0, second_raw=10.0)

    result = detector.score_file(audio)

    assert result.label == UNCERTAIN_LABEL
    assert result.second_opinion is not None
    assert result.second_opinion.placement == "middle"
    assert result.label != AI_GENERATED_LABEL


def test_neither_stage_flags_labels_human(tmp_path: Path, write_wav: WriteWav) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    detector = _detector(primary_raw=0.0, second_raw=-10.0)

    result = detector.score_file(audio)

    assert result.label == HUMAN_LABEL
    assert result.second_opinion is not None


def test_stage_label_disagreement_routes_to_uncertain(tmp_path: Path, write_wav: WriteWav) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    # Primary raw score is below the cascade's auto-AI cutoff, but its calibrated
    # probability is above the packaged high threshold, so its single-model label is
    # ai_generated. Stage two says human, so the techniques disagree and a human
    # reviewer is required; primary.label stays the single-model calibrated label.
    detector = _detector(primary_raw=1e-5, second_raw=-10.0)

    result = detector.score_file(audio)

    assert result.primary.label == AI_GENERATED_LABEL
    assert result.second_opinion is not None
    assert result.second_opinion.label == HUMAN_LABEL
    assert result.label == UNCERTAIN_LABEL


def test_stage_two_raw_cutoff_still_routes_to_uncertain(
    tmp_path: Path, write_wav: WriteWav
) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    # The raw cutoff, not the calibrated label, decides the stage-two flag.
    second_raw = get_backend_config(SECOND_OPINION_CONFIG_NAME).flag_threshold
    detector = _detector(primary_raw=0.0, second_raw=second_raw)

    result = detector.score_file(audio)

    assert result.second_opinion is not None
    assert result.second_opinion.label == HUMAN_LABEL
    assert result.label == UNCERTAIN_LABEL


def test_cascade_score_bytes_matches_score_file(tmp_path: Path, write_wav: WriteWav) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    from_file = _detector(0.0, 10.0).score_file(audio)
    from_bytes = _detector(0.0, 10.0).score_bytes(audio.read_bytes())
    assert from_file.label == from_bytes.label == UNCERTAIN_LABEL


def test_cascade_score_url_fetches_once_and_matches_bytes(
    tmp_path: Path, write_wav: WriteWav
) -> None:
    payload = write_wav(tmp_path / "ten.wav", 10.0).read_bytes()
    urls: list[str] = []

    def fetch(url: str) -> bytes:
        urls.append(url)
        return payload

    from_url = _detector(0.0, 10.0).score_url("https://example.com/track.mp3", fetcher=fetch)
    from_bytes = _detector(0.0, 10.0).score_bytes(payload)

    assert urls == ["https://example.com/track.mp3"]
    assert asdict(from_url) == asdict(from_bytes)
    assert from_url.label == UNCERTAIN_LABEL
