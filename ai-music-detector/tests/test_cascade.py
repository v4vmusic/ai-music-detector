from __future__ import annotations

from pathlib import Path

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


def test_cascade_score_bytes_matches_score_file(tmp_path: Path, write_wav: WriteWav) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    from_file = _detector(0.0, 10.0).score_file(audio)
    from_bytes = _detector(0.0, 10.0).score_bytes(audio.read_bytes())
    assert from_file.label == from_bytes.label == UNCERTAIN_LABEL
