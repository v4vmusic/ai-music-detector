from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from aidetect.cli import main
from aidetect.config import SECOND_OPINION_CONFIG_NAME
from aidetect.constants import AI_GENERATED_LABEL, HUMAN_LABEL, UNCERTAIN_LABEL

from .conftest import FixedScorer, WriteWav


def _install_backend(
    monkeypatch: pytest.MonkeyPatch, scorer_factory: Callable[[str], FixedScorer]
) -> None:
    def load(config: object, *, cache_dir: Path | None, device: str) -> FixedScorer:
        del cache_dir, device
        return scorer_factory(str(getattr(config, "config_name", "")))

    monkeypatch.setattr("aidetect.detector._load_backend", load)


def test_cli_score_json_writes_one_object_and_diagnostics_to_stderr(
    tmp_path: Path,
    write_wav: WriteWav,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio = write_wav(tmp_path / "six.wav", 6.0)
    _install_backend(monkeypatch, lambda name: FixedScorer(-15.0, 0.0))

    code = main(["score", str(audio), "--json"])
    captured = capsys.readouterr()

    assert code == 0
    payload = json.loads(captured.out)
    assert 0.0 <= payload["probability"] <= 1.0
    assert payload["label"] == HUMAN_LABEL
    assert payload["placement"] == "middle"
    assert captured.err.strip()
    assert "short_track_padded" in captured.err
    assert captured.out.count("\n") == 1


def test_cli_reports_errors_on_stderr_with_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.wav"
    _install_backend(monkeypatch, lambda name: FixedScorer(0.1, 0.1))
    code = main(["score", str(missing), "--json"])
    captured = capsys.readouterr()

    assert code != 0
    assert captured.out == ""
    assert "missing.wav" in captured.err


def test_cli_dispatches_urls_through_the_safe_fetcher(
    tmp_path: Path,
    write_wav: WriteWav,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = write_wav(tmp_path / "ten.wav", 10.0).read_bytes()
    urls: list[str] = []

    def fetch(url: str) -> bytes:
        urls.append(url)
        return payload

    _install_backend(monkeypatch, lambda name: FixedScorer(0.1, 0.1))
    monkeypatch.setattr("aidetect.detector.fetch_audio_bytes", fetch)

    code = main(["score", "https://example.com/track.mp3", "--json"])
    captured = capsys.readouterr()

    assert code == 0
    assert urls == ["https://example.com/track.mp3"]
    assert 0.0 <= json.loads(captured.out)["probability"] <= 1.0


def test_cli_cascade_json_keeps_stages_separate(
    tmp_path: Path,
    write_wav: WriteWav,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)

    def scorer(name: str) -> FixedScorer:
        if name == SECOND_OPINION_CONFIG_NAME:
            return FixedScorer(10.0, 0.9, model_id="alpha")
        return FixedScorer(0.0, 0.1)

    _install_backend(monkeypatch, scorer)
    code = main(["score", str(audio), "--cascade", "--json"])
    captured = capsys.readouterr()

    assert code == 0
    payload = json.loads(captured.out)
    assert payload["label"] == UNCERTAIN_LABEL
    assert payload["primary"]["label"] != AI_GENERATED_LABEL
    assert 0.0 <= payload["second_opinion"]["probability"] <= 1.0
    assert "probability" not in payload


def test_cli_plain_output_still_reports_label(
    tmp_path: Path,
    write_wav: WriteWav,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    _install_backend(monkeypatch, lambda name: FixedScorer(-15.0, 0.0))

    code = main(["score", str(audio)])
    captured = capsys.readouterr()

    assert code == 0
    assert HUMAN_LABEL in captured.out
    assert "probability" in captured.out


def test_cli_root_help_lists_command_flags_and_examples(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "score" in out
    assert "--json" in out
    assert "--cascade" in out
    assert "https://example.com/track.mp3" in out
    assert "--cascade requires the 'full' or 'gpu' install" in out


def test_cli_root_help_explains_usage_ordering(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage order" in out
    assert "aidetect score <source> [flags]" in out
    assert "must come after `score`" in out
    assert "aidetect --json score track.mp3" in out
    assert "# rejected" in out


def test_cli_rejects_flags_before_the_score_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--json", "score", "track.mp3"])

    assert exc.value.code == 2
    assert "unrecognized arguments: --json" in capsys.readouterr().err


def test_cli_score_help_lists_flags_and_examples(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["score", "--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--json" in out
    assert "--cascade" in out
    assert "source" in out
    assert "https://example.com/track.mp3" in out
    assert "stderr" in out
    assert "--cascade requires the 'full' or 'gpu' install" in out
