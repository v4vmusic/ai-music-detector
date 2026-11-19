from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from benches.decode import main

WriteWav = Callable[[Path, float], Path]


def test_decode_cli_prints_shape_and_duration(
    tmp_path: Path,
    write_wav: WriteWav,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    code = main(["--path", str(audio), "--offset", "2", "--window", "3"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["shape"] == [48_000]
    assert payload["sample_rate"] == 16_000
    assert payload["duration_seconds"] == 3.0
    assert payload["warnings"] == []


def test_decode_cli_reports_decode_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    code = main(["--path", str(bad), "--window", "3"])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err
