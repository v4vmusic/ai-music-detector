from __future__ import annotations

import json
import subprocess
import sys
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


def test_decode_script_runs_standalone_from_workspace_root(
    tmp_path: Path,
    write_wav: WriteWav,
) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "benches" / "decode.py"),
            "--path",
            str(audio),
            "--window",
            "3",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["shape"] == [48_000]
