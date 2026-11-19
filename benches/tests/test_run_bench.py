from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from benches._shared.audio import decode_window
from benches._shared.constants import SAMPLE_RATE
from benches._shared.manifest import load_manifest
from benches._shared.registry import MODEL_KEY_FAKEPRINT, MODEL_KEYS
from benches._shared.types import AudioWindow, ModelMetadata, ModelOutput, Timings
from benches.run_bench import DEFAULT_OFFSETS, run

WriteWav = Callable[[Path, float], Path]


class _StubBackend:
    def score(self, window: AudioWindow) -> ModelOutput:
        return ModelOutput(
            raw_score=0.42,
            calibrated_probability=0.42,
            model_id="stub/model",
            model_revision="stubrevision",
            window="10s@16000Hz",
            warnings=window.warnings,
            timings=Timings(score_seconds=0.001),
        )

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id="stub/model",
            revision="stubrevision",
            window_seconds=10.0,
            sample_rate=SAMPLE_RATE,
        )


def test_run_writes_one_row_per_entry_and_continues_after_decode_failure(
    tmp_path: Path,
    write_wav: WriteWav,
) -> None:
    good = write_wav(tmp_path / "good.wav", 10.0)
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,label\ngood.wav,human\nbad.wav,ai_generated\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"

    count = run(load_manifest(manifest), out, _StubBackend(), seed=7)

    assert count == 2
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    scored, failed = (json.loads(line) for line in lines)

    assert scored["path"] == str(good.resolve())
    assert scored["label"] == "human"
    assert scored["raw_score"] == 0.42
    assert scored["calibrated_probability"] == 0.42
    assert scored["model_id"] == "stub/model"
    assert scored["model_revision"] == "stubrevision"
    assert scored["seed"] == 7
    assert scored["config_hash"]
    assert scored["versions"]["numpy"]
    assert scored["timings"]["score_seconds"] >= 0.0

    assert failed["path"] == str(bad.resolve())
    assert failed["label"] == "ai_generated"
    assert failed["warnings"][0].startswith("decode_failed")
    assert failed["error"]
    assert "raw_score" not in failed
    assert failed["seed"] == 7
    assert failed["config_hash"] == scored["config_hash"]
    assert failed["versions"] == scored["versions"]


def test_run_bench_script_runs_standalone_from_workspace_root() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "benches" / "run_bench.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--backend" in result.stdout
    assert "--manifest" in result.stdout
    assert "--offset" in result.stdout


def test_default_offsets_pin_all_backends() -> None:
    assert set(DEFAULT_OFFSETS) == set(MODEL_KEYS)
    assert DEFAULT_OFFSETS[MODEL_KEY_FAKEPRINT] == 0.0
    assert all(
        offset == 25.0 for key, offset in DEFAULT_OFFSETS.items() if key != MODEL_KEY_FAKEPRINT
    )


def test_run_passes_window_duration_and_offset_to_decoder(
    tmp_path: Path,
    write_wav: WriteWav,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good = write_wav(tmp_path / "good.wav", 10.0)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,label\ngood.wav,human\n", encoding="utf-8")
    calls: list[tuple[Path, float, float]] = []
    real_decode = decode_window

    def recording_decode(
        path: Path | str, *, window_seconds: float, offset_seconds: float = 0.0
    ) -> AudioWindow:
        calls.append((Path(path), window_seconds, offset_seconds))
        return real_decode(path, window_seconds=window_seconds, offset_seconds=offset_seconds)

    monkeypatch.setattr("benches.run_bench.decode_window", recording_decode)

    count = run(
        load_manifest(manifest),
        tmp_path / "out.jsonl",
        _StubBackend(),
        seed=7,
        offset_seconds=25.0,
    )

    assert count == 1
    assert calls == [(good.resolve(), 10.0, 25.0)]
