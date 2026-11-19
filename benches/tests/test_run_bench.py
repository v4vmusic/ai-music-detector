from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from benches._shared.audio import decode_window
from benches._shared.constants import SAMPLE_RATE
from benches._shared.manifest import load_manifest
from benches._shared.registry import (
    AST_SPEC,
    MODEL_KEY_AST,
    MODEL_KEY_FAKEPRINT,
    MODEL_KEYS,
    AstSpec,
)
from benches._shared.types import AudioWindow, ModelMetadata, ModelOutput, Timings
from benches.run_bench import DEFAULT_OFFSETS, _load_ast_pipeline, main, run

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
    assert DEFAULT_OFFSETS[MODEL_KEY_AST] == 25.0
    assert all(
        offset == 25.0 for key, offset in DEFAULT_OFFSETS.items() if key != MODEL_KEY_FAKEPRINT
    )


def test_ast_loader_builds_the_training_feature_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This checkpoint has no preprocessor_config.json, so the loader must build one explicitly."""

    seen: dict[str, object] = {}
    fake_transformers = SimpleNamespace()

    class FakeConfig:
        id2label = {0: "human", 1: "ai_generated"}

    class FakeFeatureExtractor:
        def __init__(self, **kwargs: object) -> None:
            seen["feature_extractor"] = kwargs

    def fake_pipeline(task: str, **kwargs: object) -> object:
        seen["task"] = task
        seen["pipeline"] = kwargs
        return object()

    def fake_model_loader(model_id: str, **kwargs: object) -> object:
        seen["model_kwargs"] = kwargs
        if "map_location" in kwargs:
            raise TypeError("unexpected keyword argument 'map_location'")
        return object()

    fake_transformers.AutoConfig = SimpleNamespace(from_pretrained=lambda *a, **k: FakeConfig())
    fake_transformers.ASTFeatureExtractor = FakeFeatureExtractor
    fake_transformers.AutoModelForAudioClassification = SimpleNamespace(
        from_pretrained=fake_model_loader
    )
    fake_transformers.pipeline = fake_pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    loaded = _load_ast_pipeline(AST_SPEC, cache_dir=Path("/tmp/aidetect-cache"), device="cpu")

    options = cast(dict[str, object], seen["feature_extractor"])
    assert options["max_length"] == 6000
    assert options["num_mel_bins"] == 128
    assert options["sampling_rate"] == SAMPLE_RATE
    assert options["mean"] == -4.2677393
    assert options["std"] == 4.5689974
    assert seen["task"] == "audio-classification"
    assert "map_location" not in cast(dict[str, object], seen["model_kwargs"])
    assert loaded.id2label() == {0: "human", 1: "ai_generated"}


def test_main_dispatches_ast_with_the_pinned_revision(
    tmp_path: Path, write_wav: WriteWav, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = write_wav(tmp_path / "good.wav", 10.0)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,label\ngood.wav,human\n", encoding="utf-8")
    out = tmp_path / "ast.jsonl"

    class _Pipeline:
        def __call__(
            self, samples: NDArray[np.float32], *, sampling_rate: int
        ) -> Mapping[int, float]:
            return {0: 0.8, 1: 0.2}

        def id2label(self) -> Mapping[int, str]:
            return {0: "human", 1: "ai_generated"}

        def unload(self) -> None:
            return None

    calls: list[tuple[str, str, str]] = []

    def fake_loader(spec: AstSpec, *, cache_dir: Path | None, device: str) -> _Pipeline:
        calls.append((spec.model_id, spec.revision, device))
        return _Pipeline()

    monkeypatch.setattr("benches.run_bench._load_ast_pipeline", fake_loader)

    exit_code = main(
        [
            "--backend",
            MODEL_KEY_AST,
            "--manifest",
            str(manifest),
            "--out",
            str(out),
            "--seed",
            "42",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert calls == [(AST_SPEC.model_id, AST_SPEC.revision, "cpu")]
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["path"] == str(good.resolve())
    assert row["model_id"] == AST_SPEC.model_id
    assert row["model_revision"] == AST_SPEC.revision
    assert row["calibrated_probability"] == pytest.approx(0.2)
    assert row["window"] == "60s@16000Hz@25s"
    assert "short_track_padded" in row["warnings"]
    assert row["versions"]["transformers"]


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
