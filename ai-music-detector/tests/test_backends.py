from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from aidetect.backends import loaders
from aidetect.backends.ast import ai_probability
from aidetect.backends.ast import load as load_ast
from aidetect.backends.fakeprint import extract_fakeprint
from aidetect.backends.fakeprint import load as load_fakeprint
from aidetect.backends.spctttra import load as load_spctttra
from aidetect.constants import AI_GENERATED_LABEL, SAMPLE_RATE
from aidetect.errors import AudioDecodeError, ModelLoadError
from aidetect.registry import (
    AST_SPEC,
    FAKEPRINT_SPEC,
    MODEL_KEY_AST,
    MODEL_KEY_SPECTTRA_ALPHA,
    MODEL_KEY_SPECTTRA_BETA,
    SPECTTRA_SPECS,
)
from aidetect.types import AudioWindow


class _FakeOnnxInput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
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


class _FakeClassifier:
    def __init__(self, logit: float = 1.25) -> None:
        self.logit = logit
        self.samples: NDArray[np.float32] | None = None
        self.unloaded = False

    def __call__(self, samples: NDArray[np.float32]) -> float:
        self.samples = samples
        return self.logit

    def unload(self) -> None:
        self.unloaded = True


class _FakePipeline:
    def __init__(
        self,
        scores: Mapping[int, float] | None = None,
        id2label: Mapping[int, str] | None = None,
    ) -> None:
        self.scores = scores if scores is not None else {0: 0.9, 1: 0.1}
        self.labels = id2label if id2label is not None else {0: "human", 1: "ai_generated"}
        self.unloaded = False

    def __call__(self, samples: NDArray[np.float32], *, sampling_rate: int) -> Mapping[int, float]:
        return self.scores

    def id2label(self) -> Mapping[int, str]:
        return self.labels

    def unload(self) -> None:
        self.unloaded = True


def _tone(seconds: float = 10.0) -> NDArray[np.float32]:
    count = int(round(seconds * SAMPLE_RATE))
    return cast(
        NDArray[np.float32],
        (0.5 * np.sin(2.0 * math.pi * 2_000.0 * np.arange(count) / SAMPLE_RATE)).astype(np.float32),
    )


def _window(seconds: float, offset: float = 0.0) -> AudioWindow:
    count = int(round(seconds * SAMPLE_RATE))
    return AudioWindow(
        samples=np.full(count, 0.1, dtype=np.float32),
        offset_seconds=offset,
        window_seconds=seconds,
    )


def test_extract_fakeprint_is_deterministic_and_bounded() -> None:
    features = extract_fakeprint(_tone())
    assert features.shape == (FAKEPRINT_SPEC.feature_count,)
    assert features.dtype == np.float32
    assert 0.0 <= float(features.min()) <= float(features.max()) <= 1.0
    np.testing.assert_array_equal(features, extract_fakeprint(_tone()))


def test_fakeprint_session_probability_is_passed_through_unchanged() -> None:
    session = _FakeSession(output=0.73)
    result = load_fakeprint(session=session).score(_window(10.0))
    assert result.raw_score == pytest.approx(0.73)
    assert result.calibrated_probability == pytest.approx(0.73)
    assert result.model_id == FAKEPRINT_SPEC.model_id
    assert session.input_feed["fakeprint"].shape == (1, FAKEPRINT_SPEC.feature_count)


def test_fakeprint_rejects_bad_session_and_probability() -> None:
    with pytest.raises(ModelLoadError, match="fakeprint"):
        load_fakeprint(session=_FakeSession(input_name="audio"))
    with pytest.raises(ModelLoadError, match="probability"):
        load_fakeprint(session=_FakeSession(output=1.4)).score(_window(10.0))


def test_spctttra_logit_maps_to_probability_and_rejects_bad_windows() -> None:
    fake = _FakeClassifier(logit=2.0)
    backend = load_spctttra(MODEL_KEY_SPECTTRA_BETA, classifier=fake)
    result = backend.score(_window(120.0, offset=25.0))
    assert result.raw_score == pytest.approx(2.0)
    assert result.calibrated_probability == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))
    assert result.window == "120s@16000Hz@25s"
    np.testing.assert_array_equal(fake.samples, _window(120.0, offset=25.0).samples)

    bad = cast(
        AudioWindow, SimpleNamespace(samples=np.zeros(1_000), sample_rate=8_000, warnings=())
    )
    with pytest.raises(AudioDecodeError):
        backend.score(bad)


def test_spctttra_unload_prevents_further_scoring() -> None:
    fake = _FakeClassifier()
    backend = load_spctttra(MODEL_KEY_SPECTTRA_ALPHA, classifier=fake)
    backend.unload()
    assert fake.unloaded
    with pytest.raises(ModelLoadError, match="unloaded"):
        backend.score(_window(120.0))


def test_ai_probability_resolves_class_by_label() -> None:
    scores = {0: 0.9, 1: 0.1}
    assert ai_probability(scores, {0: "human", 1: "ai_generated"}, AI_GENERATED_LABEL) == 0.1
    assert ai_probability(scores, {0: "ai_generated", 1: "human"}, AI_GENERATED_LABEL) == 0.9
    with pytest.raises(ModelLoadError, match="ai_generated"):
        ai_probability(scores, {0: "human", 1: "other"}, AI_GENERATED_LABEL)


def test_ast_pipeline_yields_probability_and_metadata() -> None:
    fake = _FakePipeline()
    backend = load_ast(MODEL_KEY_AST, pipeline=fake)
    result = backend.score(_window(60.0, offset=25.0))
    assert result.raw_score == pytest.approx(0.1)
    assert result.model_id == AST_SPEC.model_id
    assert result.window == "60s@16000Hz@25s"
    assert backend.metadata().sample_rate == SAMPLE_RATE

    backend.unload()
    assert fake.unloaded
    with pytest.raises(ModelLoadError, match="unloaded"):
        backend.score(_window(60.0))


def test_loaders_require_injected_models_without_weights() -> None:
    with pytest.raises(ModelLoadError, match="ONNX"):
        load_fakeprint()
    with pytest.raises(ModelLoadError, match="classifier"):
        load_spctttra(MODEL_KEY_SPECTTRA_ALPHA)
    with pytest.raises(ModelLoadError, match="pipeline"):
        load_ast(MODEL_KEY_AST)


def test_missing_gpu_dependencies_point_to_full_or_gpu_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ModelLoadError, match=r"install the `full` or `gpu` extra"):
        loaders.load_spctttra_classifier(SPECTTRA_SPECS[MODEL_KEY_SPECTTRA_ALPHA])


def test_missing_ast_dependencies_point_to_ast_or_full_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ModelLoadError, match=r"install the `ast` or `full` extra"):
        loaders.load_ast_pipeline(AST_SPEC)


def test_backend_modules_do_not_import_model_libraries() -> None:
    import sys

    for module in ("torch", "onnxruntime", "transformers", "sonics"):
        sys.modules.pop(module, None)

    assert "torch" not in sys.modules
    assert "onnxruntime" not in sys.modules
    assert "transformers" not in sys.modules
    assert "sonics" not in sys.modules


def test_loader_honours_injected_cache_dir() -> None:
    calls: list[tuple[str, Path | None, str]] = []

    def loader(spec: object, *, cache_dir: Path | None, device: str) -> _FakeClassifier:
        calls.append((str(spec), cache_dir, device))
        return _FakeClassifier()

    load_spctttra(
        MODEL_KEY_SPECTTRA_ALPHA,
        classifier_loader=loader,
        cache_dir=Path("/tmp/aidetect-cache"),
        device="cpu",
    )
    assert calls == [
        (str(SPECTTRA_SPECS[MODEL_KEY_SPECTTRA_ALPHA]), Path("/tmp/aidetect-cache"), "cpu")
    ]


def test_cache_dir_honours_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "weights"
    monkeypatch.setenv("AIDETECT_CACHE_DIR", str(override))
    assert loaders.cache_dir() == override

    monkeypatch.delenv("AIDETECT_CACHE_DIR", raising=False)
    assert loaders.cache_dir() == Path.home() / ".cache" / "huggingface" / "hub"
