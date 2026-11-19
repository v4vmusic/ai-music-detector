from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

import benches.backends.ast as ast_module
from benches._shared.constants import AI_GENERATED_LABEL, SAMPLE_RATE
from benches._shared.errors import AudioDecodeError, ModelLoadError
from benches._shared.registry import AST_SPEC, MODEL_KEY_AST, AstSpec
from benches._shared.types import AudioWindow
from benches.backends.ast import ai_probability, load


class _FakePipeline:
    """Per-class-id outputs and id2label; no model, GPU or network."""

    def __init__(
        self,
        scores: Mapping[int, float] | None = None,
        id2label: Mapping[int, str] | None = None,
    ) -> None:
        self.scores = scores if scores is not None else {0: 0.9, 1: 0.1}
        self.labels = id2label if id2label is not None else {0: "human", 1: "ai_generated"}
        self.calls: list[tuple[NDArray[np.float32], int]] = []
        self.unloaded = False

    def __call__(self, samples: NDArray[np.float32], *, sampling_rate: int) -> Mapping[int, float]:
        self.calls.append((samples, sampling_rate))
        return self.scores

    def id2label(self) -> Mapping[int, str]:
        return self.labels

    def unload(self) -> None:
        self.unloaded = True


def _window(seconds: float = 60.0, offset: float = 25.0) -> AudioWindow:
    count = int(round(seconds * SAMPLE_RATE))
    return AudioWindow(
        samples=np.full(count, 0.1, dtype=np.float32),
        sample_rate=SAMPLE_RATE,
        offset_seconds=offset,
        window_seconds=seconds,
    )


def test_ai_probability_uses_id2label_not_class_index() -> None:
    scores = {0: 0.9, 1: 0.1}

    assert ai_probability(scores, {0: "human", 1: "ai_generated"}, AI_GENERATED_LABEL) == 0.1
    assert ai_probability(scores, {0: "ai_generated", 1: "human"}, AI_GENERATED_LABEL) == 0.9


def test_ai_probability_requires_one_labelled_score_and_a_valid_value() -> None:
    scores = {0: 0.9, 1: 0.1}
    with pytest.raises(ModelLoadError, match="ai_generated"):
        ai_probability(scores, {0: "human", 1: "other"}, AI_GENERATED_LABEL)
    with pytest.raises(ModelLoadError, match="missing"):
        ai_probability({0: 0.9}, {0: "human", 1: "ai_generated"}, AI_GENERATED_LABEL)
    with pytest.raises(ModelLoadError, match="probability"):
        ai_probability({0: 0.9, 1: 2.5}, {0: "human", 1: "ai_generated"}, AI_GENERATED_LABEL)


def test_fake_pipeline_yields_probability_and_metadata() -> None:
    fake = _FakePipeline()
    backend = load(MODEL_KEY_AST, pipeline=fake)
    window = _window()

    result = backend.score(window)

    assert result.raw_score == pytest.approx(0.1)
    assert result.calibrated_probability == pytest.approx(0.1)
    assert result.model_id == AST_SPEC.model_id
    assert result.model_revision == AST_SPEC.revision
    assert result.window == "60s@16000Hz@25s"
    assert result.warnings == ()
    assert result.timings.score_seconds >= 0.0
    assert len(fake.calls) == 1
    np.testing.assert_array_equal(fake.calls[0][0], window.samples)
    assert fake.calls[0][1] == SAMPLE_RATE

    metadata = backend.metadata()
    assert metadata.model_id == AST_SPEC.model_id
    assert metadata.revision == AST_SPEC.revision
    assert metadata.window_seconds == 60.0
    assert metadata.sample_rate == SAMPLE_RATE
    assert metadata.preprocessing_version == AST_SPEC.preprocessing_version


def test_short_track_warning_reaches_the_output() -> None:
    window = AudioWindow(
        samples=_window().samples,
        sample_rate=SAMPLE_RATE,
        offset_seconds=25.0,
        window_seconds=60.0,
        warnings=("short_track_padded",),
    )

    result = load(MODEL_KEY_AST, pipeline=_FakePipeline()).score(window)

    assert result.warnings == ("short_track_padded",)


def test_two_loads_reuse_the_pipeline() -> None:
    calls: list[tuple[str, Path | None, str]] = []

    def loader(spec: AstSpec, *, cache_dir: Path | None, device: str) -> _FakePipeline:
        calls.append((spec.key, cache_dir, device))
        return _FakePipeline()

    first = load(
        MODEL_KEY_AST, pipeline_loader=loader, cache_dir=Path("/tmp/aidetect"), device="cpu"
    )
    second = load(
        MODEL_KEY_AST, pipeline_loader=loader, cache_dir=Path("/tmp/aidetect"), device="cpu"
    )

    assert len(calls) == 1
    assert calls[0] == (MODEL_KEY_AST, Path("/tmp/aidetect"), "cpu")
    assert first.metadata() == second.metadata()
    for name in ("_MODEL", "_PIPELINE", "_model", "_pipeline_instance"):
        assert not hasattr(ast_module, name)


def test_reload_after_unload_builds_a_fresh_pipeline() -> None:
    calls: list[int] = []

    def loader(spec: AstSpec, *, cache_dir: Path | None, device: str) -> _FakePipeline:
        calls.append(1)
        return _FakePipeline()

    first = load(MODEL_KEY_AST, pipeline_loader=loader)
    first.unload()

    second = load(MODEL_KEY_AST, pipeline_loader=loader)

    assert len(calls) == 2
    assert second.score(_window()).calibrated_probability == pytest.approx(0.1)


def test_load_requires_a_pipeline_or_loader() -> None:
    with pytest.raises(ModelLoadError, match="pipeline"):
        load(MODEL_KEY_AST)


def test_unload_prevents_further_scoring() -> None:
    fake = _FakePipeline()
    backend = load(MODEL_KEY_AST, pipeline=fake)

    backend.unload()

    assert fake.unloaded
    with pytest.raises(ModelLoadError, match="unloaded"):
        backend.score(_window())


@pytest.mark.parametrize(
    "samples,rate",
    [
        (np.zeros(60 * SAMPLE_RATE, dtype=np.float32), 8_000),
        (np.zeros((60 * SAMPLE_RATE, 2), dtype=np.float32), SAMPLE_RATE),
        (np.zeros(1_000, dtype=np.float32), SAMPLE_RATE),
    ],
)
def test_score_rejects_invalid_windows_with_shared_error(
    samples: NDArray[np.float32], rate: int
) -> None:
    bad_window = cast(
        AudioWindow,
        SimpleNamespace(
            samples=samples,
            sample_rate=rate,
            offset_seconds=25.0,
            window_seconds=60.0,
            warnings=(),
        ),
    )

    backend = load(MODEL_KEY_AST, pipeline=_FakePipeline())
    with pytest.raises(AudioDecodeError):
        backend.score(bad_window)
