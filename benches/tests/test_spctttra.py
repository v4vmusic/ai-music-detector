from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

import benches.backends.spctttra as spctttra
from benches._shared.constants import SAMPLE_RATE
from benches._shared.errors import AudioDecodeError, ModelLoadError
from benches._shared.registry import (
    MODEL_KEY_SPECTTRA_ALPHA,
    MODEL_KEY_SPECTTRA_BETA,
    MODEL_KEY_SPECTTRA_GAMMA,
    MODEL_KEYS,
    SPECTTRA_SPECS,
    get_specttra_spec,
)
from benches._shared.types import AudioWindow
from benches.backends.spctttra import load, sigmoid

VARIANTS = (MODEL_KEY_SPECTTRA_ALPHA, MODEL_KEY_SPECTTRA_BETA, MODEL_KEY_SPECTTRA_GAMMA)


class _FakeClassifier:
    """Classifier double: no weights, no torch, no GPU; records its input."""

    def __init__(self, logit: float = 1.25) -> None:
        self.logit = logit
        self.samples: NDArray[np.float32] | None = None
        self.unloaded = False

    def __call__(self, samples: NDArray[np.float32]) -> float:
        self.samples = samples
        return self.logit

    def unload(self) -> None:
        self.unloaded = True


def _window(seconds: float = 120.0, offset: float = 25.0) -> AudioWindow:
    count = int(round(seconds * SAMPLE_RATE))
    return AudioWindow(
        samples=np.full(count, 0.1, dtype=np.float32),
        sample_rate=SAMPLE_RATE,
        offset_seconds=offset,
        window_seconds=seconds,
    )


def test_registry_maps_variants_to_model_ids_and_windows() -> None:
    assert set(SPECTTRA_SPECS) == set(VARIANTS)
    for key in VARIANTS:
        assert key in MODEL_KEYS
        spec = SPECTTRA_SPECS[key]
        assert spec.key == key
        assert spec.model_id == f"awsaf49/sonics-{key}"
        assert len(spec.revision) == 40
        assert spec.window_seconds == 120.0
        assert spec.sample_rate == SAMPLE_RATE
        assert spec.default_offset_seconds == 25.0
        assert spec.preprocessing_version


def test_unknown_variant_raises_model_load_error() -> None:
    with pytest.raises(ModelLoadError, match=MODEL_KEY_SPECTTRA_ALPHA):
        get_specttra_spec("spectttra-alpha-5s")
    with pytest.raises(ModelLoadError, match=MODEL_KEY_SPECTTRA_ALPHA):
        load("spectttra-alpha-5s", classifier=_FakeClassifier())


def test_load_requires_an_injected_classifier() -> None:
    with pytest.raises(ModelLoadError, match="classifier"):
        load(MODEL_KEY_SPECTTRA_ALPHA)


def test_fake_classifier_logit_maps_to_probability_and_metadata() -> None:
    spec = SPECTTRA_SPECS[MODEL_KEY_SPECTTRA_BETA]
    fake = _FakeClassifier(logit=2.0)
    backend = load(MODEL_KEY_SPECTTRA_BETA, classifier=fake)
    window = _window()

    result = backend.score(window)

    assert result.raw_score == pytest.approx(2.0)
    assert result.calibrated_probability == pytest.approx(sigmoid(2.0))
    assert result.model_id == spec.model_id
    assert result.model_revision == spec.revision
    assert result.window == "120s@16000Hz@25s"
    assert result.warnings == ()
    assert result.timings.score_seconds >= 0.0
    assert fake.samples is not None
    np.testing.assert_array_equal(fake.samples, window.samples)

    metadata = backend.metadata()
    assert metadata.model_id == spec.model_id
    assert metadata.revision == spec.revision
    assert metadata.window_seconds == 120.0
    assert metadata.sample_rate == SAMPLE_RATE
    assert metadata.preprocessing_version == spec.preprocessing_version


def test_short_track_warning_reaches_the_output() -> None:
    window = AudioWindow(
        samples=_window().samples,
        sample_rate=SAMPLE_RATE,
        offset_seconds=25.0,
        window_seconds=120.0,
        warnings=("short_track_padded",),
    )

    result = load(MODEL_KEY_SPECTTRA_ALPHA, classifier=_FakeClassifier()).score(window)

    assert result.warnings == ("short_track_padded",)


@pytest.mark.parametrize(
    "logit,expected",
    [(0.0, 0.5), (2.0, 1.0 / (1.0 + math.exp(-2.0))), (-3.0, 1.0 / (1.0 + math.exp(3.0)))],
)
def test_sigmoid_matches_reference(logit: float, expected: float) -> None:
    assert sigmoid(logit) == pytest.approx(expected)


@pytest.mark.parametrize("logit", [-1_000.0, 1_000.0])
def test_sigmoid_is_finite_for_extreme_logits(logit: float) -> None:
    probability = sigmoid(logit)
    assert math.isfinite(probability)
    assert 0.0 <= probability <= 1.0


def test_load_unload_leaves_no_module_level_model_handle() -> None:
    fake = _FakeClassifier()
    calls: list[tuple[str, Path | None, str]] = []

    def loader(spec: object, *, cache_dir: Path | None, device: str) -> _FakeClassifier:
        calls.append((str(spec), cache_dir, device))
        return fake

    backend = load(
        MODEL_KEY_SPECTTRA_GAMMA,
        classifier_loader=loader,
        cache_dir=Path("/tmp/aidetect-cache"),
        device="cpu",
    )

    assert len(calls) == 1
    assert calls[0][1:] == (Path("/tmp/aidetect-cache"), "cpu")
    for name in ("_MODEL", "_CLASSIFIER", "_model", "_classifier"):
        assert not hasattr(spctttra, name)

    backend.unload()
    assert fake.unloaded
    with pytest.raises(ModelLoadError, match="unloaded"):
        backend.score(_window())


@pytest.mark.parametrize(
    "samples,rate",
    [
        (np.zeros(120 * SAMPLE_RATE, dtype=np.float32), 8_000),
        (np.zeros((120 * SAMPLE_RATE, 2), dtype=np.float32), SAMPLE_RATE),
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
            window_seconds=120.0,
            warnings=(),
        ),
    )

    backend = load(MODEL_KEY_SPECTTRA_ALPHA, classifier=_FakeClassifier())
    with pytest.raises(AudioDecodeError):
        backend.score(bad_window)
