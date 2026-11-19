from __future__ import annotations

import pytest

from aidetect.constants import AI_GENERATED_LABEL, SAMPLE_RATE
from aidetect.errors import ModelLoadError
from aidetect.registry import (
    AST_SPEC,
    FAKEPRINT_SPEC,
    MODEL_KEY_AST,
    MODEL_KEY_FAKEPRINT,
    MODEL_KEY_SPECTTRA_ALPHA,
    MODEL_KEYS,
    SPECTTRA_SPECS,
    get_ast_spec,
    get_specttra_spec,
)


def test_registry_holds_all_bench_validated_backends() -> None:
    assert MODEL_KEYS == (
        MODEL_KEY_FAKEPRINT,
        MODEL_KEY_SPECTTRA_ALPHA,
        "spectttra-beta-120s",
        "spectttra-gamma-120s",
        MODEL_KEY_AST,
    )
    assert len(MODEL_KEYS) == len(set(MODEL_KEYS))
    assert set(SPECTTRA_SPECS) == {key for key in MODEL_KEYS if key.startswith("spectttra-")}


def test_fakeprint_spec_is_pinned() -> None:
    assert FAKEPRINT_SPEC.model_id == "lofcz/ai-music-detector"
    assert FAKEPRINT_SPEC.window_seconds == 10.0
    assert FAKEPRINT_SPEC.sample_rate == SAMPLE_RATE
    assert len(FAKEPRINT_SPEC.weights_sha256) == 64
    assert FAKEPRINT_SPEC.feature_count == 3585


def test_ast_spec_maps_to_the_checkpoint_and_a_60s_window() -> None:
    assert AST_SPEC.key == MODEL_KEY_AST
    assert AST_SPEC.model_id == "AI-Music-Detection/ai_music_detection_large_60s"
    assert AST_SPEC.revision == "5e29461bbfc41dc5fe4ed5f7fadc961aa84a7e3e"
    assert AST_SPEC.window_seconds == 60.0
    assert AST_SPEC.positive_label == AI_GENERATED_LABEL
    assert get_ast_spec(MODEL_KEY_AST) is AST_SPEC


def test_unknown_model_keys_raise_model_load_error() -> None:
    with pytest.raises(ModelLoadError, match=MODEL_KEY_AST):
        get_ast_spec("ast-10s")
    with pytest.raises(ModelLoadError, match=MODEL_KEY_SPECTTRA_ALPHA):
        get_specttra_spec("spectttra-alpha-5s")
