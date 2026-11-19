from __future__ import annotations

import pytest

from benches._shared.constants import AI_GENERATED_LABEL, SAMPLE_RATE
from benches._shared.errors import ModelLoadError
from benches._shared.registry import (
    AST_SPEC,
    MODEL_KEY_AST,
    MODEL_KEY_FAKEPRINT,
    MODEL_KEYS,
    get_ast_spec,
)


def test_registry_has_unique_string_keys() -> None:
    assert MODEL_KEY_FAKEPRINT in MODEL_KEYS
    assert all(isinstance(key, str) and key for key in MODEL_KEYS)
    assert len(MODEL_KEYS) == len(set(MODEL_KEYS))


def test_ast_spec_maps_to_the_checkpoint_and_a_60s_window() -> None:
    assert MODEL_KEY_AST in MODEL_KEYS
    assert AST_SPEC.key == MODEL_KEY_AST
    assert AST_SPEC.model_id == "AI-Music-Detection/ai_music_detection_large_60s"
    assert AST_SPEC.revision == "5e29461bbfc41dc5fe4ed5f7fadc961aa84a7e3e"
    assert AST_SPEC.window_seconds == 60.0
    assert AST_SPEC.sample_rate == SAMPLE_RATE
    assert AST_SPEC.default_offset_seconds == 25.0
    assert AST_SPEC.positive_label == AI_GENERATED_LABEL
    assert AST_SPEC.preprocessing_version
    assert get_ast_spec(MODEL_KEY_AST) is AST_SPEC


def test_unknown_ast_key_raises_model_load_error() -> None:
    with pytest.raises(ModelLoadError, match=MODEL_KEY_AST):
        get_ast_spec("ast-10s")
