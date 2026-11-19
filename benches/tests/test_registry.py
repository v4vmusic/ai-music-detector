from __future__ import annotations

from benches._shared.registry import MODEL_KEY_FAKEPRINT, MODEL_KEYS


def test_registry_has_unique_string_keys() -> None:
    assert MODEL_KEY_FAKEPRINT in MODEL_KEYS
    assert all(isinstance(key, str) and key for key in MODEL_KEYS)
    assert len(MODEL_KEYS) == len(set(MODEL_KEYS))
