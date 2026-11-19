from __future__ import annotations

import pytest

from aidetect.config import (
    DEFAULT_CONFIG_NAME,
    SECOND_OPINION_CONFIG_NAME,
    get_backend_config,
    get_calibration,
    get_config_for_model,
)
from aidetect.constants import MODEL_KEY_AST, MODEL_KEY_FAKEPRINT, MODEL_KEY_SPECTTRA_ALPHA
from aidetect.errors import ModelLoadError


def test_packaged_calibration_matches_exported_values() -> None:
    fakeprint = get_calibration(MODEL_KEY_FAKEPRINT)
    assert fakeprint.model_id == "lofcz/ai-music-detector"
    assert fakeprint.temperature == pytest.approx(5.0)
    assert fakeprint.low_threshold == pytest.approx(0.07959758536640082)
    assert fakeprint.high_threshold == pytest.approx(0.07959758536640082)
    assert fakeprint.warnings == ()

    alpha = get_calibration(MODEL_KEY_SPECTTRA_ALPHA)
    assert alpha.temperature == pytest.approx(1.7000000000000002)
    assert alpha.high_threshold == pytest.approx(0.3811434328992681)


def test_d1_configs_carry_dev_selected_placement_and_flag_thresholds() -> None:
    primary = get_backend_config(DEFAULT_CONFIG_NAME)
    assert primary.model_id == "lofcz/ai-music-detector"
    assert primary.window_seconds == 10.0
    assert primary.placement == "middle"
    assert primary.fill_mode == "zero-pad"
    assert primary.flag_threshold == pytest.approx(4.139542579650879e-05)

    second = get_backend_config(SECOND_OPINION_CONFIG_NAME)
    assert second.model_id == "awsaf49/sonics-spectttra-alpha-120s"
    assert second.window_seconds == 120.0
    assert second.placement == "middle"
    assert second.flag_threshold == pytest.approx(-1.8243114948272705)


def test_config_lookup_by_model_id_uses_dev_selected_defaults() -> None:
    config = get_config_for_model(MODEL_KEY_FAKEPRINT)
    assert config.config_name == DEFAULT_CONFIG_NAME
    assert config.placement == "middle"

    ast_config = get_config_for_model(MODEL_KEY_AST)
    assert ast_config.window_seconds == 60.0


def test_unknown_config_or_model_raises_actionable_error() -> None:
    with pytest.raises(ModelLoadError, match="config"):
        get_backend_config("does-not-exist")
    with pytest.raises(ModelLoadError, match="model_id"):
        get_calibration("not/a/model")
    with pytest.raises(ModelLoadError, match="model_id"):
        get_config_for_model("not-a-model")
