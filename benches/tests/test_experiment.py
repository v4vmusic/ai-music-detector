from __future__ import annotations

from pathlib import Path

import pytest

from benches._shared.experiment import (
    BASELINE_RESULTS_DIR,
    DEFAULT_EXPERIMENTS_DIR,
    ExperimentConfig,
    assert_experiment_location,
    experiment_raw_path,
    validate_config_name,
)


def test_experiment_config_records_placement_and_fill() -> None:
    config = ExperimentConfig(name="AST-1b", placement="middle", fill_mode="loop-fill")
    assert config.as_dict() == {
        "config_name": "AST-1b",
        "placement": "middle",
        "fill_mode": "loop-fill",
    }


def test_experiment_config_records_fixed_offset_only_for_fixed_placement() -> None:
    fixed = ExperimentConfig(placement="fixed", offset_seconds=25.0)
    placed = ExperimentConfig(placement="middle", offset_seconds=25.0)
    assert fixed.as_dict() == {
        "placement": "fixed",
        "fill_mode": "zero-pad",
        "offset_seconds": 25.0,
    }
    assert placed.as_dict() == {"placement": "middle", "fill_mode": "zero-pad"}


@pytest.mark.parametrize("name", ["", "../escape", "a/b", "a b"])
def test_validate_config_name_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError, match="config name"):
        validate_config_name(name)


def test_experiment_raw_path_uses_config_and_backend_names() -> None:
    path = experiment_raw_path(Path("experiments"), "AST-1a", "ast-60s")
    assert path == Path("experiments/AST-1a/raw/ast-60s.jsonl")


def test_default_experiment_dir_is_under_results_but_not_baseline() -> None:
    assert DEFAULT_EXPERIMENTS_DIR == Path("benches/results/tuning")
    assert_experiment_location(DEFAULT_EXPERIMENTS_DIR / "AST-1a")


@pytest.mark.parametrize(
    "path",
    [
        BASELINE_RESULTS_DIR,
        BASELINE_RESULTS_DIR / "raw",
        BASELINE_RESULTS_DIR / "raw" / "ast-60s.jsonl",
        BASELINE_RESULTS_DIR / "summary.json",
    ],
)
def test_assert_experiment_location_rejects_baseline_paths(path: Path) -> None:
    with pytest.raises(ValueError, match="baseline"):
        assert_experiment_location(path)


def test_experiment_config_rejects_unknown_placement_and_fill() -> None:
    from typing import cast

    from benches._shared.constants import FillMode, Placement

    with pytest.raises(ValueError, match="placement"):
        ExperimentConfig(placement=cast(Placement, "side"))
    with pytest.raises(ValueError, match="fill"):
        ExperimentConfig(fill_mode=cast(FillMode, "crossfade"))
