from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from benches._shared.constants import (
    FILL_MODES,
    PLACEMENT_FIXED,
    PLACEMENTS,
    FillMode,
    Placement,
)

__all__ = [
    "BASELINE_RESULTS_DIR",
    "DEFAULT_EXPERIMENTS_DIR",
    "ExperimentConfig",
    "assert_experiment_location",
    "experiment_raw_path",
    "validate_config_name",
]

BASELINE_RESULTS_DIR: Final = Path("benches/results")
DEFAULT_EXPERIMENTS_DIR: Final = Path("benches/results/tuning")
BASELINE_OUTPUT_NAMES: Final = ("calibration.json", "summary.json", "report.md")
CONFIG_NAME_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Windowing/fill choices that identify one tuning configuration."""

    name: str | None = None
    placement: Placement = PLACEMENT_FIXED
    offset_seconds: float = 0.0
    fill_mode: FillMode = "zero-pad"

    def __post_init__(self) -> None:
        if self.name is not None:
            validate_config_name(self.name)
        if self.placement not in PLACEMENTS:
            raise ValueError(f"unknown placement {self.placement!r}; expected one of {PLACEMENTS}")
        if self.fill_mode not in FILL_MODES:
            raise ValueError(f"unknown fill mode {self.fill_mode!r}; expected one of {FILL_MODES}")
        if not math.isfinite(self.offset_seconds) or self.offset_seconds < 0.0:
            raise ValueError(
                f"offset_seconds must be non-negative and finite, got {self.offset_seconds!r}"
            )

    def as_dict(self) -> dict[str, object]:
        """Return the recorded/committed form; a fixed offset is irrelevant for other placements."""

        payload: dict[str, object] = {
            "placement": self.placement,
            "fill_mode": self.fill_mode,
        }
        if self.placement == PLACEMENT_FIXED:
            payload["offset_seconds"] = self.offset_seconds
        if self.name is not None:
            payload["config_name"] = self.name
        return payload


def validate_config_name(name: str) -> str:
    """Reject path traversal and separators in experiment config names."""

    if not CONFIG_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Invalid config name {name!r}; expected [A-Za-z0-9][A-Za-z0-9._-]*")
    return name


def experiment_raw_path(experiments_dir: Path, config_name: str, backend_key: str) -> Path:
    """Return the raw JSONL path convention: <experiments>/<config>/raw/<backend>.jsonl."""

    return (
        experiments_dir
        / validate_config_name(config_name)
        / "raw"
        / f"{validate_config_name(backend_key)}.jsonl"
    )


def assert_experiment_location(
    path: Path, baseline_results_dir: Path = BASELINE_RESULTS_DIR
) -> None:
    """Protect Phase 5 results and raw files from tuning output paths."""

    resolved = path.resolve()
    baseline = baseline_results_dir.resolve()
    baseline_raw = (baseline_results_dir / "raw").resolve()
    if resolved == baseline:
        raise ValueError(f"Tuning output {path} would overwrite the baseline results directory")
    if resolved == baseline_raw or baseline_raw in resolved.parents:
        raise ValueError(f"Tuning output {path} would overwrite baseline raw results")
    if resolved.parent == baseline and resolved.name in BASELINE_OUTPUT_NAMES:
        raise ValueError(f"Tuning output {path} would overwrite a baseline committed artifact")
