from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Final, cast

from aidetect.calibration import Calibration
from aidetect.constants import FILL_MODES, PLACEMENTS, FillMode, Placement
from aidetect.errors import ModelLoadError

__all__ = [
    "DEFAULT_CONFIG_NAME",
    "SECOND_OPINION_CONFIG_NAME",
    "BackendConfig",
    "get_backend_config",
    "get_calibration",
    "get_config_for_model",
]

DEFAULT_CONFIG_NAME: Final = "FP-1b"
SECOND_OPINION_CONFIG_NAME: Final = "STA-1b"


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """A packaged operating point: backend, window, placement and raw flag threshold."""

    config_name: str
    backend: str
    model_key: str
    model_id: str
    window_seconds: float
    placement: Placement
    fill_mode: FillMode
    flag_threshold: float


def _read_resource(name: str) -> dict[str, object]:
    resource = files("aidetect.resources").joinpath(name)
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ModelLoadError(f"Packaged resource {name} is not a JSON object")
    return cast(dict[str, object], payload)


def _as_mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ModelLoadError(f"{description} must be a JSON object")
    return value


def _require_float(entry: dict[str, object], key: str, context: str) -> float:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelLoadError(f"{context} needs a numeric {key!r}")
    return float(value)


def _require_str(entry: dict[str, object], key: str, context: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str):
        raise ModelLoadError(f"{context} needs a string {key!r}")
    return value


def _model_key_for(value: str, models: dict[str, object]) -> str | None:
    if value in models:
        return value
    for key, raw in models.items():
        entry = _as_mapping(raw, f"Calibration entry {key!r}")
        if entry.get("model_id") == value:
            return key
    return None


def get_calibration(model_id_or_key: str) -> Calibration:
    """Load the exported temperature and tri-state thresholds for a registered backend."""

    payload = _read_resource("calibration.json")
    models = _as_mapping(payload.get("models"), "calibration.json 'models'")
    model_key = _model_key_for(model_id_or_key, models)
    if model_key is None:
        raise ModelLoadError(
            f"Unknown model_id {model_id_or_key!r}; expected one of {tuple(sorted(models))}"
        )
    raw = _as_mapping(models[model_key], f"Calibration entry {model_key!r}")
    warnings = raw.get("warnings", ())
    if not isinstance(warnings, (list, tuple)):
        raise ModelLoadError(f"Invalid calibration warnings for {model_key!r}")
    context = f"Calibration entry {model_key!r}"
    return Calibration(
        temperature=_require_float(raw, "temperature", context),
        low_threshold=_require_float(raw, "low_threshold", context),
        high_threshold=_require_float(raw, "high_threshold", context),
        warnings=tuple(str(warning) for warning in warnings),
        model_id=_require_str(raw, "model_id", context),
    )


def get_backend_config(config_name: str) -> BackendConfig:
    """Look up a packaged named config, including both D1 operating points."""

    payload = _read_resource("tuning.json")
    configs = _as_mapping(payload.get("configs"), "tuning.json 'configs'")
    raw = configs.get(config_name)
    if raw is None:
        raise ModelLoadError(
            f"Unknown config {config_name!r}; expected one of {tuple(sorted(configs))}"
        )
    entry = _as_mapping(raw, f"Tuning config {config_name!r}")
    context = f"Tuning config {config_name!r}"
    placement = _require_str(entry, "placement", context)
    fill_mode = _require_str(entry, "fill_mode", context)
    if placement not in PLACEMENTS:
        raise ModelLoadError(f"Invalid placement in config {config_name!r}: {placement!r}")
    if fill_mode not in FILL_MODES:
        raise ModelLoadError(f"Invalid fill_mode in config {config_name!r}: {fill_mode!r}")
    return BackendConfig(
        config_name=config_name,
        backend=_require_str(entry, "backend", context),
        model_key=_require_str(entry, "model_key", context),
        model_id=_require_str(entry, "model_id", context),
        window_seconds=_require_float(entry, "window_seconds", context),
        placement=placement,
        fill_mode=fill_mode,
        flag_threshold=_require_float(entry, "flag_threshold", context),
    )


def get_config_for_model(model_id_or_key: str) -> BackendConfig:
    """Return the dev-selected config for a model id or registry key."""

    payload = _read_resource("tuning.json")
    configs = _as_mapping(payload.get("configs"), "tuning.json 'configs'")
    for config_name, raw in configs.items():
        entry = _as_mapping(raw, f"Tuning config {config_name!r}")
        if model_id_or_key in (entry.get("model_id"), entry.get("model_key")):
            return get_backend_config(config_name)
    raise ModelLoadError(
        f"No packaged config for model_id {model_id_or_key!r}; "
        "use a bench-validated model id or registry key"
    )
