"""Calibrated, tri-state AI-generated music detection.

Heavy model libraries are imported only when a detector actually loads weights, so a plain
``import aidetect`` stays lightweight and offline.
"""

from __future__ import annotations

from aidetect.calibration import Calibration, calibration_fallback, label_for_probability
from aidetect.config import (
    DEFAULT_CONFIG_NAME,
    SECOND_OPINION_CONFIG_NAME,
    BackendConfig,
    get_backend_config,
    get_calibration,
    get_config_for_model,
)
from aidetect.constants import (
    AI_GENERATED_LABEL,
    HUMAN_LABEL,
    UNCERTAIN_LABEL,
    Placement,
    TriStateLabel,
)
from aidetect.detector import CascadeDetector, Detector
from aidetect.errors import AidetectError, AudioDecodeError, ModelLoadError
from aidetect.types import CascadeResult, Result, Timings

__all__ = [
    "AI_GENERATED_LABEL",
    "AidetectError",
    "AudioDecodeError",
    "BackendConfig",
    "Calibration",
    "CascadeDetector",
    "CascadeResult",
    "DEFAULT_CONFIG_NAME",
    "Detector",
    "HUMAN_LABEL",
    "ModelLoadError",
    "Placement",
    "Result",
    "SECOND_OPINION_CONFIG_NAME",
    "Timings",
    "TriStateLabel",
    "UNCERTAIN_LABEL",
    "calibration_fallback",
    "get_backend_config",
    "get_calibration",
    "get_config_for_model",
    "label_for_probability",
]
