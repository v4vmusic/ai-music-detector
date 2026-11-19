from __future__ import annotations

from benches._shared.audio import decode_window
from benches._shared.constants import (
    AI_GENERATED_LABEL,
    FILL_MODES,
    HUMAN_LABEL,
    LABELS,
    MIN_TRACK_SECONDS,
    PLACEMENT_FIXED,
    PLACEMENTS,
    SAMPLE_RATE,
    WARNING_LOOP_FILLED,
    WARNING_PEAK_NORMALIZED,
    WARNING_SHORT_TRACK_PADDED,
    FillMode,
    Label,
    Placement,
)
from benches._shared.errors import (
    AidetectError,
    AudioDecodeError,
    InvalidManifestError,
    ModelLoadError,
)
from benches._shared.experiment import ExperimentConfig
from benches._shared.manifest import ManifestEntry, load_manifest
from benches._shared.placement import resolve_offset_seconds
from benches._shared.registry import FAKEPRINT_SPEC, MODEL_KEY_FAKEPRINT, MODEL_KEYS, ModelSpec
from benches._shared.types import (
    AudioWindow,
    ModelMetadata,
    ModelOutput,
    Timings,
    validate_audio_window,
)

__all__ = [
    "AI_GENERATED_LABEL",
    "FAKEPRINT_SPEC",
    "FILL_MODES",
    "HUMAN_LABEL",
    "LABELS",
    "MIN_TRACK_SECONDS",
    "MODEL_KEY_FAKEPRINT",
    "MODEL_KEYS",
    "PLACEMENT_FIXED",
    "PLACEMENTS",
    "SAMPLE_RATE",
    "WARNING_LOOP_FILLED",
    "WARNING_PEAK_NORMALIZED",
    "WARNING_SHORT_TRACK_PADDED",
    "AidetectError",
    "AudioDecodeError",
    "AudioWindow",
    "ExperimentConfig",
    "FillMode",
    "InvalidManifestError",
    "Label",
    "ManifestEntry",
    "ModelLoadError",
    "ModelMetadata",
    "ModelOutput",
    "ModelSpec",
    "Placement",
    "Timings",
    "decode_window",
    "load_manifest",
    "resolve_offset_seconds",
    "validate_audio_window",
]
