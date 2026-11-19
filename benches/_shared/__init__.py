from __future__ import annotations

from benches._shared.audio import decode_window
from benches._shared.constants import (
    AI_GENERATED_LABEL,
    HUMAN_LABEL,
    LABELS,
    MIN_TRACK_SECONDS,
    SAMPLE_RATE,
    WARNING_PEAK_NORMALIZED,
    WARNING_SHORT_TRACK_PADDED,
    Label,
)
from benches._shared.errors import (
    AidetectError,
    AudioDecodeError,
    InvalidManifestError,
    ModelLoadError,
)
from benches._shared.manifest import ManifestEntry, load_manifest
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
    "HUMAN_LABEL",
    "LABELS",
    "MIN_TRACK_SECONDS",
    "MODEL_KEY_FAKEPRINT",
    "MODEL_KEYS",
    "SAMPLE_RATE",
    "WARNING_PEAK_NORMALIZED",
    "WARNING_SHORT_TRACK_PADDED",
    "AidetectError",
    "AudioDecodeError",
    "AudioWindow",
    "InvalidManifestError",
    "Label",
    "ManifestEntry",
    "ModelLoadError",
    "ModelMetadata",
    "ModelOutput",
    "ModelSpec",
    "Timings",
    "decode_window",
    "load_manifest",
    "validate_audio_window",
]
