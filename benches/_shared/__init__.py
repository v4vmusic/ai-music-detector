from __future__ import annotations

from benches._shared.audio import decode_window
from benches._shared.constants import (
    AI_GENERATED_LABEL,
    HUMAN_LABEL,
    LABELS,
    MIN_TRACK_SECONDS,
    SAMPLE_RATE,
    WARNING_SHORT_TRACK_PADDED,
    Label,
)
from benches._shared.errors import AidetectError, AudioDecodeError, InvalidManifestError
from benches._shared.manifest import ManifestEntry, load_manifest
from benches._shared.registry import MODEL_KEY_FAKEPRINT, MODEL_KEYS
from benches._shared.types import AudioWindow, ModelMetadata, ModelOutput, Timings

__all__ = [
    "AI_GENERATED_LABEL",
    "HUMAN_LABEL",
    "LABELS",
    "MIN_TRACK_SECONDS",
    "MODEL_KEY_FAKEPRINT",
    "MODEL_KEYS",
    "SAMPLE_RATE",
    "WARNING_SHORT_TRACK_PADDED",
    "AidetectError",
    "AudioDecodeError",
    "AudioWindow",
    "InvalidManifestError",
    "Label",
    "ManifestEntry",
    "ModelMetadata",
    "ModelOutput",
    "Timings",
    "decode_window",
    "load_manifest",
]
