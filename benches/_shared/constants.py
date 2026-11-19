from __future__ import annotations

from typing import Final, Literal

SAMPLE_RATE: Final = 16_000
MIN_TRACK_SECONDS: Final = 5.0
WARNING_SHORT_TRACK_PADDED: Final = "short_track_padded"

HUMAN_LABEL: Final = "human"
AI_GENERATED_LABEL: Final = "ai_generated"
LABELS: Final = (HUMAN_LABEL, AI_GENERATED_LABEL)

Label = Literal["human", "ai_generated"]

__all__ = [
    "AI_GENERATED_LABEL",
    "HUMAN_LABEL",
    "LABELS",
    "MIN_TRACK_SECONDS",
    "SAMPLE_RATE",
    "WARNING_SHORT_TRACK_PADDED",
    "Label",
]
