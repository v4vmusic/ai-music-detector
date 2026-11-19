from __future__ import annotations

from typing import Final, Literal

SAMPLE_RATE: Final = 16_000
MIN_TRACK_SECONDS: Final = 5.0
WARNING_SHORT_TRACK_PADDED: Final = "short_track_padded"
WARNING_LOOP_FILLED: Final = "loop_filled"
WARNING_PEAK_NORMALIZED: Final = "peak_normalized_to_fit_range"

HUMAN_LABEL: Final = "human"
AI_GENERATED_LABEL: Final = "ai_generated"
LABELS: Final = (HUMAN_LABEL, AI_GENERATED_LABEL)

Label = Literal["human", "ai_generated"]
Placement = Literal["fixed", "start", "middle", "end"]
FillMode = Literal["zero-pad", "loop-fill"]

PLACEMENT_FIXED: Final = "fixed"
PLACEMENTS: Final[tuple[Placement, ...]] = (PLACEMENT_FIXED, "start", "middle", "end")
FILL_MODES: Final[tuple[FillMode, ...]] = ("zero-pad", "loop-fill")

__all__ = [
    "AI_GENERATED_LABEL",
    "FILL_MODES",
    "HUMAN_LABEL",
    "LABELS",
    "MIN_TRACK_SECONDS",
    "PLACEMENT_FIXED",
    "PLACEMENTS",
    "SAMPLE_RATE",
    "WARNING_LOOP_FILLED",
    "WARNING_PEAK_NORMALIZED",
    "WARNING_SHORT_TRACK_PADDED",
    "FillMode",
    "Label",
    "Placement",
]
