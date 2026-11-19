from __future__ import annotations

from typing import Final, Literal

SAMPLE_RATE: Final = 16_000
MIN_TRACK_SECONDS: Final = 5.0

WARNING_SHORT_TRACK_PADDED: Final = "short_track_padded"
WARNING_LOOP_FILLED: Final = "loop_filled"
WARNING_PEAK_NORMALIZED: Final = "peak_normalized_to_fit_range"

MODEL_KEY_FAKEPRINT: Final = "fakeprint"
MODEL_KEY_SPECTTRA_ALPHA: Final = "spectttra-alpha-120s"
MODEL_KEY_SPECTTRA_BETA: Final = "spectttra-beta-120s"
MODEL_KEY_SPECTTRA_GAMMA: Final = "spectttra-gamma-120s"
MODEL_KEY_AST: Final = "ast-60s"

HUMAN_LABEL: Final = "human"
AI_GENERATED_LABEL: Final = "ai_generated"
UNCERTAIN_LABEL: Final = "uncertain"
LABELS: Final = (HUMAN_LABEL, AI_GENERATED_LABEL)
TRI_STATE_LABELS: Final = (HUMAN_LABEL, UNCERTAIN_LABEL, AI_GENERATED_LABEL)

FALLBACK_LOW_THRESHOLD: Final = 0.35
FALLBACK_HIGH_THRESHOLD: Final = 0.65

Label = Literal["human", "ai_generated"]
TriStateLabel = Literal["human", "uncertain", "ai_generated"]
Placement = Literal["fixed", "start", "middle", "end"]
FillMode = Literal["zero-pad", "loop-fill"]

PLACEMENT_FIXED: Final = "fixed"
PLACEMENTS: Final[tuple[Placement, ...]] = (PLACEMENT_FIXED, "start", "middle", "end")
FILL_MODES: Final[tuple[FillMode, ...]] = ("zero-pad", "loop-fill")

__all__ = [
    "AI_GENERATED_LABEL",
    "FALLBACK_HIGH_THRESHOLD",
    "FALLBACK_LOW_THRESHOLD",
    "FILL_MODES",
    "HUMAN_LABEL",
    "LABELS",
    "MIN_TRACK_SECONDS",
    "MODEL_KEY_AST",
    "MODEL_KEY_FAKEPRINT",
    "MODEL_KEY_SPECTTRA_ALPHA",
    "MODEL_KEY_SPECTTRA_BETA",
    "MODEL_KEY_SPECTTRA_GAMMA",
    "PLACEMENT_FIXED",
    "PLACEMENTS",
    "SAMPLE_RATE",
    "TRI_STATE_LABELS",
    "UNCERTAIN_LABEL",
    "WARNING_LOOP_FILLED",
    "WARNING_PEAK_NORMALIZED",
    "WARNING_SHORT_TRACK_PADDED",
    "FillMode",
    "Label",
    "Placement",
    "TriStateLabel",
]
