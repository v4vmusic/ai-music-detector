from __future__ import annotations

import math

from aidetect.constants import PLACEMENT_FIXED, PLACEMENTS, Placement

__all__ = ["resolve_offset_seconds"]


def resolve_offset_seconds(
    duration_seconds: float, window_seconds: float, placement: Placement
) -> float:
    """Resolve a duration-relative window offset for one track.

    ``fixed`` has no duration-derived offset: callers pass the fixed offset directly.
    This is the shared placement rule promoted from the bench harness.
    """

    if not math.isfinite(duration_seconds) or duration_seconds < 0.0:
        raise ValueError(
            f"duration_seconds must be non-negative and finite, got {duration_seconds!r}"
        )
    if not math.isfinite(window_seconds) or window_seconds <= 0.0:
        raise ValueError(f"window_seconds must be positive and finite, got {window_seconds!r}")
    if placement not in PLACEMENTS:
        raise ValueError(f"unknown placement {placement!r}; expected one of {PLACEMENTS}")
    if placement == PLACEMENT_FIXED:
        raise ValueError("fixed placement uses the caller's offset_seconds, not the track duration")
    if placement == "start":
        return 0.0
    if placement == "middle":
        return max(0.0, (duration_seconds - window_seconds) / 2.0)
    return max(0.0, duration_seconds - window_seconds)
