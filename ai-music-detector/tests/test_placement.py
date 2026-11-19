from __future__ import annotations

from typing import cast

import pytest

from aidetect.constants import Placement
from aidetect.placement import resolve_offset_seconds


@pytest.mark.parametrize(
    ("duration_seconds", "window_seconds", "placement", "expected"),
    [
        (30.0, 60.0, "start", 0.0),
        (100.0, 60.0, "middle", 20.0),
        (100.0, 60.0, "end", 40.0),
        (212.6, 120.0, "middle", 46.3),
        (121.0, 120.0, "end", 1.0),
    ],
)
def test_resolve_offset_seconds_is_duration_relative(
    duration_seconds: float, window_seconds: float, placement: Placement, expected: float
) -> None:
    assert resolve_offset_seconds(duration_seconds, window_seconds, placement) == pytest.approx(
        expected
    )


def test_resolve_offset_seconds_rejects_fixed_placement() -> None:
    with pytest.raises(ValueError, match="fixed"):
        resolve_offset_seconds(100.0, 60.0, "fixed")


@pytest.mark.parametrize(
    ("duration_seconds", "window_seconds", "placement"),
    [
        (-1.0, 60.0, "start"),
        (100.0, 0.0, "middle"),
        (100.0, float("nan"), "end"),
        (100.0, 60.0, "unknown"),
    ],
)
def test_resolve_offset_seconds_rejects_invalid_inputs(
    duration_seconds: float, window_seconds: float, placement: str
) -> None:
    with pytest.raises(ValueError):
        resolve_offset_seconds(duration_seconds, window_seconds, cast(Placement, placement))
