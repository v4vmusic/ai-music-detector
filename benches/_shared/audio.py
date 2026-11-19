from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import aidetect.audio as _audio
from aidetect.constants import FillMode, Placement
from aidetect.types import AudioWindow

__all__ = ["decode_window", "decode_window_bytes"]


def _decode_pcm(path: Path) -> NDArray[np.float32]:
    return _audio._decode_pcm(path)


def decode_window(
    path: Path | str,
    *,
    window_seconds: float,
    offset_seconds: float = 0.0,
    fill_mode: FillMode = "zero-pad",
    placement: Placement | None = None,
) -> AudioWindow:
    """Bench-facing alias for the promoted decoder, with an injectable PCM reader."""

    return _audio.decode_window(
        path,
        window_seconds=window_seconds,
        offset_seconds=offset_seconds,
        fill_mode=fill_mode,
        placement=placement,
        _pcm_decoder=_decode_pcm,
    )


decode_window_bytes = _audio.decode_window_bytes
