from __future__ import annotations

import math
import subprocess
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from aidetect.constants import (
    FILL_MODES,
    MIN_TRACK_SECONDS,
    PLACEMENT_FIXED,
    SAMPLE_RATE,
    WARNING_LOOP_FILLED,
    WARNING_PEAK_NORMALIZED,
    WARNING_SHORT_TRACK_PADDED,
    FillMode,
    Placement,
)
from aidetect.errors import AudioDecodeError
from aidetect.placement import resolve_offset_seconds
from aidetect.types import AudioWindow

__all__ = ["decode_window", "decode_window_bytes"]

_FFMPEG_BASE = [
    "ffmpeg",
    "-hide_banner",
    "-loglevel",
    "error",
    "-nostdin",
]
_FFMPEG_OUTPUT = [
    "-vn",
    "-ac",
    "1",
    "-ar",
    str(SAMPLE_RATE),
    "-f",
    "f32le",
    "-acodec",
    "pcm_f32le",
    "pipe:1",
]


def _as_pcm(payload: bytes, source: str) -> NDArray[np.float32]:
    if not payload:
        raise AudioDecodeError(f"ffmpeg produced no audio samples for {source}")
    if len(payload) % 4 != 0:
        raise AudioDecodeError(f"ffmpeg produced incomplete float32 samples for {source}")
    return cast(NDArray[np.float32], np.frombuffer(payload, dtype="<f4").copy())


def _decode_pcm(path: Path) -> NDArray[np.float32]:
    command = [*_FFMPEG_BASE, "-i", str(path), *_FFMPEG_OUTPUT]
    result = subprocess.run(command, capture_output=True, check=False, shell=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise AudioDecodeError(f"ffmpeg could not decode {path}: {detail or result.returncode}")
    return _as_pcm(result.stdout, str(path))


def _decode_pcm_bytes(data: bytes) -> NDArray[np.float32]:
    command = [*_FFMPEG_BASE, "-i", "pipe:0", *_FFMPEG_OUTPUT]
    result = subprocess.run(command, input=data, capture_output=True, check=False, shell=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise AudioDecodeError(
            f"ffmpeg could not decode audio bytes: {detail or result.returncode}"
        )
    return _as_pcm(result.stdout, "audio bytes")


def _loop_fill(segment: NDArray[np.float32], required: int) -> NDArray[np.float32]:
    """Repeat the available segment to fill the window, with no crossfade."""

    repeats = (required + segment.size - 1) // segment.size
    return np.tile(segment, repeats)[:required]


def _window_from_samples(
    samples: NDArray[np.float32],
    source: str,
    *,
    window_seconds: float,
    offset_seconds: float,
    fill_mode: FillMode,
    placement: Placement | None,
) -> AudioWindow:
    if not math.isfinite(window_seconds) or window_seconds <= 0.0:
        raise ValueError(f"window_seconds must be positive and finite, got {window_seconds!r}")
    if not math.isfinite(offset_seconds) or offset_seconds < 0.0:
        raise ValueError(f"offset_seconds must be non-negative and finite, got {offset_seconds!r}")
    if fill_mode not in FILL_MODES:
        raise ValueError(f"unknown fill mode {fill_mode!r}; expected one of {FILL_MODES}")

    duration_seconds = samples.size / SAMPLE_RATE
    if duration_seconds < MIN_TRACK_SECONDS:
        raise AudioDecodeError(
            f"Track {source} is {duration_seconds:.2f} s; "
            f"at least {MIN_TRACK_SECONDS:.0f} s is required"
        )
    if placement == PLACEMENT_FIXED:
        placement = None
    if placement is not None:
        offset_seconds = resolve_offset_seconds(duration_seconds, window_seconds, placement)
        if not math.isfinite(offset_seconds) or offset_seconds < 0.0:
            raise ValueError(
                f"resolved offset must be non-negative and finite, got {offset_seconds!r}"
            )

    required = int(round(window_seconds * SAMPLE_RATE))
    start = int(round(offset_seconds * SAMPLE_RATE))
    segment = samples[start : start + required]
    if segment.size == required:
        window = segment
        window_warnings: tuple[str, ...] = ()
    elif fill_mode == "loop-fill":
        if segment.size == 0:
            raise AudioDecodeError(
                f"Track {source} has no audio at offset {offset_seconds:g} s; "
                "loop-fill cannot fill the window"
            )
        window = _loop_fill(segment, required)
        window_warnings = (WARNING_LOOP_FILLED,)
        warnings.warn(
            f"Track {source} cannot fill the {window_seconds:g} s window at "
            f"offset {offset_seconds:g} s ({WARNING_LOOP_FILLED}); loop-filled without crossfade",
            UserWarning,
            stacklevel=2,
        )
    else:
        window = np.zeros(required, dtype=np.float32)
        window[: segment.size] = segment
        window_warnings = (WARNING_SHORT_TRACK_PADDED,)
        warnings.warn(
            f"Track {source} cannot fill the {window_seconds:g} s window at "
            f"offset {offset_seconds:g} s ({WARNING_SHORT_TRACK_PADDED}); zero-padded",
            UserWarning,
            stacklevel=2,
        )

    peak = float(np.max(np.abs(window))) if window.size else 0.0
    if peak > 1.0:
        scale = np.float32(peak)
        window = (window / scale).astype(np.float32, copy=False)
        window_warnings = (*window_warnings, WARNING_PEAK_NORMALIZED)
        warnings.warn(
            f"Track {source} peak {peak:.6g} exceeds 1.0 in the "
            f"{window_seconds:g} s window at offset {offset_seconds:g} s "
            f"({WARNING_PEAK_NORMALIZED}); peak-normalized to [-1, 1]",
            UserWarning,
            stacklevel=2,
        )

    return AudioWindow(
        samples=window,
        sample_rate=SAMPLE_RATE,
        offset_seconds=offset_seconds,
        window_seconds=window_seconds,
        warnings=window_warnings,
    )


def decode_window(
    path: Path | str,
    *,
    window_seconds: float,
    offset_seconds: float = 0.0,
    fill_mode: FillMode = "zero-pad",
    placement: Placement | None = None,
    _pcm_decoder: Callable[[Path], NDArray[np.float32]] | None = None,
) -> AudioWindow:
    """Decode a local file to one 16 kHz mono float32 window.

    ``placement`` resolves a duration-relative offset per track and overrides
    ``offset_seconds``; ``None`` keeps the fixed offset. Short segments are zero-padded or
    loop-filled according to ``fill_mode``. ffmpeg is always called with an argument list and
    ``shell=False``.
    """

    decoder = _pcm_decoder if _pcm_decoder is not None else _decode_pcm
    samples = decoder(Path(path))
    return _window_from_samples(
        samples,
        str(path),
        window_seconds=window_seconds,
        offset_seconds=offset_seconds,
        fill_mode=fill_mode,
        placement=placement,
    )


def decode_window_bytes(
    data: bytes,
    *,
    window_seconds: float,
    offset_seconds: float = 0.0,
    fill_mode: FillMode = "zero-pad",
    placement: Placement | None = None,
) -> AudioWindow:
    """Decode in-memory audio bytes to the same window shape as :func:`decode_window`."""

    if not data:
        raise ValueError("audio bytes must not be empty")
    samples = _decode_pcm_bytes(data)
    return _window_from_samples(
        samples,
        "audio bytes",
        window_seconds=window_seconds,
        offset_seconds=offset_seconds,
        fill_mode=fill_mode,
        placement=placement,
    )
