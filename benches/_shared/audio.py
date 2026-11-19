from __future__ import annotations

import math
import subprocess
import warnings
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from benches._shared.constants import MIN_TRACK_SECONDS, SAMPLE_RATE, WARNING_SHORT_TRACK_PADDED
from benches._shared.errors import AudioDecodeError
from benches._shared.types import AudioWindow

__all__ = ["decode_window"]


def _decode_pcm(path: Path) -> NDArray[np.float32]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(path),
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
    result = subprocess.run(command, capture_output=True, check=False, shell=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise AudioDecodeError(f"ffmpeg could not decode {path}: {detail or result.returncode}")
    if not result.stdout:
        raise AudioDecodeError(f"ffmpeg produced no audio samples for {path}")
    if len(result.stdout) % 4 != 0:
        raise AudioDecodeError(f"ffmpeg produced incomplete float32 samples for {path}")
    return cast(NDArray[np.float32], np.frombuffer(result.stdout, dtype="<f4").copy())


def decode_window(
    path: Path | str,
    *,
    window_seconds: float,
    offset_seconds: float = 0.0,
) -> AudioWindow:
    """Decode a local file to one 16 kHz mono float32 window at the given offset."""

    if not math.isfinite(window_seconds) or window_seconds <= 0.0:
        raise ValueError(f"window_seconds must be positive and finite, got {window_seconds!r}")
    if not math.isfinite(offset_seconds) or offset_seconds < 0.0:
        raise ValueError(f"offset_seconds must be non-negative and finite, got {offset_seconds!r}")

    audio_path = Path(path)
    samples = _decode_pcm(audio_path)
    duration_seconds = samples.size / SAMPLE_RATE
    if duration_seconds < MIN_TRACK_SECONDS:
        raise AudioDecodeError(
            f"Track {audio_path} is {duration_seconds:.2f} s; "
            f"at least {MIN_TRACK_SECONDS:.0f} s is required"
        )

    required = int(round(window_seconds * SAMPLE_RATE))
    start = int(round(offset_seconds * SAMPLE_RATE))
    segment = samples[start : start + required]
    if segment.size == required:
        window = segment
        window_warnings: tuple[str, ...] = ()
    else:
        window = np.zeros(required, dtype=np.float32)
        window[: segment.size] = segment
        window_warnings = (WARNING_SHORT_TRACK_PADDED,)
        warnings.warn(
            f"Track {audio_path} cannot fill the {window_seconds:g} s window at "
            f"offset {offset_seconds:g} s ({WARNING_SHORT_TRACK_PADDED}); zero-padded",
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
