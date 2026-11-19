from __future__ import annotations

import math
import struct
import wave
from collections.abc import Callable
from pathlib import Path

import pytest

from benches._shared.constants import SAMPLE_RATE

WriteWav = Callable[[Path, float], Path]


@pytest.fixture
def write_wav() -> WriteWav:
    """Create a mono 16-bit PCM sine-wave WAV with the standard library."""

    def _write_wav(path: Path, seconds: float, frequency: float = 440.0) -> Path:
        frame_count = int(round(seconds * SAMPLE_RATE))
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            frames = b"".join(
                struct.pack(
                    "<h",
                    int(32767 * 0.5 * math.sin(2 * math.pi * frequency * index / SAMPLE_RATE)),
                )
                for index in range(frame_count)
            )
            handle.writeframes(frames)
        return path

    return _write_wav
