from __future__ import annotations

import math
import struct
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from aidetect.constants import SAMPLE_RATE
from aidetect.registry import MODEL_KEY_FAKEPRINT
from aidetect.types import ModelMetadata, ModelOutput, Timings

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


class FixedScorer:
    """Deterministic backend double for detector tests (no weights, no I/O)."""

    def __init__(self, raw_score: float, probability: float, model_id: str = MODEL_KEY_FAKEPRINT):
        self.raw_score = raw_score
        self.probability = probability
        self.model_id = model_id
        self.calls: list[tuple[int, float]] = []

    def score(self, window: object) -> ModelOutput:
        samples = window.samples
        offset = window.offset_seconds
        self.calls.append((len(samples), float(offset)))
        return ModelOutput(
            raw_score=self.raw_score,
            calibrated_probability=self.probability,
            model_id=self.model_id,
            model_revision="test-revision",
            window="10s@16000Hz",
            warnings=(),
            timings=Timings(score_seconds=0.001),
        )

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self.model_id,
            revision="test-revision",
            window_seconds=10.0,
            sample_rate=SAMPLE_RATE,
        )


def silent_window(seconds: float = 10.0, offset: float = 0.0) -> NDArray[np.float32]:
    """Allocate a finite, in-range window without any model or audio I/O."""

    return np.zeros(int(round(seconds * SAMPLE_RATE)), dtype=np.float32)
