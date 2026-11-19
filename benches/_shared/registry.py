from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from benches._shared.constants import SAMPLE_RATE

MODEL_KEY_FAKEPRINT: Final = "fakeprint"
MODEL_KEYS: Final[tuple[str, ...]] = (MODEL_KEY_FAKEPRINT,)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One registered model's identity, weights and preprocessing parameters."""

    key: str
    model_id: str
    revision: str
    window_seconds: float
    sample_rate: int
    weights_file: str
    weights_sha256: str
    feature_count: int
    n_fft: int
    freq_min_hz: int
    freq_max_hz: int
    hull_area_bins: int
    min_db: float
    max_db: float
    preprocessing_version: str = "1"


FAKEPRINT_SPEC: Final = ModelSpec(
    key=MODEL_KEY_FAKEPRINT,
    model_id="lofcz/ai-music-detector",
    revision="d2180598fed79e3f917e8050a00439982466e5c6",
    window_seconds=10.0,
    sample_rate=SAMPLE_RATE,
    weights_file="ai_music_detector.onnx",
    weights_sha256="af7a75c6ed457bc5b6941c8bc76aa06a66d48de40db944b761ed2bebfc0fbbd3",
    feature_count=3585,
    n_fft=8192,
    freq_min_hz=1_000,
    freq_max_hz=8_000,
    hull_area_bins=10,
    min_db=-45.0,
    max_db=5.0,
)

__all__ = ["FAKEPRINT_SPEC", "MODEL_KEY_FAKEPRINT", "MODEL_KEYS", "ModelSpec"]
