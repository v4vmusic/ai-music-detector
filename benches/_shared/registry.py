from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from benches._shared.constants import SAMPLE_RATE
from benches._shared.errors import ModelLoadError

MODEL_KEY_FAKEPRINT: Final = "fakeprint"
MODEL_KEY_SPECTTRA_ALPHA: Final = "spectttra-alpha-120s"
MODEL_KEY_SPECTTRA_BETA: Final = "spectttra-beta-120s"
MODEL_KEY_SPECTTRA_GAMMA: Final = "spectttra-gamma-120s"
SPECTTRA_DEFAULT_OFFSET_SECONDS: Final = 25.0
MODEL_KEYS: Final[tuple[str, ...]] = (
    MODEL_KEY_FAKEPRINT,
    MODEL_KEY_SPECTTRA_ALPHA,
    MODEL_KEY_SPECTTRA_BETA,
    MODEL_KEY_SPECTTRA_GAMMA,
)


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


@dataclass(frozen=True, slots=True)
class SpecTTTraSpec:
    """Identity and preprocessing facts for one SONICS SpecTTTra checkpoint."""

    key: str
    model_id: str
    revision: str
    window_seconds: float
    sample_rate: int
    default_offset_seconds: float
    preprocessing_version: str


def _specttra_spec(variant: str, revision: str) -> SpecTTTraSpec:
    return SpecTTTraSpec(
        key=variant,
        model_id=f"awsaf49/sonics-{variant}",
        revision=revision,
        window_seconds=120.0,
        sample_rate=SAMPLE_RATE,
        default_offset_seconds=SPECTTRA_DEFAULT_OFFSET_SECONDS,
        preprocessing_version="sonics-1",
    )


SPECTTRA_SPECS: Final[Mapping[str, SpecTTTraSpec]] = MappingProxyType(
    {
        spec.key: spec
        for spec in (
            _specttra_spec(MODEL_KEY_SPECTTRA_ALPHA, "094b32a5545098a71c113f6ae9d5c55310564268"),
            _specttra_spec(MODEL_KEY_SPECTTRA_BETA, "cc49422a98b5b92700876c608d0f9cdf27214a3e"),
            _specttra_spec(MODEL_KEY_SPECTTRA_GAMMA, "57659d8cb04767893537bbeb89da973e2e5a0290"),
        )
    }
)


def get_specttra_spec(key: str) -> SpecTTTraSpec:
    """Look up a registered variant; unknown keys fail with an actionable error."""

    spec = SPECTTRA_SPECS.get(key)
    if spec is None:
        raise ModelLoadError(
            f"Unknown SpecTTTra variant {key!r}; expected one of {tuple(SPECTTRA_SPECS)}"
        )
    return spec


__all__ = [
    "FAKEPRINT_SPEC",
    "MODEL_KEY_FAKEPRINT",
    "MODEL_KEY_SPECTTRA_ALPHA",
    "MODEL_KEY_SPECTTRA_BETA",
    "MODEL_KEY_SPECTTRA_GAMMA",
    "MODEL_KEYS",
    "SPECTTRA_DEFAULT_OFFSET_SECONDS",
    "SPECTTRA_SPECS",
    "ModelSpec",
    "SpecTTTraSpec",
    "get_specttra_spec",
]
