from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from aidetect.constants import (
    AI_GENERATED_LABEL,
    MODEL_KEY_AST,
    MODEL_KEY_FAKEPRINT,
    MODEL_KEY_SPECTTRA_ALPHA,
    MODEL_KEY_SPECTTRA_BETA,
    MODEL_KEY_SPECTTRA_GAMMA,
    SAMPLE_RATE,
)
from aidetect.errors import ModelLoadError

SPECTTRA_DEFAULT_OFFSET_SECONDS: Final = 25.0
AST_DEFAULT_OFFSET_SECONDS: Final = 25.0
MODEL_KEYS: Final[tuple[str, ...]] = (
    MODEL_KEY_FAKEPRINT,
    MODEL_KEY_SPECTTRA_ALPHA,
    MODEL_KEY_SPECTTRA_BETA,
    MODEL_KEY_SPECTTRA_GAMMA,
    MODEL_KEY_AST,
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


@dataclass(frozen=True, slots=True)
class AstSpec:
    """Identity, window and positive label for the AST audio-classification checkpoint."""

    key: str
    model_id: str
    revision: str
    window_seconds: float
    sample_rate: int
    default_offset_seconds: float
    positive_label: str
    preprocessing_version: str
    feature_max_length: int
    feature_num_mel_bins: int
    feature_mean: float
    feature_std: float


AST_SPEC: Final = AstSpec(
    key=MODEL_KEY_AST,
    model_id="AI-Music-Detection/ai_music_detection_large_60s",
    revision="5e29461bbfc41dc5fe4ed5f7fadc961aa84a7e3e",
    window_seconds=60.0,
    sample_rate=SAMPLE_RATE,
    default_offset_seconds=AST_DEFAULT_OFFSET_SECONDS,
    positive_label=AI_GENERATED_LABEL,
    preprocessing_version="ast-1",
    feature_max_length=6_000,
    feature_num_mel_bins=128,
    feature_mean=-4.2677393,
    feature_std=4.5689974,
)


def get_ast_spec(key: str) -> AstSpec:
    """Look up the AST backend key; unknown keys fail with an actionable error."""

    if key != AST_SPEC.key:
        raise ModelLoadError(f"Unknown AST model {key!r}; expected {AST_SPEC.key!r}")
    return AST_SPEC


__all__ = [
    "AST_DEFAULT_OFFSET_SECONDS",
    "AST_SPEC",
    "FAKEPRINT_SPEC",
    "MODEL_KEY_AST",
    "MODEL_KEY_FAKEPRINT",
    "MODEL_KEY_SPECTTRA_ALPHA",
    "MODEL_KEY_SPECTTRA_BETA",
    "MODEL_KEY_SPECTTRA_GAMMA",
    "MODEL_KEYS",
    "SPECTTRA_DEFAULT_OFFSET_SECONDS",
    "SPECTTRA_SPECS",
    "AstSpec",
    "ModelSpec",
    "SpecTTTraSpec",
    "get_ast_spec",
    "get_specttra_spec",
]
