from __future__ import annotations

__all__ = [
    "AidetectError",
    "AudioDecodeError",
    "InvalidManifestError",
    "ModelLoadError",
    "NetworkAudioError",
]


class AidetectError(Exception):
    """Base class for expected detector failures."""


class AudioDecodeError(AidetectError):
    """Audio cannot be decoded or does not satisfy the shared input contract."""


class InvalidManifestError(AidetectError):
    """The labeled CSV manifest is missing, malformed or invalid."""


class ModelLoadError(AidetectError):
    """A model session, weights or config do not satisfy the backend contract."""


class NetworkAudioError(AidetectError):
    """A remote audio source is unsafe, unreachable or not usable audio."""
