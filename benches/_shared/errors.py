from __future__ import annotations

__all__ = ["AidetectError", "AudioDecodeError", "InvalidManifestError", "ModelLoadError"]


class AidetectError(Exception):
    """Base class for expected detector and bench failures."""


class AudioDecodeError(AidetectError):
    """Audio cannot be decoded or does not satisfy the shared input contract."""


class InvalidManifestError(AidetectError):
    """The labeled CSV manifest is missing, malformed or invalid."""


class ModelLoadError(AidetectError):
    """A model session or its weights do not satisfy the backend contract."""
