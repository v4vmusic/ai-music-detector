from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.request import urlopen

import numpy as np
from numpy.typing import NDArray

from aidetect.backends.ast import Pipeline
from aidetect.backends.spctttra import Classifier
from aidetect.constants import SAMPLE_RATE
from aidetect.errors import ModelLoadError
from aidetect.registry import AstSpec, ModelSpec, SpecTTTraSpec

__all__ = [
    "MAX_WEIGHTS_BYTES",
    "cache_dir",
    "load_ast_pipeline",
    "load_fakeprint_weights",
    "load_spctttra_classifier",
]

MAX_WEIGHTS_BYTES = 1_000_000


class _TorchModule(Protocol):
    def from_numpy(self, array: NDArray[np.float32]) -> Any: ...

    def inference_mode(self) -> Any: ...

    @property
    def cuda(self) -> Any: ...


def cache_dir() -> Path:
    """Return the shared model cache, honoring ``AIDETECT_CACHE_DIR``."""

    configured = os.environ.get("AIDETECT_CACHE_DIR")
    return Path(configured) if configured else Path.home() / ".cache" / "huggingface" / "hub"


_default_cache_dir = cache_dir


def load_fakeprint_weights(
    spec: ModelSpec, *, cache_dir: Path | None = None, device: str = "cpu"
) -> bytes:
    """Fetch pinned ONNX weights into the shared cache and verify their digest."""

    del device  # CPU-only ONNX backend; the parameter keeps all loaders interchangeable.
    root = cache_dir if cache_dir is not None else _default_cache_dir()
    repo_cache = f"models--{spec.model_id.replace('/', '--')}"
    path = root / repo_cache / "snapshots" / spec.revision / spec.weights_file
    if path.exists():
        data = path.read_bytes()
    else:
        url = f"https://huggingface.co/{spec.model_id}/resolve/{spec.revision}/{spec.weights_file}"
        with urlopen(url, timeout=30) as response:  # noqa: S310 - pinned HTTPS URL
            data = response.read(MAX_WEIGHTS_BYTES + 1)
        if len(data) > MAX_WEIGHTS_BYTES:
            raise ModelLoadError(
                f"Weights for {spec.model_id} exceed the {MAX_WEIGHTS_BYTES} byte cap"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    digest = hashlib.sha256(data).hexdigest()
    if digest != spec.weights_sha256:
        raise ModelLoadError(
            f"Weights digest mismatch for {spec.model_id}: expected {spec.weights_sha256}, "
            f"got {digest}"
        )
    return data


class _HFClassifier:
    """Wraps a sonics classifier; torch/sonics imports stay inside the loader."""

    def __init__(self, model: Any, device: str, torch_module: _TorchModule) -> None:
        # Any: torch/sonics are imported lazily and have no library-side stubs (CC §3).
        self._model = model
        self._device = device
        self._torch = torch_module

    def __call__(self, samples: NDArray[np.float32]) -> float:
        scale = max(float(np.std(samples)), 1e-6)
        scaled = (samples / np.float32(scale)).astype(np.float32, copy=False)
        tensor = self._torch.from_numpy(np.ascontiguousarray(scaled)).unsqueeze(0).to(self._device)
        with self._torch.inference_mode():
            logits = self._model(tensor)
        return float(logits.reshape(-1)[0].item())

    def unload(self) -> None:
        del self._model
        if self._device.startswith("cuda") and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def load_spctttra_classifier(
    spec: SpecTTTraSpec, *, cache_dir: Path | None = None, device: str = "cpu"
) -> Classifier:
    """Download pinned weights once into the shared cache and build a torch classifier."""

    try:
        import torch
        from huggingface_hub import hf_hub_download
        from sonics import HFAudioClassifier  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ModelLoadError(
            "SpecTTTra needs sonics, torch and huggingface_hub; install the `full` or `gpu` extra"
        ) from exc

    def download(filename: str) -> Path:
        target = cache_dir if cache_dir is not None else _default_cache_dir()
        downloaded = hf_hub_download(
            repo_id=spec.model_id,
            filename=filename,
            revision=spec.revision,
            cache_dir=str(target),
        )
        return Path(downloaded)

    try:
        config_path = download("config.json")
        weights_path = download("pytorch_model.bin")
    except Exception as exc:
        raise ModelLoadError(f"Cannot download {spec.model_id}@{spec.revision}: {exc}") from exc

    if config_path.parent != weights_path.parent:
        raise ModelLoadError(f"Pinned snapshot for {spec.model_id}@{spec.revision} is inconsistent")
    try:
        model = HFAudioClassifier.from_pretrained(str(config_path.parent), map_location="cpu")
        model.to(device)
        model.eval()
    except Exception as exc:
        raise ModelLoadError(f"Cannot load {spec.model_id}@{spec.revision}: {exc}") from exc
    return _HFClassifier(model=model, device=device, torch_module=cast(_TorchModule, torch))


class _HFPipeline:
    """Adapts the HF audio-classification pipeline to the backend's id-keyed surface."""

    def __init__(self, classifier: Any, id2label: Mapping[int, str]) -> None:
        # Any: transformers ships no stubs/py.typed, so there is no library-side type (CC §3).
        self._classifier = classifier
        self._id2label = id2label

    def __call__(self, samples: NDArray[np.float32], *, sampling_rate: int) -> Mapping[int, float]:
        predictions = self._classifier(samples, sampling_rate=sampling_rate, top_k=None)
        by_label: dict[str, float] = {}
        for item in predictions:
            label, score = item["label"], item["score"]
            if not isinstance(label, str) or not isinstance(score, (int, float)):
                raise ModelLoadError("HF AST pipeline returned an invalid prediction")
            by_label[label] = float(score)
        return {
            index: by_label[label] for index, label in self._id2label.items() if label in by_label
        }

    def id2label(self) -> Mapping[int, str]:
        return self._id2label

    def unload(self) -> None:
        del self._classifier


def load_ast_pipeline(
    spec: AstSpec, *, cache_dir: Path | None = None, device: str = "cpu"
) -> Pipeline:
    """Pin the HF AST snapshot and supply its training feature extractor."""

    try:
        from transformers import (  # type: ignore[import-untyped]
            ASTFeatureExtractor,
            AutoConfig,
            AutoModelForAudioClassification,
        )
        from transformers import pipeline as hf_pipeline
    except ImportError as exc:
        raise ModelLoadError("AST needs transformers; install the `ast` or `full` extra") from exc

    download_kwargs: dict[str, str] = {"revision": spec.revision}
    target = cache_dir if cache_dir is not None else _default_cache_dir()
    download_kwargs["cache_dir"] = str(target)
    try:
        config = AutoConfig.from_pretrained(spec.model_id, **download_kwargs)
        feature_extractor = ASTFeatureExtractor(
            num_mel_bins=spec.feature_num_mel_bins,
            sampling_rate=SAMPLE_RATE,
            max_length=spec.feature_max_length,
            mean=spec.feature_mean,
            std=spec.feature_std,
        )
        model = AutoModelForAudioClassification.from_pretrained(spec.model_id, **download_kwargs)
        classifier = hf_pipeline(
            "audio-classification",
            model=model,
            config=config,
            feature_extractor=feature_extractor,
            device=0 if device.startswith("cuda") else -1,
        )
    except Exception as exc:
        raise ModelLoadError(f"Cannot load {spec.model_id}@{spec.revision}: {exc}") from exc

    labels = getattr(config, "id2label", None)
    if not isinstance(labels, Mapping) or not labels:
        raise ModelLoadError(f"No id2label in {spec.model_id}@{spec.revision}")
    id2label = {int(index): str(label) for index, label in labels.items()}
    return _HFPipeline(classifier=classifier, id2label=id2label)
