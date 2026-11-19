from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Protocol
from urllib.request import urlopen

import numpy as np
from numpy.typing import NDArray

# Standalone scripts (CC §10) get the script directory on sys.path, not the workspace
# root; add the root so `benches` imports resolve when invoked as `python benches/run_bench.py`.
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from benches._shared.audio import decode_window  # noqa: E402
from benches._shared.errors import AidetectError, ModelLoadError  # noqa: E402
from benches._shared.manifest import ManifestEntry, load_manifest  # noqa: E402
from benches._shared.registry import (  # noqa: E402
    FAKEPRINT_SPEC,
    MODEL_KEY_FAKEPRINT,
    MODEL_KEYS,
    SPECTTRA_SPECS,
    ModelSpec,
    SpecTTTraSpec,
    get_specttra_spec,
)
from benches._shared.types import AudioWindow, ModelMetadata, ModelOutput  # noqa: E402
from benches.backends.fakeprint import load as load_fakeprint  # noqa: E402
from benches.backends.spctttra import Classifier  # noqa: E402
from benches.backends.spctttra import load as load_spctttra  # noqa: E402

__all__ = ["DEFAULT_OFFSETS", "main", "run"]

DEFAULT_SEED = 42
MAX_WEIGHTS_BYTES = 1_000_000
VERSION_NAMES: Final[tuple[str, ...]] = ("numpy", "onnxruntime")
SPECTTRA_VERSION_NAMES: Final[tuple[str, ...]] = (
    "numpy",
    "torch",
    "torchaudio",
    "sonics",
    "huggingface_hub",
)
DEFAULT_OFFSETS: Final[Mapping[str, float]] = MappingProxyType(
    {
        MODEL_KEY_FAKEPRINT: 0.0,
        **{key: spec.default_offset_seconds for key, spec in SPECTTRA_SPECS.items()},
    }
)


class Scorer(Protocol):
    """Backend surface the runner needs."""

    def score(self, window: AudioWindow) -> ModelOutput:
        """Score one shared audio window."""

    def metadata(self) -> ModelMetadata:
        """Describe the loaded model."""


def _package_versions(names: Sequence[str] = VERSION_NAMES) -> dict[str, str]:
    return {name: version(name) for name in names}


def _config_hash(metadata: ModelMetadata, seed: int, versions: Mapping[str, str]) -> str:
    payload = {"metadata": asdict(metadata), "seed": seed, "versions": dict(versions)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run(
    entries: Sequence[ManifestEntry],
    out_path: Path,
    backend: Scorer,
    *,
    seed: int = DEFAULT_SEED,
    offset_seconds: float = 0.0,
    version_names: Sequence[str] = VERSION_NAMES,
) -> int:
    """Score every local manifest entry, writing one JSONL row per entry."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = backend.metadata()
    versions = _package_versions(version_names)
    config_hash = _config_hash(metadata, seed, versions)

    with out_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            try:
                window = decode_window(
                    entry.path,
                    window_seconds=metadata.window_seconds,
                    offset_seconds=offset_seconds,
                )
            except (AidetectError, ValueError, OSError) as exc:
                row: dict[str, object] = {
                    "path": str(entry.path),
                    "label": entry.label,
                    "warnings": [f"decode_failed: {exc}"],
                    "error": str(exc),
                    "seed": seed,
                    "config_hash": config_hash,
                    "versions": versions,
                }
            else:
                result = backend.score(window)
                row = {
                    "path": str(entry.path),
                    "label": entry.label,
                    **asdict(result),
                    "seed": seed,
                    "config_hash": config_hash,
                    "versions": versions,
                }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return len(entries)


def _cache_dir() -> Path:
    configured = os.environ.get("AIDETECT_CACHE_DIR")
    return Path(configured) if configured else Path.home() / ".cache" / "huggingface" / "hub"


def _weights_path(spec: ModelSpec) -> Path:
    repo_cache = f"models--{spec.model_id.replace('/', '--')}"
    return _cache_dir() / repo_cache / "snapshots" / spec.revision / spec.weights_file


def _load_weights(spec: ModelSpec) -> bytes:
    """Fetch the pinned weights once into AIDETECT_CACHE_DIR and verify their digest."""

    path = _weights_path(spec)
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
    """Wraps a sonics classifier; torch/sonics imports stay inside the loader below."""

    def __init__(self, model: Any, device: str, torch_module: Any) -> None:
        # Any: torch/sonics are imported lazily and have no bench-side stubs (CC §3).
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


def _load_spctttra_classifier(
    spec: SpecTTTraSpec, *, cache_dir: Path | None, device: str
) -> Classifier:
    """Download pinned weights once into the HF cache and build a torch classifier."""

    try:
        import torch
        from huggingface_hub import hf_hub_download
        from sonics import HFAudioClassifier  # type: ignore[import-untyped]  # no py.typed upstream
    except ImportError as exc:
        raise ModelLoadError(
            "SpecTTTra needs sonics, torch and huggingface_hub; install them with `uv sync`"
        ) from exc

    def download(filename: str) -> Path:
        if cache_dir is None:
            downloaded = hf_hub_download(
                repo_id=spec.model_id, filename=filename, revision=spec.revision
            )
        else:
            downloaded = hf_hub_download(
                repo_id=spec.model_id,
                filename=filename,
                revision=spec.revision,
                cache_dir=str(cache_dir),
            )
        return Path(downloaded)

    try:
        config_path = download("config.json")
        weights_path = download("pytorch_model.bin")
    except Exception as exc:
        raise ModelLoadError(f"Cannot download {spec.model_id}@{spec.revision}: {exc}") from exc

    # sonics' from_pretrained takes a local directory and does not accept a revision;
    # both files came from the same pinned snapshot, so use that snapshot directory.
    if config_path.parent != weights_path.parent:
        raise ModelLoadError(f"Pinned snapshot for {spec.model_id}@{spec.revision} is inconsistent")
    try:
        model = HFAudioClassifier.from_pretrained(str(config_path.parent), map_location="cpu")
        model.to(device)
        model.eval()
    except Exception as exc:
        raise ModelLoadError(f"Cannot load {spec.model_id}@{spec.revision}: {exc}") from exc
    return _HFClassifier(model=model, device=device, torch_module=torch)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a labeled manifest with one bench backend.")
    parser.add_argument("--backend", required=True, choices=MODEL_KEYS)
    parser.add_argument("--manifest", required=True, type=Path, help="path,label CSV manifest")
    parser.add_argument("--out", required=True, type=Path, help="raw JSONL output path")
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int, help="run seed (default: 42)")
    parser.add_argument(
        "--offset",
        type=float,
        default=None,
        help="window offset in seconds (default: per-backend value)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="torch device for GPU backends (default: cuda)",
    )
    args = parser.parse_args(argv)

    try:
        entries = load_manifest(args.manifest)
        if args.backend == MODEL_KEY_FAKEPRINT:
            backend: Scorer = load_fakeprint(model_bytes=_load_weights(FAKEPRINT_SPEC))
            version_names = VERSION_NAMES
        else:
            spec = get_specttra_spec(args.backend)
            classifier = _load_spctttra_classifier(spec, cache_dir=_cache_dir(), device=args.device)
            backend = load_spctttra(args.backend, classifier=classifier)
            version_names = SPECTTRA_VERSION_NAMES
        offset = DEFAULT_OFFSETS[args.backend] if args.offset is None else args.offset
        count = run(
            entries,
            args.out,
            backend,
            seed=args.seed,
            offset_seconds=offset,
            version_names=version_names,
        )
    except (AidetectError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {count} rows to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
