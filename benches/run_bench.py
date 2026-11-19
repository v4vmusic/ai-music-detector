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
from typing import Protocol
from urllib.request import urlopen

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
    ModelSpec,
)
from benches._shared.types import AudioWindow, ModelMetadata, ModelOutput  # noqa: E402
from benches.backends.fakeprint import load as load_fakeprint  # noqa: E402

__all__ = ["main", "run"]

DEFAULT_SEED = 42
MAX_WEIGHTS_BYTES = 1_000_000
VERSION_NAMES = ("numpy", "onnxruntime")


class Scorer(Protocol):
    """Backend surface the runner needs."""

    def score(self, window: AudioWindow) -> ModelOutput:
        """Score one shared audio window."""

    def metadata(self) -> ModelMetadata:
        """Describe the loaded model."""


def _package_versions() -> dict[str, str]:
    return {name: version(name) for name in VERSION_NAMES}


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
) -> int:
    """Score every local manifest entry, writing one JSONL row per entry."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = backend.metadata()
    versions = _package_versions()
    config_hash = _config_hash(metadata, seed, versions)

    with out_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            try:
                window = decode_window(entry.path, window_seconds=metadata.window_seconds)
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a labeled manifest with one bench backend.")
    parser.add_argument("--backend", required=True, choices=MODEL_KEYS)
    parser.add_argument("--manifest", required=True, type=Path, help="path,label CSV manifest")
    parser.add_argument("--out", required=True, type=Path, help="raw JSONL output path")
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int, help="run seed (default: 42)")
    args = parser.parse_args(argv)

    try:
        if args.backend != MODEL_KEY_FAKEPRINT:
            raise ModelLoadError(f"No runner wiring for backend {args.backend!r}")
        entries = load_manifest(args.manifest)
        backend = load_fakeprint(model_bytes=_load_weights(FAKEPRINT_SPEC))
        count = run(entries, args.out, backend, seed=args.seed)
    except (AidetectError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {count} rows to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
