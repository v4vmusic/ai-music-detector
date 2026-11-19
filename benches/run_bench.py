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
from typing import Any, Final, Protocol, cast
from urllib.request import urlopen

import numpy as np
from numpy.typing import NDArray

# Standalone scripts (CC §10) get the script directory on sys.path, not the workspace
# root; add the root so `benches` imports resolve when invoked as `python benches/run_bench.py`.
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from benches._shared.audio import decode_window  # noqa: E402
from benches._shared.constants import (  # noqa: E402
    FILL_MODES,
    PLACEMENT_FIXED,
    PLACEMENTS,
    SAMPLE_RATE,
    FillMode,
    Placement,
)
from benches._shared.errors import AidetectError, ModelLoadError  # noqa: E402
from benches._shared.experiment import (  # noqa: E402
    BASELINE_RESULTS_DIR,
    DEFAULT_EXPERIMENTS_DIR,
    ExperimentConfig,
    assert_experiment_location,
    experiment_raw_path,
    validate_config_name,
)
from benches._shared.manifest import ManifestEntry, load_manifest  # noqa: E402
from benches._shared.registry import (  # noqa: E402
    AST_SPEC,
    FAKEPRINT_SPEC,
    MODEL_KEY_AST,
    MODEL_KEY_FAKEPRINT,
    MODEL_KEYS,
    SPECTTRA_SPECS,
    AstSpec,
    ModelSpec,
    SpecTTTraSpec,
    get_specttra_spec,
)
from benches._shared.types import AudioWindow, ModelMetadata, ModelOutput  # noqa: E402
from benches.backends.ast import Pipeline  # noqa: E402
from benches.backends.ast import load as load_ast  # noqa: E402
from benches.backends.fakeprint import load as load_fakeprint  # noqa: E402
from benches.backends.spctttra import Classifier  # noqa: E402
from benches.backends.spctttra import load as load_spctttra  # noqa: E402

__all__ = ["DEFAULT_OFFSETS", "compare_from_raw", "main", "run", "run_compare"]

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
AST_VERSION_NAMES: Final[tuple[str, ...]] = (
    "numpy",
    "torch",
    "transformers",
    "tokenizers",
    "huggingface_hub",
)
DEFAULT_OFFSETS: Final[Mapping[str, float]] = MappingProxyType(
    {
        MODEL_KEY_FAKEPRINT: 0.0,
        **{key: spec.default_offset_seconds for key, spec in SPECTTRA_SPECS.items()},
        MODEL_KEY_AST: AST_SPEC.default_offset_seconds,
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


def _config_hash(
    metadata: ModelMetadata, seed: int, versions: Mapping[str, str], config: ExperimentConfig
) -> str:
    payload = {
        "metadata": asdict(metadata),
        "seed": seed,
        "versions": dict(versions),
        "experiment": config.as_dict(),
    }
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
    config: ExperimentConfig | None = None,
) -> int:
    """Score every local manifest entry, writing one JSONL row per entry.

    ``config`` carries tuning placement/fill choices; ``offset_seconds`` remains the fixed-offset
    convenience for callers that do not supply a config.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_config = config or ExperimentConfig(offset_seconds=offset_seconds)
    placement = (
        None
        if experiment_config.placement == PLACEMENT_FIXED
        else cast(Placement, experiment_config.placement)
    )
    metadata = backend.metadata()
    versions = _package_versions(version_names)
    config_hash = _config_hash(metadata, seed, versions, experiment_config)
    experiment = experiment_config.as_dict()

    with out_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            try:
                window = decode_window(
                    entry.path,
                    window_seconds=metadata.window_seconds,
                    offset_seconds=experiment_config.offset_seconds,
                    fill_mode=experiment_config.fill_mode,
                    placement=placement,
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
                    "experiment": experiment,
                }
            else:
                result = backend.score(window)
                row = {
                    "path": str(entry.path),
                    "label": entry.label,
                    **asdict(result),
                    "offset_seconds": window.offset_seconds,
                    "seed": seed,
                    "config_hash": config_hash,
                    "versions": versions,
                    "experiment": experiment,
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


class _HFPipeline:
    """Adapts the HF audio-classification pipeline to the backend's id-keyed surface."""

    def __init__(self, classifier: Any, id2label: Mapping[int, str]) -> None:
        # Any: transformers ships no stubs/py.typed, so there is no bench-side type (CC §3).
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


def _load_ast_pipeline(spec: AstSpec, *, cache_dir: Path | None, device: str) -> Pipeline:
    """Pin the HF AST snapshot and supply its training feature extractor; imports stay lazy."""

    try:
        from transformers import (  # type: ignore[import-untyped]
            ASTFeatureExtractor,
            AutoConfig,
            AutoModelForAudioClassification,
        )
        from transformers import pipeline as hf_pipeline
    except ImportError as exc:
        raise ModelLoadError("AST needs transformers; install it with `uv sync`") from exc

    download_kwargs: dict[str, str] = {"revision": spec.revision}
    if cache_dir is not None:
        download_kwargs["cache_dir"] = str(cache_dir)
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_raw_dir(raw_dir: Path) -> dict[str, list[dict[str, Any]]]:
    from benches._shared.eval import load_raw_rows

    per_backend: dict[str, list[dict[str, Any]]] = {}
    if not raw_dir.exists():
        return per_backend
    allowed = set(MODEL_KEYS)
    for path in sorted(raw_dir.glob("*.jsonl")):
        if path.stem in allowed:
            per_backend[path.stem] = load_raw_rows(path)
    return per_backend


def compare_from_raw(
    raw_dir: Path,
    results_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    k_folds: int = 5,
    backend_status_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from benches._shared.eval import (
        evaluate_model_rows,
        summarize_backends,
        write_calibration,
        write_report,
        write_summary,
    )

    per_backend_rows = _load_raw_dir(raw_dir)
    if results_dir.resolve() == BASELINE_RESULTS_DIR.resolve():
        for rows in per_backend_rows.values():
            if any("experiment" in row for row in rows):
                raise ValueError(
                    "Tuning experiment rows cannot be evaluated into the baseline results "
                    "directory; use --experiment for a separate results location"
                )
    model_calibrations: dict[str, dict[str, Any]] = {}
    model_results: dict[str, dict[str, Any]] = {}
    backend_status: dict[str, dict[str, Any]] = {}
    split_strategy = "none"

    for key in MODEL_KEYS:
        override = dict((backend_status_overrides or {}).get(key, {}))
        rows = per_backend_rows.get(key, [])
        if not rows:
            status = {"ok": False, "error": "no raw results"}
            status.update(override)
            backend_status[key] = status
            model_results[key] = {
                "model_id": None,
                "model_revision": None,
                "config_hash": None,
                "versions": None,
                "timings": {"total_score_seconds": 0.0, "peak_vram_bytes": None},
                "metrics": {},
                "dev_metrics": {},
                "scored_tracks": 0,
                "failed_tracks": 0,
            }
            continue
        evaluation = evaluate_model_rows(rows, seed=seed, k_folds=k_folds)
        split_strategy = str(evaluation.get("split_strategy") or split_strategy)
        scored = int(evaluation.get("scored_tracks") or 0)
        has_scored = scored > 0
        status = {"ok": has_scored}
        if not has_scored:
            status["error"] = "no scored rows"
        status.update(override)
        backend_status[key] = status
        model_results[key] = {
            "model_id": evaluation.get("model_id"),
            "model_revision": evaluation.get("model_revision"),
            "config_hash": evaluation.get("config_hash"),
            "versions": evaluation.get("versions"),
            "timings": evaluation.get(
                "timings", {"total_score_seconds": 0.0, "peak_vram_bytes": None}
            ),
            "metrics": evaluation.get("test_metrics", {}),
            "dev_metrics": evaluation.get("dev_metrics", {}),
            "scored_tracks": evaluation.get("scored_tracks", 0),
            "failed_tracks": evaluation.get("failed_tracks", 0),
        }
        model_calibrations[key] = {
            "model_id": evaluation.get("model_id"),
            "model_revision": evaluation.get("model_revision"),
            "config_hash": evaluation.get("config_hash"),
            "calibration": evaluation["calibration"],
            "dev_metrics": evaluation.get("dev_metrics", {}),
        }

    summary = summarize_backends(
        per_backend_rows,
        backend_status=backend_status,
        seed=seed,
        split_strategy=split_strategy,
        model_results=model_results,
    )

    calibration_path = results_dir / "calibration.json"
    summary_path = results_dir / "summary.json"
    report_path = results_dir / "report.md"
    write_calibration(
        calibration_path,
        model_calibrations=model_calibrations,
        seed=seed,
        split_strategy=split_strategy,
    )
    write_summary(summary_path, summary)
    calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    write_report(report_path, summary, calibration_payload)
    return summary, calibration_payload


def _default_backend_loader(key: str, device: str) -> tuple[Scorer, Sequence[str]]:
    if key == MODEL_KEY_FAKEPRINT:
        return load_fakeprint(model_bytes=_load_weights(FAKEPRINT_SPEC)), VERSION_NAMES
    if key == MODEL_KEY_AST:
        pipeline = _load_ast_pipeline(AST_SPEC, cache_dir=_cache_dir(), device=device)
        return load_ast(MODEL_KEY_AST, pipeline=pipeline), AST_VERSION_NAMES
    spec = get_specttra_spec(key)
    classifier = _load_spctttra_classifier(spec, cache_dir=_cache_dir(), device=device)
    return load_spctttra(key, classifier=classifier), SPECTTRA_VERSION_NAMES


def run_compare(
    entries: Sequence[ManifestEntry],
    results_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    device: str = "cuda",
    backend_loader: Any = None,
    k_folds: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    loader = backend_loader or _default_backend_loader
    backend_status_overrides: dict[str, dict[str, Any]] = {}

    for key in MODEL_KEYS:
        out_path = raw_dir / f"{key}.jsonl"
        try:
            backend, version_names = loader(key, device)
            offset = DEFAULT_OFFSETS[key]
            run(
                entries,
                out_path,
                backend,
                seed=seed,
                offset_seconds=offset,
                version_names=version_names,
            )
            unload = getattr(backend, "unload", None)
            if callable(unload):
                unload()
            backend_status_overrides[key] = {"ok": True}
        except Exception as exc:  # noqa: BLE001 - record and continue with other backends
            backend_status_overrides[key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            # Ensure a raw file exists for provenance even on total failure.
            if not out_path.exists():
                out_path.write_text("", encoding="utf-8")

    return compare_from_raw(
        raw_dir,
        results_dir,
        seed=seed,
        k_folds=k_folds,
        backend_status_overrides=backend_status_overrides,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a labeled manifest with bench backends.")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="evaluate available backends and write calibration/summary/report",
    )
    parser.add_argument("--backend", choices=MODEL_KEYS, help="single backend to score")
    parser.add_argument("--manifest", required=True, type=Path, help="path,label CSV manifest")
    parser.add_argument("--out", type=Path, help="raw JSONL output path for single-backend mode")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="directory for raw/, calibration.json, summary.json and report.md "
        "(compare mode; default: benches/results)",
    )
    parser.add_argument(
        "--from-raw",
        type=Path,
        default=None,
        help="evaluate existing raw JSONL files from this directory instead of running models",
    )
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int, help="run seed (default: 42)")
    parser.add_argument(
        "--folds",
        type=int,
        default=None,
        help="k for stratified k-fold when classes are small (default: 5)",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=None,
        help="fixed window offset in seconds (default: per-backend value; not with --placement)",
    )
    parser.add_argument(
        "--placement",
        choices=PLACEMENTS,
        default=PLACEMENT_FIXED,
        help="per-track window placement: start/middle/end are duration-relative (default: fixed)",
    )
    parser.add_argument(
        "--fill",
        choices=FILL_MODES,
        default="zero-pad",
        help="short-segment fill: zero-pad (baseline) or loop-fill (default: zero-pad)",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="tuning config name; writes under --experiments-dir instead of baseline results",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=DEFAULT_EXPERIMENTS_DIR,
        help="root for tuning outputs (default: benches/results/tuning)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="torch device for GPU backends (default: cuda)",
    )
    args = parser.parse_args(argv)

    if args.compare:
        if args.backend or args.out:
            print("error: --compare does not take --backend or --out", file=sys.stderr)
            return 2
        if args.placement != PLACEMENT_FIXED or args.fill != "zero-pad":
            parser.error("--placement/--fill apply to single-backend runs, not --compare")
        k_folds = args.folds if args.folds is not None else 5
        try:
            if args.experiment:
                if args.results_dir is not None:
                    parser.error("--experiment and --results-dir are mutually exclusive")
                results_dir = args.experiments_dir / validate_config_name(args.experiment)
                assert_experiment_location(results_dir)
                raw_dir = args.from_raw if args.from_raw is not None else results_dir / "raw"
                if args.from_raw is not None:
                    assert_experiment_location(raw_dir)
            else:
                results_dir = args.results_dir or BASELINE_RESULTS_DIR
                raw_dir = args.from_raw if args.from_raw is not None else results_dir / "raw"
            if args.from_raw:
                summary, _calibration = compare_from_raw(
                    raw_dir, results_dir, seed=args.seed, k_folds=k_folds
                )
            else:
                entries = load_manifest(args.manifest)
                summary, _calibration = run_compare(
                    entries,
                    results_dir,
                    seed=args.seed,
                    device=args.device,
                    k_folds=k_folds,
                )
        except (AidetectError, ValueError, OSError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        ok = sum(1 for b in summary.get("backends", {}).values() if b.get("ok"))
        print(
            f"wrote calibration/summary/report to {results_dir} ({ok} backends with scored rows)",
            file=sys.stderr,
        )
        return 0

    if not args.backend:
        parser.error("the following arguments are required: --backend")
    if args.experiment and args.out is not None:
        parser.error("--experiment and --out are mutually exclusive")
    if not args.experiment and (args.placement != PLACEMENT_FIXED or args.fill != "zero-pad"):
        parser.error("--experiment is required for tuning configs (--placement/--fill)")
    if args.placement != PLACEMENT_FIXED and args.offset is not None:
        parser.error("--offset cannot be combined with --placement start/middle/end")

    try:
        if args.experiment:
            out = experiment_raw_path(args.experiments_dir, args.experiment, args.backend)
            assert_experiment_location(out)
        else:
            if args.out is None:
                parser.error("the following arguments are required: --out")
            out = args.out
        entries = load_manifest(args.manifest)
        if args.backend == MODEL_KEY_FAKEPRINT:
            backend: Scorer = load_fakeprint(model_bytes=_load_weights(FAKEPRINT_SPEC))
            version_names = VERSION_NAMES
        elif args.backend == MODEL_KEY_AST:
            pipeline = _load_ast_pipeline(AST_SPEC, cache_dir=_cache_dir(), device=args.device)
            backend = load_ast(MODEL_KEY_AST, pipeline=pipeline)
            version_names = AST_VERSION_NAMES
        else:
            spec = get_specttra_spec(args.backend)
            classifier = _load_spctttra_classifier(spec, cache_dir=_cache_dir(), device=args.device)
            backend = load_spctttra(args.backend, classifier=classifier)
            version_names = SPECTTRA_VERSION_NAMES
        offset = DEFAULT_OFFSETS[args.backend] if args.offset is None else args.offset
        if args.placement != PLACEMENT_FIXED:
            offset = 0.0
        config = ExperimentConfig(
            name=args.experiment,
            placement=cast(Placement, args.placement),
            offset_seconds=offset,
            fill_mode=cast(FillMode, args.fill),
        )
        count = run(
            entries,
            out,
            backend,
            seed=args.seed,
            version_names=version_names,
            config=config,
        )
    except (AidetectError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {count} rows to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
