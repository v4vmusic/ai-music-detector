from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from benches._shared.constants import AI_GENERATED_LABEL, HUMAN_LABEL, Label

__all__ = [
    "Calibration",
    "EvalFold",
    "EvalSample",
    "EvalSplit",
    "calibration_fallback",
    "compute_binary_metrics",
    "expected_calibration_error",
    "fit_temperature_and_thresholds",
    "log_loss",
    "make_split",
    "pr_auc",
    "roc_auc",
    "summarize_backends",
    "temperature_scale_scores",
    "write_calibration",
]

SMALL_CLASS_THRESHOLD = 50
DEFAULT_K_FOLDS = 5
FIXED_DEV_FRACTION = 0.8
FALLBACK_LOW_THRESHOLD = 0.35
FALLBACK_HIGH_THRESHOLD = 0.65
PROB_EPS = 1e-7


@dataclass(frozen=True, slots=True)
class EvalSample:
    path: str
    label: Label
    raw_score: float


@dataclass(frozen=True, slots=True)
class EvalFold:
    dev: list[EvalSample]
    test: list[EvalSample]


@dataclass(frozen=True, slots=True)
class EvalSplit:
    strategy: str
    folds: list[EvalFold]
    k_folds: int | None = None


@dataclass(frozen=True, slots=True)
class Calibration:
    temperature: float
    low_threshold: float
    high_threshold: float
    warnings: tuple[str, ...] = ()


def _stratified_k_fold(samples: Sequence[EvalSample], *, seed: int, k: int) -> list[EvalFold]:
    by_label: dict[Label, list[EvalSample]] = {HUMAN_LABEL: [], AI_GENERATED_LABEL: []}
    for sample in samples:
        by_label[sample.label].append(sample)
    rng = random.Random(seed)
    for label_samples in by_label.values():
        rng.shuffle(label_samples)

    folds_test: list[list[EvalSample]] = [[] for _ in range(k)]
    for label_samples in by_label.values():
        for index, sample in enumerate(label_samples):
            folds_test[index % k].append(sample)

    all_samples = list(samples)
    folds: list[EvalFold] = []
    for test_samples in folds_test:
        test_keys = {sample.path for sample in test_samples}
        dev_samples = [sample for sample in all_samples if sample.path not in test_keys]
        folds.append(EvalFold(dev=dev_samples, test=test_samples))
    return folds


def _fixed_dev_test(samples: Sequence[EvalSample], *, seed: int) -> list[EvalFold]:
    by_label: dict[Label, list[EvalSample]] = {HUMAN_LABEL: [], AI_GENERATED_LABEL: []}
    for sample in samples:
        by_label[sample.label].append(sample)
    rng = random.Random(seed)
    for label_samples in by_label.values():
        rng.shuffle(label_samples)

    dev: list[EvalSample] = []
    test: list[EvalSample] = []
    for label_samples in by_label.values():
        n = len(label_samples)
        if n <= 1:
            dev_count = n
        else:
            dev_count = int(round(n * FIXED_DEV_FRACTION))
            dev_count = max(1, min(dev_count, n - 1))
        dev.extend(label_samples[:dev_count])
        test.extend(label_samples[dev_count:])
    return [EvalFold(dev=dev, test=test)]


def make_split(
    samples: Sequence[EvalSample], seed: int = 42, k_folds: int = DEFAULT_K_FOLDS
) -> EvalSplit:
    if k_folds < 2:
        raise ValueError("k_folds must be at least 2")
    counts = {HUMAN_LABEL: 0, AI_GENERATED_LABEL: 0}
    for sample in samples:
        counts[sample.label] += 1
    min_count = min(counts.values())
    if min_count < SMALL_CLASS_THRESHOLD:
        effective_k = min(k_folds, min_count)
        if effective_k < 2:
            # Too few samples per class for a meaningful fold split; use a fixed split.
            folds = _fixed_dev_test(samples, seed=seed)
            return EvalSplit(strategy="fixed_dev_test", folds=folds, k_folds=None)
        folds = _stratified_k_fold(samples, seed=seed, k=effective_k)
        return EvalSplit(strategy="stratified_k_fold", folds=folds, k_folds=effective_k)
    folds = _fixed_dev_test(samples, seed=seed)
    return EvalSplit(strategy="fixed_dev_test", folds=folds, k_folds=None)


def _as_arrays(
    labels: Sequence[int], scores: Sequence[float]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    y = np.asarray(labels, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError("labels and scores must have the same length")
    return y, s


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    y, s = _as_arrays(labels, scores)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    # Average ranks for ties (1-based ranks).
    i = 0
    n = len(s)
    while i < n:
        j = i + 1
        while j < n and s[order[j]] == s[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    sum_pos_ranks = float(ranks[pos].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def pr_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    y, s = _as_arrays(labels, scores)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = 0.0
    fp = 0.0
    prev_recall = 0.0
    area = 0.0
    for label in y_sorted:
        if label == 1:
            tp += 1.0
        else:
            fp += 1.0
        recall = tp / n_pos
        precision = tp / (tp + fp)
        area += precision * (recall - prev_recall)
        prev_recall = recall
    return float(area)


def f1_at_threshold(labels: Sequence[int], scores: Sequence[float], threshold: float) -> float:
    y, s = _as_arrays(labels, scores)
    pred = s >= threshold
    tp = float(np.sum(pred & (y == 1)))
    fp = float(np.sum(pred & (y == 0)))
    fn = float(np.sum(~pred & (y == 1)))
    denom = 2 * tp + fp + fn
    if denom == 0.0:
        return 0.0
    return float((2 * tp) / denom)


def fpr_at_recall(
    labels: Sequence[int], scores: Sequence[float], target_recall: float = 0.95
) -> float:
    y, s = _as_arrays(labels, scores)
    positives = y == 1
    negatives = y == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    thresholds = np.unique(s)
    # Try thresholds from high to low to find recall >= target with smallest predicted-positive set?
    # We want FPR at ~95% recall: choose threshold achieving recall >= target (or closest above).
    best_fpr = 1.0
    achieved = False
    for threshold in thresholds:
        pred = s >= float(threshold)
        recall = float(np.sum(pred & positives)) / n_pos
        if recall + 1e-12 >= target_recall:
            fpr = float(np.sum(pred & negatives)) / n_neg
            best_fpr = min(best_fpr, fpr)
            achieved = True
    if achieved:
        return float(best_fpr)
    # If target recall not achievable, return FPR at maximum recall.
    best_recall = -1.0
    fpr_at_best = 1.0
    for threshold in thresholds:
        pred = s >= float(threshold)
        recall = float(np.sum(pred & positives)) / n_pos
        fpr = float(np.sum(pred & negatives)) / n_neg
        if recall > best_recall or (math.isclose(recall, best_recall) and fpr < fpr_at_best):
            best_recall = recall
            fpr_at_best = fpr
    return float(fpr_at_best)


def expected_calibration_error(
    labels: Sequence[int], scores: Sequence[float], bins: int = 10
) -> float:
    y, s = _as_arrays(labels, scores)
    if len(y) == 0:
        return float("nan")
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == bins - 1:
            mask = (s >= lo) & (s <= hi)
        else:
            mask = (s >= lo) & (s < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        acc = float(y[mask].mean())
        conf = float(s[mask].mean())
        ece += (count / n) * abs(acc - conf)
    return float(ece)


def compute_binary_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> dict[str, float]:
    return {
        "roc_auc": roc_auc(labels, scores),
        "pr_auc": pr_auc(labels, scores),
        "f1": f1_at_threshold(labels, scores, threshold),
        "fpr_at_95_recall": fpr_at_recall(labels, scores, target_recall=0.95),
        "ece": expected_calibration_error(labels, scores),
    }


def _clip_probabilities(probs: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.clip(probs, PROB_EPS, 1.0 - PROB_EPS)


def _raw_scores_to_logits(raw_scores: Sequence[float]) -> NDArray[np.float64]:
    values = np.asarray(raw_scores, dtype=np.float64)
    if np.any(values < 0.0) or np.any(values > 1.0):
        return values
    probs = _clip_probabilities(values)
    return np.log(probs / (1.0 - probs))


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    positive = values >= 0
    out = np.empty_like(values)
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_v = np.exp(values[~positive])
    out[~positive] = exp_v / (1.0 + exp_v)
    return out


def temperature_scale_scores(raw_scores: Sequence[float], temperature: float) -> list[float]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    logits = _raw_scores_to_logits(raw_scores)
    probs = _sigmoid(logits / temperature)
    return [float(p) for p in probs]


def log_loss(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    y, p = _as_arrays(labels, probabilities)
    if len(y) == 0:
        return float("nan")
    clipped = _clip_probabilities(p)
    losses = -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
    return float(losses.mean())


def _is_degenerate(labels: Sequence[int], scores: Sequence[float]) -> bool:
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return True
    if not np.all(np.isfinite(s)):
        return True
    return bool(np.allclose(s, s[0]))


def calibration_fallback(*, reason: str) -> Calibration:
    return Calibration(
        temperature=1.0,
        low_threshold=FALLBACK_LOW_THRESHOLD,
        high_threshold=FALLBACK_HIGH_THRESHOLD,
        warnings=(f"calibration_fallback: {reason}",),
    )


def _fit_temperature(labels: Sequence[int], logits: NDArray[np.float64]) -> float:
    base = log_loss(labels, _sigmoid(logits).tolist())
    best_t = 1.0
    best_loss = base
    # Coarse-to-fine search over a wide range; deterministic and dependency-free.
    for lo, hi, steps in ((0.05, 5.0, 100), (max(0.05, best_t - 0.5), best_t + 0.5, 200)):
        candidates = np.linspace(lo, hi, steps)
        for t in candidates:
            loss = log_loss(labels, _sigmoid(logits / float(t)).tolist())
            if loss < best_loss:
                best_loss = loss
                best_t = float(t)
    return best_t


def _choose_thresholds(
    labels: Sequence[int], probabilities: Sequence[float]
) -> tuple[float, float]:
    # High threshold favors precision for ai_generated decisions while keeping recall reasonable.
    # Low threshold favors recall for human decisions. Use dev-only search.
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    candidates = np.unique(np.clip(p, 0.0, 1.0))
    if candidates.size == 0:
        return FALLBACK_LOW_THRESHOLD, FALLBACK_HIGH_THRESHOLD

    # High threshold: p > high => ai_generated. Prefer F1 on ai class, tie-break by lower FPR.
    best_high = FALLBACK_HIGH_THRESHOLD
    best_high_score = (-1.0, 1.0)
    for t in candidates:
        pred_ai = p > t
        tp = float(np.sum(pred_ai & (y == 1)))
        fp = float(np.sum(pred_ai & (y == 0)))
        fn = float(np.sum(~pred_ai & (y == 1)))
        tn = float(np.sum(~pred_ai & (y == 0)))
        f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 1.0
        score = (f1, -fpr)
        if score > best_high_score:
            best_high_score = score
            best_high = float(t)

    # Low threshold: p < low => human. Pick threshold maximizing F1 for human-as-positive class.
    best_low = FALLBACK_LOW_THRESHOLD
    best_low_f1 = -1.0
    for t in candidates:
        pred_human = p < t
        tp = float(np.sum(pred_human & (y == 0)))
        fp = float(np.sum(pred_human & (y == 1)))
        fn = float(np.sum(~pred_human & (y == 0)))
        f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        if f1 > best_low_f1:
            best_low_f1 = f1
            best_low = float(t)

    if best_low > best_high:
        mid = (best_low + best_high) / 2.0
        best_low = mid
        best_high = mid
    return best_low, best_high


def fit_temperature_and_thresholds(
    labels: Sequence[int], raw_scores: Sequence[float]
) -> tuple[float, float, float, tuple[str, ...]]:
    if _is_degenerate(labels, raw_scores):
        cal = calibration_fallback(reason="degenerate dev scores")
        return cal.temperature, cal.low_threshold, cal.high_threshold, cal.warnings
    logits = _raw_scores_to_logits(raw_scores)
    temperature = _fit_temperature(labels, logits)
    probs = _sigmoid(logits / temperature).tolist()
    low, high = _choose_thresholds(labels, probs)
    return temperature, low, high, ()


def write_calibration(
    out_path: Path,
    *,
    model_calibrations: Mapping[str, Mapping[str, Any]],
    seed: int,
    split_strategy: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = {}
    for key, entry in model_calibrations.items():
        cal = entry["calibration"]
        if isinstance(cal, Calibration):
            cal_data = {
                "temperature": cal.temperature,
                "low_threshold": cal.low_threshold,
                "high_threshold": cal.high_threshold,
                "warnings": list(cal.warnings),
            }
        else:
            cal_data = dict(cal)
        models[key] = {
            "model_id": entry.get("model_id"),
            "model_revision": entry.get("model_revision"),
            "config_hash": entry.get("config_hash"),
            "dev_metrics": entry.get("dev_metrics", {}),
            **cal_data,
        }
    payload = {
        "seed": seed,
        "split_strategy": split_strategy,
        "models": models,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize_backends(
    per_track: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    *,
    backend_status: Mapping[str, Mapping[str, Any]] | None = None,
    seed: int | None = None,
    split_strategy: str | None = None,
    model_results: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the committed summary.json payload."""

    backends: dict[str, Any] = {}
    all_keys = set(per_track or {}) | set(backend_status or {}) | set(model_results or {})
    for key in sorted(all_keys):
        status = dict((backend_status or {}).get(key, {}))
        result = dict((model_results or {}).get(key, {}))
        ok = bool(status.get("ok", False))
        entry: dict[str, Any] = {"ok": ok}
        if not ok:
            if "error" in status:
                entry["error"] = status["error"]
            elif status:
                entry["status"] = status
        if "model_id" in result:
            entry["model_id"] = result["model_id"]
        if "model_revision" in result:
            entry["model_revision"] = result["model_revision"]
        if "config_hash" in result:
            entry["config_hash"] = result["config_hash"]
        if "versions" in result:
            entry["versions"] = result["versions"]
        if "timings" in result:
            entry["timings"] = result["timings"]
        if "metrics" in result:
            entry["metrics"] = result["metrics"]
        if "dev_metrics" in result:
            entry["dev_metrics"] = result["dev_metrics"]
        if "scored_tracks" in result:
            entry["scored_tracks"] = result["scored_tracks"]
        if "failed_tracks" in result:
            entry["failed_tracks"] = result["failed_tracks"]
        # Derive basic identity/timing from rows if not explicitly provided.
        rows = list((per_track or {}).get(key, []))
        scored_rows = [row for row in rows if "raw_score" in row]
        if scored_rows and "model_id" not in entry:
            entry["model_id"] = scored_rows[0].get("model_id")
        if scored_rows and "model_revision" not in entry:
            entry["model_revision"] = scored_rows[0].get("model_revision")
        if scored_rows and "config_hash" not in entry:
            entry["config_hash"] = scored_rows[0].get("config_hash")
        if scored_rows and "versions" not in entry:
            entry["versions"] = scored_rows[0].get("versions")
        if "timings" not in entry and rows:
            total = 0.0
            for row in scored_rows:
                timings = row.get("timings") or {}
                total += float(timings.get("score_seconds") or 0.0)
            entry["timings"] = {"total_score_seconds": total, "peak_vram_bytes": None}
        backends[key] = entry

    summary: dict[str, Any] = {"backends": backends}
    if seed is not None:
        summary["seed"] = seed
    if split_strategy is not None:
        summary["split_strategy"] = split_strategy
    return summary


def load_raw_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def scored_samples_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[EvalSample]:
    samples: list[EvalSample] = []
    for row in rows:
        if "raw_score" not in row or "label" not in row or "path" not in row:
            continue
        label = row["label"]
        if label not in (HUMAN_LABEL, AI_GENERATED_LABEL):
            continue
        samples.append(
            EvalSample(path=str(row["path"]), label=label, raw_score=float(row["raw_score"]))
        )
    return samples


def label_to_int(label: str) -> int:
    return 1 if label == AI_GENERATED_LABEL else 0


def evaluate_model_rows(
    rows: Sequence[Mapping[str, Any]], *, seed: int = 42, k_folds: int = DEFAULT_K_FOLDS
) -> dict[str, Any]:
    samples = scored_samples_from_rows(rows)
    failed_tracks = sum(1 for r in rows if "raw_score" not in r)
    if not samples:
        cal = calibration_fallback(reason="no scored rows")
        return {
            "calibration": cal,
            "dev_metrics": {},
            "test_metrics": {},
            "split_strategy": "none",
            "scored_tracks": 0,
            "failed_tracks": failed_tracks,
        }
    split = make_split(samples, seed=seed, k_folds=k_folds)

    # Fit calibration on each fold's dev data and apply only to that fold's test data.
    oof_labels: list[int] = []
    oof_raw: list[float] = []
    oof_probs: list[float] = []
    dev_fit_labels: list[int] = []
    dev_fit_raw: list[float] = []
    dev_fit_warnings: list[str] = []

    for fold in split.folds:
        dev_labels = [label_to_int(s.label) for s in fold.dev]
        dev_raw = [s.raw_score for s in fold.dev]
        test_labels = [label_to_int(s.label) for s in fold.test]
        test_raw = [s.raw_score for s in fold.test]
        temp, _low, _high, warnings = fit_temperature_and_thresholds(dev_labels, dev_raw)
        test_probs = temperature_scale_scores(test_raw, temp) if test_raw else []
        oof_labels.extend(test_labels)
        oof_raw.extend(test_raw)
        oof_probs.extend(test_probs)
        dev_fit_labels.extend(dev_labels)
        dev_fit_raw.extend(dev_raw)
        dev_fit_warnings.extend(warnings)

    # Test metrics use out-of-fold probabilities calibrated with per-fold dev-fitted temps.
    # Exported calibration: fit on dev data only (fixed split dev; k-fold uses all samples as
    # the deployment artifact while reported metrics remain leak-free via OOF predictions).
    temperature, low, high, warnings = fit_temperature_and_thresholds(dev_fit_labels, dev_fit_raw)
    if not warnings and dev_fit_warnings:
        warnings = tuple(dict.fromkeys(dev_fit_warnings))

    dev_probs = temperature_scale_scores(dev_fit_labels, temperature) if dev_fit_raw else []
    dev_metrics = (
        compute_binary_metrics(dev_fit_labels, dev_probs, threshold=high) if dev_probs else {}
    )
    test_metrics = (
        compute_binary_metrics(oof_labels, oof_probs, threshold=high) if oof_probs else {}
    )
    if dev_probs:
        dev_metrics["log_loss"] = log_loss(dev_fit_labels, dev_probs)
        dev_metrics["log_loss_before_temperature"] = log_loss(
            dev_fit_labels, temperature_scale_scores(dev_fit_raw, 1.0)
        )
    if oof_probs:
        test_metrics["log_loss"] = log_loss(oof_labels, oof_probs)
        test_metrics["ece_after_temperature"] = expected_calibration_error(oof_labels, oof_probs)

    cal = Calibration(
        temperature=temperature, low_threshold=low, high_threshold=high, warnings=warnings
    )
    scored_rows = [r for r in rows if "raw_score" in r]
    model_id = scored_rows[0].get("model_id") if scored_rows else None
    model_revision = scored_rows[0].get("model_revision") if scored_rows else None
    config_hash = scored_rows[0].get("config_hash") if scored_rows else None
    versions = scored_rows[0].get("versions") if scored_rows else None
    total_score = sum(
        float((r.get("timings") or {}).get("score_seconds") or 0.0) for r in scored_rows
    )
    return {
        "calibration": cal,
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "split_strategy": split.strategy,
        "k_folds": split.k_folds,
        "scored_tracks": len(scored_rows),
        "failed_tracks": failed_tracks,
        "model_id": model_id,
        "model_revision": model_revision,
        "config_hash": config_hash,
        "versions": versions,
        "timings": {"total_score_seconds": total_score, "peak_vram_bytes": None},
    }


def write_summary(out_path: Path, summary: Mapping[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_report(summary: Mapping[str, Any], calibration: Mapping[str, Any]) -> str:
    lines = [
        "# AI Music Detector — Benchmark Report",
        "",
        "## Scope",
        "",
        "This report compares available bench backends on the labeled manifest using seeded,",
        "leak-free splits. MusicDET is deferred (gate D2, 2026-11-19) and is not included.",
        "",
        "## Limitations",
        "",
        "- The labeled set is small and genre-skewed (human tracks are podcast-derived; AI tracks",
        "  are an untargeted sample), so results are conditional on this mix and directional",
        "  rather than general.",
        "- Generator identity is unknown; only binary human-vs-AI performance can be measured and",
        "  cross-generator generalization cannot be claimed.",
        "- Decoded windows whose peak magnitude exceeds 1.0 are conditionally peak-normalized",
        "  to [-1.0, 1.0] (gain adjustment, with `peak_normalized_to_fit_range` warning)",
        "  rather than clipped; results are conditional on this preprocessing.",
        "- Backends represent the specific model revisions and preprocessing used in this bench;",
        "  older model generations and different decoding windows may behave differently.",
        "- Thresholds and temperature scaling are fitted on dev folds only and may not transfer to",
        "  other distributions.",
        "",
        "## Results",
        "",
        f"- Seed: `{summary.get('seed')}`",
        f"- Split strategy: `{summary.get('split_strategy')}`",
        "",
    ]
    backends = summary.get("backends", {})
    for key, info in backends.items():
        lines.append(f"### `{key}`")
        lines.append("")
        if not info.get("ok", False):
            lines.append(f"- Status: failed (`{info.get('error', 'unknown error')}`)")
            lines.append("")
            continue
        lines.append(f"- Model: `{info.get('model_id')}` @ `{info.get('model_revision')}`")
        if "config_hash" in info:
            lines.append(f"- Config hash: `{info['config_hash']}`")
        scored_tracks = info.get("scored_tracks")
        failed_tracks = info.get("failed_tracks")
        if scored_tracks is not None or failed_tracks is not None:
            lines.append(f"- Tracks: scored `{scored_tracks}`, failed `{failed_tracks}`")
        metrics = info.get("metrics") or {}
        if metrics:
            lines.append("- Test metrics:")
            for name, value in sorted(metrics.items()):
                lines.append(f"  - {name}: `{value}`")
        dev_metrics = info.get("dev_metrics") or {}
        if dev_metrics:
            lines.append("- Dev metrics:")
            for name, value in sorted(dev_metrics.items()):
                lines.append(f"  - {name}: `{value}`")
        timings = info.get("timings") or {}
        if timings:
            lines.append("- Timing:")
            for name, value in sorted(timings.items()):
                lines.append(f"  - {name}: `{value}`")
        model_cal = (calibration.get("models") or {}).get(key, {})
        if model_cal:
            lines.append(
                f"- Calibration: temperature `{model_cal.get('temperature')}`, "
                f"low `{model_cal.get('low_threshold')}`, high `{model_cal.get('high_threshold')}`"
            )
            if model_cal.get("warnings"):
                lines.append(f"- Calibration warnings: `{model_cal['warnings']}`")
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            "Gate D1 is a human decision. If real results are present, choose the default backend",
            "from the measured evidence above and record D1 in `ai/plans/plan.md` section 5. If no",
            "real benchmark results are available, record D1 as pending.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    out_path: Path, summary: Mapping[str, Any], calibration: Mapping[str, Any]
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(summary, calibration), encoding="utf-8")
