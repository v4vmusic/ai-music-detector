from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Final

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from benches._shared.constants import (  # noqa: E402
    AI_GENERATED_LABEL,
    HUMAN_LABEL,
    WARNING_LOOP_FILLED,
    WARNING_PEAK_NORMALIZED,
    WARNING_SHORT_TRACK_PADDED,
)
from benches._shared.eval import (  # noqa: E402
    evaluate_model_rows,
    label_to_int,
    load_raw_rows,
    make_split,
    roc_auc,
    scored_samples_from_rows,
)
from benches._shared.experiment import DEFAULT_EXPERIMENTS_DIR, experiment_raw_path  # noqa: E402
from benches._shared.manifest import load_manifest  # noqa: E402
from benches._shared.registry import (  # noqa: E402
    MODEL_KEY_AST,
    MODEL_KEY_SPECTTRA_ALPHA,
    MODEL_KEY_SPECTTRA_BETA,
    MODEL_KEY_SPECTTRA_GAMMA,
)
from benches._shared.selection import (  # noqa: E402
    GOAL_RECALL_BAND,
    GOAL_TARGET_AI_RECALL,
    operating_point_at_recall,
)

__all__ = [
    "LOOP_FILL_CONFIGS",
    "LoopFillSpec",
    "build_summary",
    "main",
    "render_markdown",
]


@dataclass(frozen=True, slots=True)
class LoopFillSpec:
    """One backend's loop-fill follow-up compared against its T1-selected placement."""

    backend: str
    placement: str
    selected_zero_pad_config: str
    loop_fill_config: str


LOOP_FILL_CONFIGS: Final[tuple[LoopFillSpec, ...]] = (
    LoopFillSpec(MODEL_KEY_AST, "end", "AST-1c", "AST-2a"),
    LoopFillSpec(MODEL_KEY_SPECTTRA_ALPHA, "middle", "STA-1b", "STA-2a"),
    LoopFillSpec(MODEL_KEY_SPECTTRA_BETA, "middle", "STB-1b", "STB-2a"),
    LoopFillSpec(MODEL_KEY_SPECTTRA_GAMMA, "start", "STG-1a", "STG-2a"),
)

_CRITERION: Final[dict[str, object]] = {
    "target_recall": GOAL_TARGET_AI_RECALL,
    "acceptable_recall_band": list(GOAL_RECALL_BAND),
    "primary": "minimize human FPR at AI recall 0.85, computed on dev only (raw scores)",
    "comparison": (
        "compare loop-fill against the T1 dev-selected zero-pad placement per backend using the "
        "same goal-aligned framework; report held-out test once per reported config"
    ),
    "selection_data": (
        "dev-only goal metrics for comparison; held-out test reported once per config"
    ),
    "note": (
        "FPR-at-target-recall is invariant to monotonic temperature scaling; raw scores are used."
    ),
}


def _identifiers_match(
    rows: Sequence[dict[str, object]], name: str, placement: str, fill_mode: str, seed: int
) -> bool:
    for row in rows:
        experiment = row.get("experiment")
        if not isinstance(experiment, dict):
            return False
        if (
            experiment.get("config_name") != name
            or experiment.get("placement") != placement
            or experiment.get("fill_mode") != fill_mode
            or row.get("seed") != seed
        ):
            return False
    return True


def _distribution(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": median(values),
        "max": max(values),
    }


def _warning_counts(rows: Sequence[dict[str, object]]) -> dict[str, int]:
    names = (WARNING_SHORT_TRACK_PADDED, WARNING_LOOP_FILLED, WARNING_PEAK_NORMALIZED)
    counts = {name: 0 for name in names}
    for row in rows:
        row_warnings = row.get("warnings")
        if not isinstance(row_warnings, (list, tuple)):
            continue
        for name in names:
            if name in row_warnings:
                counts[name] += 1
    return counts


def _fill_diagnostics(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    affected: list[dict[str, object]] = []
    for row in rows:
        row_warnings = row.get("warnings")
        if isinstance(row_warnings, (list, tuple)) and WARNING_LOOP_FILLED in row_warnings:
            path = row.get("path")
            affected.append(
                {
                    "path": Path(str(path)).name if path is not None else None,
                    "label": row.get("label"),
                }
            )
    by_label = {
        HUMAN_LABEL: sum(1 for item in affected if item["label"] == HUMAN_LABEL),
        AI_GENERATED_LABEL: sum(1 for item in affected if item["label"] == AI_GENERATED_LABEL),
    }
    return {
        "loop_filled_tracks": len(affected),
        "loop_filled_by_label": by_label,
        "affected_track_examples": affected[:10],
    }


def _evaluate_config(
    experiments_dir: Path,
    spec: LoopFillSpec,
    *,
    config_name: str,
    fill_mode: str,
    seed: int,
    expected_rows: int,
) -> dict[str, object]:
    raw_path = experiment_raw_path(experiments_dir, config_name, spec.backend)
    if not raw_path.exists():
        return {
            "config_name": config_name,
            "fill_mode": fill_mode,
            "error": f"missing raw file: {raw_path}",
            "complete": False,
        }
    rows = load_raw_rows(raw_path)
    samples = scored_samples_from_rows(rows)
    split = make_split(samples, seed=seed)
    if split.strategy != "fixed_dev_test" or len(split.folds) != 1:
        return {
            "config_name": config_name,
            "fill_mode": fill_mode,
            "error": f"unexpected split strategy: {split.strategy}",
            "complete": False,
        }
    dev = split.folds[0].dev
    dev_labels = [label_to_int(sample.label) for sample in dev]
    dev_scores = [sample.raw_score for sample in dev]
    evaluation = evaluate_model_rows(rows, seed=seed)
    point = operating_point_at_recall(dev_labels, dev_scores)
    by_label = {
        label: [sample.raw_score for sample in samples if sample.label == label]
        for label in (HUMAN_LABEL, AI_GENERATED_LABEL)
    }
    scored_rows = [row for row in rows if "raw_score" in row]
    failed = len(rows) - len(scored_rows)
    complete = len(rows) == expected_rows and failed == 0
    identifiers_ok = _identifiers_match(rows, config_name, spec.placement, fill_mode, seed)

    held_out = split.folds[0].test
    threshold = point.threshold
    false_positives = [
        sample.path
        for sample in held_out
        if sample.label == HUMAN_LABEL and sample.raw_score >= threshold
    ]
    false_negatives = [
        sample.path
        for sample in held_out
        if sample.label == AI_GENERATED_LABEL and sample.raw_score < threshold
    ]
    return {
        "config_name": config_name,
        "placement": spec.placement,
        "fill_mode": fill_mode,
        "rows": len(rows),
        "scored_tracks": len(scored_rows),
        "failed_tracks": failed,
        "complete": complete and identifiers_ok,
        "identifiers_ok": identifiers_ok,
        "config_hash": scored_rows[0].get("config_hash") if scored_rows else None,
        "model_id": scored_rows[0].get("model_id") if scored_rows else None,
        "model_revision": scored_rows[0].get("model_revision") if scored_rows else None,
        "warning_counts": _warning_counts(rows),
        "fill_diagnostics": _fill_diagnostics(rows),
        "score_distribution_by_label": {
            label: _distribution(values) for label, values in by_label.items()
        },
        "dev_criterion": {
            "target_recall": GOAL_TARGET_AI_RECALL,
            "threshold": point.threshold,
            "ai_recall": point.ai_recall,
            "human_fpr": point.human_fpr,
            "separation_margin": point.separation_margin,
            "roc_auc": roc_auc(dev_labels, dev_scores),
        },
        "dev_metrics": evaluation.get("dev_metrics", {}),
        "test_metrics": evaluation.get("test_metrics", {}),
        "operating_point_diagnostics": {
            "threshold_raw_score": threshold,
            "test_human_false_positives": len(false_positives),
            "test_ai_false_negatives": len(false_negatives),
            "example_false_positive_paths": [Path(path).name for path in false_positives[:10]],
            "example_false_negative_paths": [Path(path).name for path in false_negatives[:10]],
        },
    }


def build_summary(experiments_dir: Path, *, seed: int, expected_rows: int) -> dict[str, object]:
    """Compare loop-fill follow-ups against T1-selected zero-pad placements."""

    backends: dict[str, object] = {}
    for spec in LOOP_FILL_CONFIGS:
        zero_pad = _evaluate_config(
            experiments_dir,
            spec,
            config_name=spec.selected_zero_pad_config,
            fill_mode="zero-pad",
            seed=seed,
            expected_rows=expected_rows,
        )
        loop_fill = _evaluate_config(
            experiments_dir,
            spec,
            config_name=spec.loop_fill_config,
            fill_mode="loop-fill",
            seed=seed,
            expected_rows=expected_rows,
        )
        backends[spec.backend] = {
            "backend": spec.backend,
            "placement": spec.placement,
            "selected_zero_pad_config": spec.selected_zero_pad_config,
            "loop_fill_config": spec.loop_fill_config,
            "zero_pad": zero_pad,
            "loop_fill": loop_fill,
        }
    return {
        "seed": seed,
        "expected_rows": expected_rows,
        "split_strategy": "fixed_dev_test",
        "criterion": _CRITERION,
        "backends": backends,
    }


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _dev_criterion_line(record: Mapping[str, object]) -> str:
    dev = record.get("dev_criterion")
    if not isinstance(dev, Mapping):
        return "n/a"
    return (
        f"FPR@0.85 {_fmt(dev['human_fpr'])}, recall {_fmt(dev['ai_recall'])}, "
        f"margin {_fmt(dev['separation_margin'])}, ROC-AUC {_fmt(dev['roc_auc'])}"
    )


def _test_metrics_lines(record: Mapping[str, object]) -> list[str]:
    metrics = record.get("test_metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        return ["  - n/a"]
    return [f"  - {key}: {_fmt(metrics[key])}" for key in sorted(metrics)]


def render_markdown(summary: Mapping[str, object]) -> str:
    """Render the human-readable loop-fill follow-up comparison."""

    criterion = summary["criterion"]
    assert isinstance(criterion, Mapping)
    backends = summary["backends"]
    assert isinstance(backends, Mapping)

    lines = [
        "# Loop-fill follow-up comparison (T2)",
        "",
        "Loop-fill (`simple repeat, no crossfade`, applied only when the selected segment is",
        "shorter than required) compared against the T1 dev-selected zero-pad placement for each",
        "backend.",
        "",
        "## Evaluation framework (fixed, goal-aligned)",
        "",
        f"- Primary: {criterion['primary']}",
        f"- Acceptable recall band: {criterion['acceptable_recall_band']}",
        f"- Comparison: {criterion['comparison']}",
        f"- {criterion['selection_data']}",
        f"- {criterion['note']}",
        "",
        "## Caveats",
        "",
        "- 400 local tracks (200 human / 200 AI), genre-skewed and unknown generator; all",
        "  differences are directional and conditional on this mix.",
        "- Placements were dev-selected in T1; *-2a is a follow-up on that placement (not a new",
        "  placement sweep). Held-out test is reported once per reported config.",
        "- Selected zero-pad test metrics are the same configurations already reported in the T1",
        "  placement sweep artifact; they are shown here for side-by-side comparison only.",
        "- Dev criterion values use raw scores at the dev-derived 0.85-AI-recall operating point.",
        "  Post-calibration `dev_metrics` may saturate on this split and are not the selection",
        "  basis.",
        "- Loop-fill affects only tracks whose selected segment is shorter than the window.",
        "",
        "## Per-backend comparison",
        "",
    ]

    for backend in sorted(backends):
        record = backends[backend]
        assert isinstance(record, Mapping)
        lines.append(
            f"### `{backend}` — placement `{record['placement']}` "
            f"(`{record['selected_zero_pad_config']}` vs `{record['loop_fill_config']}`)"
        )
        lines.append("")
        for key, label in (("zero_pad", "Selected zero-pad"), ("loop_fill", "Loop-fill")):
            cfg = record[key]
            assert isinstance(cfg, Mapping)
            lines.append(f"#### {label}: `{cfg.get('config_name')}`")
            if "error" in cfg:
                lines.append(f"- Error: {cfg['error']}")
                lines.append("")
                continue
            diagnostics = cfg.get("fill_diagnostics", {})
            assert isinstance(diagnostics, Mapping)
            warnings = cfg.get("warning_counts", {})
            assert isinstance(warnings, Mapping)
            by_label = diagnostics.get("loop_filled_by_label", {})
            assert isinstance(by_label, Mapping)
            lines.append(
                f"- Rows/scored/failed: {cfg.get('rows')}/{cfg.get('scored_tracks')}/"
                f"{cfg.get('failed_tracks')} (complete: {cfg.get('complete')})"
            )
            lines.append(f"- Dev goal (raw): {_dev_criterion_line(cfg)}")
            lines.append("- Held-out test metrics:")
            lines.extend(_test_metrics_lines(cfg))
            op = cfg.get("operating_point_diagnostics", {})
            assert isinstance(op, Mapping)
            if op:
                lines.append(
                    f"- At dev-derived operating point (threshold "
                    f"{_fmt(op.get('threshold_raw_score'))}): "
                    f"{op.get('test_human_false_positives')} human FPs, "
                    f"{op.get('test_ai_false_negatives')} AI FNs on held-out test."
                )
            lines.append(
                f"- Fill diagnostics: loop_filled {diagnostics.get('loop_filled_tracks', 0)} "
                f"(human {by_label.get(HUMAN_LABEL, 0)}, ai_generated "
                f"{by_label.get(AI_GENERATED_LABEL, 0)}), "
                f"short_track_padded {warnings.get(WARNING_SHORT_TRACK_PADDED, 0)}, "
                f"peak_normalized {warnings.get(WARNING_PEAK_NORMALIZED, 0)}"
            )
            lines.append(f"- Config hash: `{cfg.get('config_hash')}`")
            lines.append("")

        zero = record["zero_pad"]
        loop = record["loop_fill"]
        comparable = (
            isinstance(zero, Mapping)
            and isinstance(loop, Mapping)
            and "error" not in zero
            and "error" not in loop
        )
        if comparable:
            zdev = zero["dev_criterion"]
            ldev = loop["dev_criterion"]
            assert isinstance(zdev, Mapping) and isinstance(ldev, Mapping)
            ztest = zero["test_metrics"]
            ltest = loop["test_metrics"]
            assert isinstance(ztest, Mapping) and isinstance(ltest, Mapping)
            lines.append("**Delta (loop-fill minus zero-pad):**")
            lines.append(
                f"- Dev human FPR@0.85: {_fmt(float(ldev['human_fpr']) - float(zdev['human_fpr']))}"
            )
            lines.append(
                f"- Dev separation margin: "
                f"{_fmt(float(ldev['separation_margin']) - float(zdev['separation_margin']))}"
            )
            lines.append(
                f"- Dev ROC-AUC (raw): {_fmt(float(ldev['roc_auc']) - float(zdev['roc_auc']))}"
            )
            if "roc_auc" in ztest and "roc_auc" in ltest:
                lines.append(
                    f"- Held-out test ROC-AUC: "
                    f"{_fmt(float(ltest['roc_auc']) - float(ztest['roc_auc']))}"
                )
            if "fpr_at_95_recall" in ztest and "fpr_at_95_recall" in ltest:
                lines.append(
                    "- Held-out test FPR@95% recall: "
                    f"{_fmt(float(ltest['fpr_at_95_recall']) - float(ztest['fpr_at_95_recall']))}"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize loop-fill follow-ups against T1-selected zero-pad placements."
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=DEFAULT_EXPERIMENTS_DIR,
        help="tuning root (default: benches/results/tuning)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("test-audio/test-songs.csv"),
        help="manifest used for the runs; its row count is the expected raw row count",
    )
    parser.add_argument("--seed", type=int, default=42, help="run seed (default: 42)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory for summary files (default: <experiments-dir>)",
    )
    args = parser.parse_args(argv)

    expected_rows = len(load_manifest(args.manifest))
    summary = build_summary(args.experiments_dir, seed=args.seed, expected_rows=expected_rows)
    out_dir = args.out or args.experiments_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "loop-fill-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "loop-fill-summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(f"wrote loop-fill-summary.json/md to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
