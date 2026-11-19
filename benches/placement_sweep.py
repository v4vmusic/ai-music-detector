from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from types import MappingProxyType
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
    EvalSample,
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
    GoalCandidate,
    operating_point_at_recall,
    select_by_goal_criterion,
)

__all__ = [
    "PLACEMENT_BY_CONFIG",
    "PLACEMENT_MATRIX",
    "build_summary",
    "main",
    "render_markdown",
]

PLACEMENT_MATRIX: Final[Mapping[str, tuple[tuple[str, str], ...]]] = MappingProxyType(
    {
        MODEL_KEY_AST: (("AST-1a", "start"), ("AST-1b", "middle"), ("AST-1c", "end")),
        MODEL_KEY_SPECTTRA_ALPHA: (
            ("STA-1a", "start"),
            ("STA-1b", "middle"),
            ("STA-1c", "end"),
        ),
        MODEL_KEY_SPECTTRA_BETA: (("STB-1a", "start"), ("STB-1b", "middle"), ("STB-1c", "end")),
        MODEL_KEY_SPECTTRA_GAMMA: (("STG-1a", "start"), ("STG-1b", "middle"), ("STG-1c", "end")),
    }
)
PLACEMENT_BY_CONFIG: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {
        name: (backend, placement)
        for backend, configs in PLACEMENT_MATRIX.items()
        for name, placement in configs
    }
)

_CRITERION: Final[dict[str, object]] = {
    "target_recall": GOAL_TARGET_AI_RECALL,
    "acceptable_recall_band": list(GOAL_RECALL_BAND),
    "primary": "minimize human FPR at AI recall 0.85, computed on dev only (raw scores)",
    "tie_breaks": [
        "larger separation margin (decision threshold minus highest human score) on dev",
        "higher dev ROC-AUC",
        "config name ascending",
    ],
    "selection_data": "dev only; held-out test reported once per selected config",
    "note": (
        "FPR-at-target-recall is invariant to monotonic temperature scaling; raw scores are used."
    ),
}


def _identifiers_match(
    rows: Sequence[dict[str, object]], name: str, placement: str, seed: int
) -> bool:
    for row in rows:
        experiment = row.get("experiment")
        if not isinstance(experiment, dict):
            return False
        if (
            experiment.get("config_name") != name
            or experiment.get("placement") != placement
            or experiment.get("fill_mode") != "zero-pad"
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
    names = (
        WARNING_SHORT_TRACK_PADDED,
        WARNING_LOOP_FILLED,
        WARNING_PEAK_NORMALIZED,
    )
    counts = {name: 0 for name in names}
    for row in rows:
        row_warnings = row.get("warnings")
        if not isinstance(row_warnings, (list, tuple)):
            continue
        for name in names:
            if name in row_warnings:
                counts[name] += 1
    return counts


def build_summary(experiments_dir: Path, *, seed: int, expected_rows: int) -> dict[str, object]:
    """Evaluate every placement config from its raw JSONL and select one per backend on dev."""

    config_records: dict[str, dict[str, object]] = {}
    dev_candidates: dict[str, list[GoalCandidate]] = {backend: [] for backend in PLACEMENT_MATRIX}
    evaluations: dict[str, dict[str, object]] = {}
    test_samples: dict[str, list[EvalSample]] = {}

    for backend, configs in PLACEMENT_MATRIX.items():
        for name, placement in configs:
            raw_path = experiment_raw_path(experiments_dir, name, backend)
            if not raw_path.exists():
                config_records[name] = {
                    "backend": backend,
                    "placement": placement,
                    "error": f"missing raw file: {raw_path}",
                    "complete": False,
                }
                continue
            rows = load_raw_rows(raw_path)
            samples = scored_samples_from_rows(rows)
            split = make_split(samples, seed=seed)
            if split.strategy != "fixed_dev_test" or len(split.folds) != 1:
                config_records[name] = {
                    "backend": backend,
                    "placement": placement,
                    "error": f"unexpected split strategy: {split.strategy}",
                    "complete": False,
                }
                continue
            dev = split.folds[0].dev
            dev_labels = [label_to_int(sample.label) for sample in dev]
            dev_scores = [sample.raw_score for sample in dev]
            evaluation = evaluate_model_rows(rows, seed=seed)
            evaluations[name] = evaluation
            test_samples[name] = split.folds[0].test
            dev_candidates[backend].append(GoalCandidate(name, dev_labels, dev_scores))
            point = operating_point_at_recall(dev_labels, dev_scores)
            by_label = {
                label: [sample.raw_score for sample in samples if sample.label == label]
                for label in (HUMAN_LABEL, AI_GENERATED_LABEL)
            }
            scored_rows = [row for row in rows if "raw_score" in row]
            failed = len(rows) - len(scored_rows)
            complete = len(rows) == expected_rows and failed == 0
            identifiers_ok = _identifiers_match(rows, name, placement, seed)
            config_records[name] = {
                "backend": backend,
                "placement": placement,
                "fill_mode": "zero-pad",
                "rows": len(rows),
                "scored_tracks": len(scored_rows),
                "failed_tracks": failed,
                "complete": complete and identifiers_ok,
                "identifiers_ok": identifiers_ok,
                "config_hash": scored_rows[0].get("config_hash") if scored_rows else None,
                "model_id": scored_rows[0].get("model_id") if scored_rows else None,
                "model_revision": scored_rows[0].get("model_revision") if scored_rows else None,
                "warning_counts": _warning_counts(rows),
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
            }

    selected: dict[str, dict[str, object]] = {}
    for backend, candidates in dev_candidates.items():
        if not candidates:
            selected[backend] = {"error": "no config produced scored rows"}
            continue
        result = select_by_goal_criterion(candidates)
        for rank, item in enumerate(result.ranked, start=1):
            config_records[item.name]["dev_rank"] = rank
        chosen = result.selected_evaluation
        name = result.selected
        evaluation = evaluations[name]
        held_out = test_samples[name]
        threshold = chosen.threshold
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
        selected[backend] = {
            "config": name,
            "placement": PLACEMENT_BY_CONFIG[name][1],
            "dev_criterion": {
                "target_recall": chosen.target_recall,
                "threshold": chosen.threshold,
                "ai_recall": chosen.ai_recall,
                "human_fpr": chosen.human_fpr,
                "separation_margin": chosen.separation_margin,
                "roc_auc": chosen.roc_auc,
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
            "config_hash": config_records[name].get("config_hash"),
            "split_strategy": evaluation.get("split_strategy"),
        }

    return {
        "seed": seed,
        "expected_rows": expected_rows,
        "split_strategy": "fixed_dev_test",
        "criterion": _CRITERION,
        "configs": config_records,
        "selected": selected,
    }


def render_markdown(summary: Mapping[str, object]) -> str:
    """Render the human-readable placement-sweep comparison from a built summary."""

    criterion = summary["criterion"]
    assert isinstance(criterion, Mapping)
    selected = summary["selected"]
    assert isinstance(selected, Mapping)
    configs = summary["configs"]
    assert isinstance(configs, Mapping)

    lines = [
        "# Placement sweep comparison (T1)",
        "",
        "Single-window placement sweeps (`start`/`middle`/`end`, native windows, `zero-pad`).",
        "",
        "## Selection criterion (fixed before the runs)",
        "",
        f"- Primary: {criterion['primary']}",
        f"- Acceptable recall band: {criterion['acceptable_recall_band']}",
        "- Tie-breaks: " + "; ".join(str(item) for item in criterion["tie_breaks"]),
        f"- {criterion['selection_data']}",
        f"- {criterion['note']}",
        "",
        "## Caveats",
        "",
        "- 400 local tracks (200 human / 200 AI), genre-skewed and unknown generator; all",
        "  differences are directional and conditional on this mix.",
        "- Phase 5 calibration fits saturated on dev (collapsed thresholds). That affects",
        "  post-calibration `dev_metrics`, not the raw-score FPR@0.85 used here; selection is",
        "  still directional on this small set.",
        "- Held-out test metrics below are reported once per selected config only and were not",
        "  used to choose among placements.",
        "- Dev values in the table use raw scores. The shared evaluator's `dev_metrics` are",
        "  post-calibration and saturate on this dev split, so they are not shown here.",
        "",
        "## All configs (dev only, raw scores)",
        "",
        "| Config | Backend | Placement | Rows | Scored | Padded | Dev FPR@0.85 | Margin"
        " | Dev ROC-AUC (raw) | Rank |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in sorted(configs):
        record = configs[name]
        assert isinstance(record, Mapping)
        if "dev_criterion" not in record:
            lines.append(
                f"| {name} | {record.get('backend')} | {record.get('placement')} | - | - | - "
                f"| - | - | - | - (error: {record.get('error')}) |"
            )
            continue
        dev_criterion = record["dev_criterion"]
        assert isinstance(dev_criterion, Mapping)
        warnings = record["warning_counts"]
        assert isinstance(warnings, Mapping)
        lines.append(
            f"| {name} | {record['backend']} | {record['placement']} "
            f"| {record['rows']} | {record['scored_tracks']} "
            f"| {warnings[WARNING_SHORT_TRACK_PADDED]} "
            f"| {_fmt(dev_criterion['human_fpr'])} | {_fmt(dev_criterion['separation_margin'])} "
            f"| {_fmt(dev_criterion.get('roc_auc'))} | {record.get('dev_rank')} |"
        )

    lines.extend(["", "## Selected config per backend (held-out test, reported once)", ""])
    for backend in sorted(selected):
        choice = selected[backend]
        assert isinstance(choice, Mapping)
        lines.append(f"### `{backend}` — `{choice.get('config')}` ({choice.get('placement')})")
        lines.append("")
        if "error" in choice:
            lines.append(f"- No selection: {choice['error']}")
            lines.append("")
            continue
        dev_criterion = choice["dev_criterion"]
        assert isinstance(dev_criterion, Mapping)
        test_metrics = choice["test_metrics"]
        assert isinstance(test_metrics, Mapping)
        diagnostics = choice["operating_point_diagnostics"]
        assert isinstance(diagnostics, Mapping)
        lines.append(
            f"- Dev: FPR@0.85 {_fmt(dev_criterion['human_fpr'])}, "
            f"recall {_fmt(dev_criterion['ai_recall'])}, "
            f"margin {_fmt(dev_criterion['separation_margin'])}, "
            f"ROC-AUC {_fmt(dev_criterion['roc_auc'])}"
        )
        lines.append("- Held-out test metrics:")
        for key in sorted(test_metrics):
            lines.append(f"  - {key}: {_fmt(test_metrics[key])}")
        lines.append(
            f"- At the dev-derived operating point: {diagnostics['test_human_false_positives']} "
            f"human false positives, {diagnostics['test_ai_false_negatives']} AI false negatives "
            f"on held-out test (raw-score threshold "
            f"{_fmt(diagnostics['threshold_raw_score'])})."
        )
        lines.append(f"- Config hash: `{choice.get('config_hash')}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize placement sweeps and select one config per backend on dev."
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
    (out_dir / "placement-sweep-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "placement-sweep-summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(f"wrote placement-sweep-summary.json/md to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
