from __future__ import annotations

import json
from pathlib import Path

from benches._shared.constants import (
    AI_GENERATED_LABEL,
    HUMAN_LABEL,
    WARNING_LOOP_FILLED,
)
from benches._shared.experiment import experiment_raw_path
from benches.loop_fill_followup import (
    LOOP_FILL_CONFIGS,
    build_summary,
    render_markdown,
)


def _write_config_rows(
    experiments_dir: Path,
    config_name: str,
    backend: str,
    *,
    placement: str,
    fill_mode: str,
    seed: int,
    rows: list[dict[str, object]],
) -> None:
    path = experiment_raw_path(experiments_dir, config_name, backend)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _rows(
    *,
    config_name: str,
    backend: str,
    placement: str,
    fill_mode: str,
    seed: int,
    human_scores: list[float],
    ai_scores: list[float],
    loop_filled_human: int = 0,
    loop_filled_ai: int = 0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, score in enumerate(human_scores):
        warnings = [WARNING_LOOP_FILLED] if index < loop_filled_human else []
        rows.append(
            {
                "path": f"/audio/human-{index}.wav",
                "label": HUMAN_LABEL,
                "raw_score": score,
                "calibrated_probability": score,
                "model_id": f"model/{backend}",
                "model_revision": "rev1",
                "window": "60s@16000Hz@0s",
                "warnings": warnings,
                "timings": {"score_seconds": 0.1, "gpu_seconds": None},
                "offset_seconds": 0.0,
                "seed": seed,
                "config_hash": f"hash-{config_name}",
                "versions": {"numpy": "1.26.4"},
                "experiment": {
                    "config_name": config_name,
                    "placement": placement,
                    "fill_mode": fill_mode,
                },
            }
        )
    for index, score in enumerate(ai_scores):
        warnings = [WARNING_LOOP_FILLED] if index < loop_filled_ai else []
        rows.append(
            {
                "path": f"/audio/ai-{index}.wav",
                "label": AI_GENERATED_LABEL,
                "raw_score": score,
                "calibrated_probability": score,
                "model_id": f"model/{backend}",
                "model_revision": "rev1",
                "window": "60s@16000Hz@0s",
                "warnings": warnings,
                "timings": {"score_seconds": 0.1, "gpu_seconds": None},
                "offset_seconds": 0.0,
                "seed": seed,
                "config_hash": f"hash-{config_name}",
                "versions": {"numpy": "1.26.4"},
                "experiment": {
                    "config_name": config_name,
                    "placement": placement,
                    "fill_mode": fill_mode,
                },
            }
        )
    return rows


def test_build_summary_compares_loop_fill_against_selected_zero_pad(tmp_path: Path) -> None:
    seed = 42
    # 60 per class => fixed dev/test split (>= SMALL_CLASS_THRESHOLD), 48 dev / 12 test each.
    human_base = [0.1] * 48 + [0.2] * 12
    ai_base = [0.9] * 48 + [0.8] * 12
    # Loop-fill makes some affected humans look a bit less AI-like in this synthetic setup.
    human_loop = [0.05] * 48 + [0.15] * 12
    ai_loop = [0.9] * 48 + [0.8] * 12

    for spec in LOOP_FILL_CONFIGS:
        _write_config_rows(
            tmp_path,
            spec.selected_zero_pad_config,
            spec.backend,
            placement=spec.placement,
            fill_mode="zero-pad",
            seed=seed,
            rows=_rows(
                config_name=spec.selected_zero_pad_config,
                backend=spec.backend,
                placement=spec.placement,
                fill_mode="zero-pad",
                seed=seed,
                human_scores=human_base,
                ai_scores=ai_base,
            ),
        )
        _write_config_rows(
            tmp_path,
            spec.loop_fill_config,
            spec.backend,
            placement=spec.placement,
            fill_mode="loop-fill",
            seed=seed,
            rows=_rows(
                config_name=spec.loop_fill_config,
                backend=spec.backend,
                placement=spec.placement,
                fill_mode="loop-fill",
                seed=seed,
                human_scores=human_loop,
                ai_scores=ai_loop,
                loop_filled_human=4,
                loop_filled_ai=1,
            ),
        )

    summary = build_summary(tmp_path, seed=seed, expected_rows=120)

    assert summary["seed"] == seed
    assert summary["expected_rows"] == 120
    assert "criterion" in summary
    backends = summary["backends"]
    assert isinstance(backends, dict)
    assert set(backends) == {spec.backend for spec in LOOP_FILL_CONFIGS}

    for spec in LOOP_FILL_CONFIGS:
        record = backends[spec.backend]
        assert isinstance(record, dict)
        assert record["backend"] == spec.backend
        assert record["placement"] == spec.placement
        assert record["selected_zero_pad_config"] == spec.selected_zero_pad_config
        assert record["loop_fill_config"] == spec.loop_fill_config
        for key in ("zero_pad", "loop_fill"):
            cfg = record[key]
            assert cfg["complete"] is True
            assert cfg["rows"] == 120
            assert cfg["scored_tracks"] == 120
            assert cfg["failed_tracks"] == 0
            assert "dev_criterion" in cfg
            assert "test_metrics" in cfg
            assert cfg["dev_criterion"]["target_recall"] == 0.85
        loop_diag = record["loop_fill"]["fill_diagnostics"]
        assert loop_diag["loop_filled_tracks"] == 5
        assert loop_diag["loop_filled_by_label"] == {HUMAN_LABEL: 4, AI_GENERATED_LABEL: 1}
        zero_diag = record["zero_pad"]["fill_diagnostics"]
        assert zero_diag["loop_filled_tracks"] == 0

    md = render_markdown(summary)
    assert "Loop-fill follow-up comparison (T2)" in md
    assert "held-out test" in md.lower()
    assert "AST-2a" in md
    assert "loop_filled" in md
    for spec in LOOP_FILL_CONFIGS:
        assert spec.backend in md
        assert spec.loop_fill_config in md
        assert spec.selected_zero_pad_config in md


def test_loop_fill_configs_match_t1_selections() -> None:
    mapping = {spec.backend: spec for spec in LOOP_FILL_CONFIGS}
    assert mapping["ast-60s"].selected_zero_pad_config == "AST-1c"
    assert mapping["ast-60s"].placement == "end"
    assert mapping["ast-60s"].loop_fill_config == "AST-2a"
    assert mapping["spectttra-alpha-120s"].selected_zero_pad_config == "STA-1b"
    assert mapping["spectttra-alpha-120s"].placement == "middle"
    assert mapping["spectttra-beta-120s"].selected_zero_pad_config == "STB-1b"
    assert mapping["spectttra-beta-120s"].placement == "middle"
    assert mapping["spectttra-gamma-120s"].selected_zero_pad_config == "STG-1a"
    assert mapping["spectttra-gamma-120s"].placement == "start"
