from __future__ import annotations

import json
from pathlib import Path

import pytest

from benches._shared.eval import (
    EvalSample,
    calibration_fallback,
    compute_binary_metrics,
    expected_calibration_error,
    fit_temperature_and_thresholds,
    log_loss,
    make_split,
    pr_auc,
    roc_auc,
    summarize_backends,
    temperature_scale_scores,
    write_calibration,
)


def _sample(key: str, label: str, raw_score: float) -> EvalSample:
    return EvalSample(path=key, label=label, raw_score=raw_score)  # type: ignore[arg-type]


def test_stratified_split_is_seeded_and_leak_free_small_classes() -> None:
    # Fewer than 50 per class => stratified k-fold.
    samples = [_sample(f"h{i}", "human", 0.1) for i in range(10)] + [
        _sample(f"a{i}", "ai_generated", 0.9) for i in range(10)
    ]
    split_a = make_split(samples, seed=42)
    split_b = make_split(samples, seed=42)
    split_c = make_split(samples, seed=7)

    assert split_a.strategy == "stratified_k_fold"
    assert split_a.k_folds == 5
    assert split_a == split_b
    assert split_a.folds != split_c.folds

    all_keys = {sample.path for sample in samples}
    seen_test: set[str] = set()
    for fold in split_a.folds:
        dev_keys = {sample.path for sample in fold.dev}
        test_keys = {sample.path for sample in fold.test}
        assert dev_keys.isdisjoint(test_keys)
        assert dev_keys | test_keys == all_keys
        assert seen_test.isdisjoint(test_keys)
        seen_test |= test_keys
        dev_labels = [s.label for s in fold.dev]
        test_labels = [s.label for s in fold.test]
        assert dev_labels.count("human") == 8
        assert dev_labels.count("ai_generated") == 8
        assert test_labels.count("human") == 2
        assert test_labels.count("ai_generated") == 2
    assert seen_test == all_keys


def test_fixed_dev_test_split_when_classes_are_large_enough() -> None:
    samples = [_sample(f"h{i}", "human", 0.1) for i in range(60)] + [
        _sample(f"a{i}", "ai_generated", 0.9) for i in range(60)
    ]
    split = make_split(samples, seed=42)
    assert split.strategy == "fixed_dev_test"
    assert len(split.folds) == 1
    fold = split.folds[0]
    dev_keys = {s.path for s in fold.dev}
    test_keys = {s.path for s in fold.test}
    assert dev_keys.isdisjoint(test_keys)
    assert dev_keys | test_keys == {s.path for s in samples}
    assert [s.label for s in fold.dev].count("human") == 48
    assert [s.label for s in fold.dev].count("ai_generated") == 48
    assert [s.label for s in fold.test].count("human") == 12
    assert [s.label for s in fold.test].count("ai_generated") == 12


def test_metrics_match_hand_computed_values() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]

    assert roc_auc(labels, scores) == pytest.approx(0.75)
    # PR curve points by descending score: (0.8,1)->P=1,R=0.5; (0.4,0)->P=0.5,R=0.5;
    # (0.35,1)->P=2/3,R=1.0; (0.1,0)->P=0.5,R=1.0. Step AP = 1*0.5 + (2/3)*0.5 = 5/6.
    assert pr_auc(labels, scores) == pytest.approx(5 / 6)
    metrics = compute_binary_metrics(labels, scores, threshold=0.5)
    assert metrics["f1"] == pytest.approx(2 / 3)
    # Recall >= 0.95 only at full recall; threshold 0.35 classifies indices 1,2,3 positive.
    assert metrics["fpr_at_95_recall"] == pytest.approx(0.5)
    # bins [0,0.5): scores 0.1,0.4,0.35 labels 0,0,1 -> acc 1/3, conf 0.85/3; [0.5,1]: 0.8,1
    assert expected_calibration_error(labels, scores, bins=2) == pytest.approx(0.0875)


def test_temperature_scaling_improves_dev_log_loss_and_does_not_touch_test() -> None:
    dev_labels = [0, 0, 0, 1, 1, 1]
    dev_logits = [-2.0, -1.5, -0.5, 0.5, 1.5, 2.0]
    test_labels = [0, 1]
    test_logits = [-0.2, 0.2]

    before = log_loss(dev_labels, temperature_scale_scores(dev_logits, 1.0))
    temperature, low, high, warnings = fit_temperature_and_thresholds(dev_labels, dev_logits)
    after = log_loss(dev_labels, temperature_scale_scores(dev_logits, temperature))

    assert after <= before + 1e-12
    assert temperature > 0.0
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert low <= high
    assert warnings == ()

    # Fit uses only dev data; test inputs remain valid scores after applying fitted temperature.
    test_probs = temperature_scale_scores(test_logits, temperature)
    assert all(0.0 <= p <= 1.0 for p in test_probs)
    assert test_logits == [-0.2, 0.2]
    assert test_labels == [0, 1]


def test_calibration_fallback_and_export(tmp_path: Path) -> None:
    fallback = calibration_fallback(reason="degenerate dev scores")
    assert fallback.temperature == 1.0
    assert fallback.low_threshold == 0.35
    assert fallback.high_threshold == 0.65
    assert fallback.warnings

    cal = calibration_fallback(reason="no dev data")
    out = tmp_path / "calibration.json"
    write_calibration(
        out,
        model_calibrations={
            "fakeprint": {
                "model_id": "lofcz/ai-music-detector",
                "model_revision": "rev1",
                "config_hash": "abc",
                "calibration": cal,
                "dev_metrics": {},
            }
        },
        seed=42,
        split_strategy="fixed_dev_test",
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    entry = data["models"]["fakeprint"]
    assert entry["temperature"] == 1.0
    assert entry["low_threshold"] == 0.35
    assert entry["high_threshold"] == 0.65
    assert entry["model_id"] == "lofcz/ai-music-detector"
    assert entry["model_revision"] == "rev1"
    assert entry["config_hash"] == "abc"
    assert entry["dev_metrics"] == {}
    assert entry["warnings"]
    assert data["seed"] == 42
    assert data["split_strategy"] == "fixed_dev_test"


def test_summary_contains_required_fields_and_records_failed_backend() -> None:
    per_track = {
        "fakeprint": [
            {
                "path": "/a.wav",
                "label": "human",
                "raw_score": 0.1,
                "calibrated_probability": 0.1,
                "model_id": "lofcz/ai-music-detector",
                "model_revision": "rev-fp",
                "timings": {"score_seconds": 0.01, "gpu_seconds": None},
                "config_hash": "hash-fp",
                "versions": {"numpy": "1.26.4"},
                "seed": 42,
                "warnings": [],
                "window": "10s@16000Hz",
            },
            {
                "path": "/b.wav",
                "label": "ai_generated",
                "raw_score": 0.9,
                "calibrated_probability": 0.9,
                "model_id": "lofcz/ai-music-detector",
                "model_revision": "rev-fp",
                "timings": {"score_seconds": 0.02, "gpu_seconds": None},
                "config_hash": "hash-fp",
                "versions": {"numpy": "1.26.4"},
                "seed": 42,
                "warnings": [],
                "window": "10s@16000Hz",
            },
        ],
        "ast-60s": [],  # no scored rows
    }
    backend_status: dict[str, dict[str, object]] = {
        "fakeprint": {"ok": True},
        "ast-60s": {"ok": False, "error": "ModelLoadError: boom"},
    }
    summary = summarize_backends(
        per_track,
        backend_status=backend_status,
        seed=42,
        split_strategy="fixed_dev_test",
        model_results={
            "fakeprint": {
                "model_id": "lofcz/ai-music-detector",
                "model_revision": "rev-fp",
                "config_hash": "hash-fp",
                "versions": {"numpy": "1.26.4"},
                "timings": {"total_score_seconds": 0.03, "peak_vram_bytes": None},
                "metrics": {"roc_auc": 1.0},
                "dev_metrics": {"roc_auc": 1.0},
            }
        },
    )

    assert summary["seed"] == 42
    assert summary["split_strategy"] == "fixed_dev_test"
    fp = summary["backends"]["fakeprint"]
    assert fp["ok"] is True
    assert fp["model_id"] == "lofcz/ai-music-detector"
    assert fp["model_revision"] == "rev-fp"
    assert fp["config_hash"] == "hash-fp"
    assert fp["versions"] == {"numpy": "1.26.4"}
    assert fp["timings"]["total_score_seconds"] == 0.03
    assert "peak_vram_bytes" in fp["timings"]
    assert fp["metrics"]["roc_auc"] == 1.0

    ast = summary["backends"]["ast-60s"]
    assert ast["ok"] is False
    assert "boom" in ast["error"]
    # Failed backend recorded but does not corrupt the successful backend's summary.
    assert summary["backends"]["fakeprint"]["ok"] is True
