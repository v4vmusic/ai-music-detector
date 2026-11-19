# Fakeprint placement sweep (T1-style, post-T3 addendum)

Evaluated for v4vmusic RSS-ingested music spam-filtering: **minimize human false flags at ~80–90% AI recall** (catch most AI, avoid tagging legitimate human songs). Uses the same dev-only goal-aligned criterion/protocol as `placement-sweep-summary.md`.

## Selection criterion (fixed before comparison)

- minimize human FPR at AI recall 0.85, computed on dev only (raw scores)
- dev only; held-out test reported once per selected config
- FPR-at-target-recall is invariant to monotonic temperature scaling; raw scores are used.
- Acceptable recall band: [0.8, 0.9]
- Tie-breaks: larger separation margin (decision threshold minus highest human score) on dev; higher dev ROC-AUC; config name ascending

## Caveats

- 400 local v4vmusic.com RSS-sourced tracks (200 human / 200 AI), genre-skewed and unknown generator; directional/conditional.
- Fakeprint window 10s (CPU/ONNX). Not placement/fill-tuned in Phase 5 baseline; `fakeprint` was outside original T1 scope.
- Dev selection on raw scores only; held-out test reported once per selected config.
- Phase 5 baseline (fixed offset 0s) is equivalent to start placement here; included for reference.

## Configs (dev only, raw scores)

| Config | Placement | Fill | Rows | Padded | Loop-filled | Dev FPR@0.85 | Margin | Dev ROC-AUC | Rank |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| FP-1a | start | zero-pad | 400 | 0 | 0 | 0.0063 | -0.9475 | 0.9683 | 2 |
| FP-1b | middle | zero-pad | 400 | 0 | 0 | 0.0000 | 0.0000 | 0.9747 | 1 |
| FP-1c | end | zero-pad | 400 | 0 | 0 | 0.1062 | -0.9989 | 0.9121 | 3 |

## Selected: `FP-1b` (middle)

- Dev: FPR@0.85 0.0000, recall 0.8500, margin 0.0000, ROC-AUC 0.9747, threshold 4.139542579650879e-05
- Held-out test metrics:
  - ece: 0.2857
  - ece_after_temperature: 0.2857
  - f1: 0.9091
  - fpr_at_95_recall: 0.1500
  - log_loss: 0.6333
  - pr_auc: 0.9756
  - roc_auc: 0.9647
- At dev-derived 85%-recall operating point (held-out): **0 human false positives / 40**, **5 AI false negatives / 40** (threshold 4.139542579650879e-05).

## Phase 5 baseline reference (fixed offset 0s, start-equivalent)

- Dev: FPR@0.85 0.0063, recall 0.8500, margin -0.9475, ROC-AUC 0.9683, threshold 6.562471389770508e-05
- Held-out test metrics:
  - ece: 0.3240
  - ece_after_temperature: 0.3240
  - f1: 0.8767
  - fpr_at_95_recall: 1.0000
  - log_loss: 0.7956
  - pr_auc: 0.9271
  - roc_auc: 0.9231
- At dev-derived 85%-recall operating point (held-out): **1 human FP / 40**, **8 AI FN / 40**.

## Interpretation for v4vmusic use case

- `FP-1b` (middle, 10s) is best by the goal-aligned dev criterion and on held-out at that point (0 human FP / 5 AI FN), improving over baseline/start (1/8) and end (2/6).
- This supports a conservative “protect humans” operating point better than other tested configs, but `FPR@95% recall` remains 0.15 and `ECE` is high (~0.29) — pushing toward catch-all/high recall can reintroduce human false flags and calibration is weak.
- For the stated policy (auto-tag only high-confidence AI, route uncertain to review), per-track raw scores in `benches/results/tuning/FP-1b/raw/fakeprint.jsonl` (and baseline `benches/results/raw/fakeprint.jsonl`) can be analyzed to choose tri-state thresholds without reruns.

