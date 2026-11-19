# Placement sweep comparison (T1)

Single-window placement sweeps (`start`/`middle`/`end`, native windows, `zero-pad`).

## Selection criterion (fixed before the runs)

- Primary: minimize human FPR at AI recall 0.85, computed on dev only (raw scores)
- Acceptable recall band: [0.8, 0.9]
- Tie-breaks: larger separation margin (decision threshold minus highest human score) on dev; higher dev ROC-AUC; config name ascending
- dev only; held-out test reported once per selected config
- FPR-at-target-recall is invariant to monotonic temperature scaling; raw scores are used.

## Caveats

- 400 local tracks (200 human / 200 AI), genre-skewed and unknown generator; all
  differences are directional and conditional on this mix.
- Phase 5 calibration fits saturated on dev (collapsed thresholds). That affects
  post-calibration `dev_metrics`, not the raw-score FPR@0.85 used here; selection is
  still directional on this small set.
- Held-out test metrics below are reported once per selected config only and were not
  used to choose among placements.
- Dev values in the table use raw scores. The shared evaluator's `dev_metrics` are
  post-calibration and saturate on this dev split, so they are not shown here.

## All configs (dev only, raw scores)

| Config | Backend | Placement | Rows | Scored | Padded | Dev FPR@0.85 | Margin | Dev ROC-AUC (raw) | Rank |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AST-1a | ast-60s | start | 400 | 400 | 5 | 0.2562 | -0.1013 | 0.8929 | 2 |
| AST-1b | ast-60s | middle | 400 | 400 | 5 | 0.4125 | -0.0090 | 0.8183 | 3 |
| AST-1c | ast-60s | end | 400 | 400 | 5 | 0.2125 | -0.0263 | 0.8864 | 1 |
| STA-1a | spectttra-alpha-120s | start | 400 | 400 | 26 | 0.3187 | -5.4428 | 0.8554 | 3 |
| STA-1b | spectttra-alpha-120s | middle | 400 | 400 | 26 | 0.2562 | -4.9551 | 0.8833 | 1 |
| STA-1c | spectttra-alpha-120s | end | 400 | 400 | 26 | 0.3000 | -5.6150 | 0.8688 | 2 |
| STB-1a | spectttra-beta-120s | start | 400 | 400 | 26 | 0.5000 | -7.8879 | 0.7782 | 2 |
| STB-1b | spectttra-beta-120s | middle | 400 | 400 | 26 | 0.4500 | -9.1741 | 0.8028 | 1 |
| STB-1c | spectttra-beta-120s | end | 400 | 400 | 26 | 0.5437 | -9.1057 | 0.7671 | 3 |
| STG-1a | spectttra-gamma-120s | start | 400 | 400 | 26 | 0.5437 | -5.7420 | 0.7772 | 1 |
| STG-1b | spectttra-gamma-120s | middle | 400 | 400 | 26 | 0.5938 | -6.4445 | 0.7532 | 2 |
| STG-1c | spectttra-gamma-120s | end | 400 | 400 | 26 | 0.8063 | -5.9610 | 0.6748 | 3 |

## Selected config per backend (held-out test, reported once)

### `ast-60s` — `AST-1c` (end)

- Dev: FPR@0.85 0.2125, recall 0.8500, margin -0.0263, ROC-AUC 0.8864
- Held-out test metrics:
  - ece: 0.0861
  - ece_after_temperature: 0.0861
  - f1: 0.7838
  - fpr_at_95_recall: 0.5500
  - log_loss: 0.4749
  - pr_auc: 0.8933
  - roc_auc: 0.8756
- At the dev-derived operating point: 8 human false positives, 6 AI false negatives on held-out test (raw-score threshold 0.9737).
- Config hash: `0305dbb9450d63f0892b80ea022a1c9b03b23c87811b507bd705b2ec8b7898db`

### `spectttra-alpha-120s` — `STA-1b` (middle)

- Dev: FPR@0.85 0.2562, recall 0.8500, margin -4.9551, ROC-AUC 0.8833
- Held-out test metrics:
  - ece: 0.1373
  - ece_after_temperature: 0.1373
  - f1: 0.7895
  - fpr_at_95_recall: 0.5250
  - log_loss: 0.4597
  - pr_auc: 0.8771
  - roc_auc: 0.8806
- At the dev-derived operating point: 9 human false positives, 4 AI false negatives on held-out test (raw-score threshold -1.8243).
- Config hash: `6296f0efd1ffaf09e9f1248be5b0047ef21965cc40825656de29bf6cb2b59a15`

### `spectttra-beta-120s` — `STB-1b` (middle)

- Dev: FPR@0.85 0.4500, recall 0.8500, margin -9.1741, ROC-AUC 0.8028
- Held-out test metrics:
  - ece: 0.1400
  - ece_after_temperature: 0.1400
  - f1: 0.7561
  - fpr_at_95_recall: 0.5000
  - log_loss: 0.5934
  - pr_auc: 0.8313
  - roc_auc: 0.8356
- At the dev-derived operating point: 20 human false positives, 3 AI false negatives on held-out test (raw-score threshold -4.2132).
- Config hash: `0236e120f8b4cbd2e869fb92f326a9bd9c9a62024d5ad057e7d6ffbd610d5def`

### `spectttra-gamma-120s` — `STG-1a` (start)

- Dev: FPR@0.85 0.5437, recall 0.8500, margin -5.7420, ROC-AUC 0.7772
- Held-out test metrics:
  - ece: 0.0833
  - ece_after_temperature: 0.0833
  - f1: 0.6835
  - fpr_at_95_recall: 0.7250
  - log_loss: 0.6169
  - pr_auc: 0.7496
  - roc_auc: 0.7356
- At the dev-derived operating point: 22 human false positives, 6 AI false negatives on held-out test (raw-score threshold -2.2436).
- Config hash: `8908d71e8790c73aa1880adf83cd25dce9d32251d8de0b7ca1892c197baf48fc`

