# Loop-fill follow-up comparison (T2)

Loop-fill (`simple repeat, no crossfade`, applied only when the selected segment is
shorter than required) compared against the T1 dev-selected zero-pad placement for each
backend.

## Evaluation framework (fixed, goal-aligned)

- Primary: minimize human FPR at AI recall 0.85, computed on dev only (raw scores)
- Acceptable recall band: [0.8, 0.9]
- Comparison: compare loop-fill against the T1 dev-selected zero-pad placement per backend using the same goal-aligned framework; report held-out test once per reported config
- dev-only goal metrics for comparison; held-out test reported once per config
- FPR-at-target-recall is invariant to monotonic temperature scaling; raw scores are used.

## Caveats

- 400 local tracks (200 human / 200 AI), genre-skewed and unknown generator; all
  differences are directional and conditional on this mix.
- Placements were dev-selected in T1; *-2a is a follow-up on that placement (not a new
  placement sweep). Held-out test is reported once per reported config.
- Selected zero-pad test metrics are the same configurations already reported in the T1
  placement sweep artifact; they are shown here for side-by-side comparison only.
- Dev criterion values use raw scores at the dev-derived 0.85-AI-recall operating point.
  Post-calibration `dev_metrics` may saturate on this split and are not the selection
  basis.
- Loop-fill affects only tracks whose selected segment is shorter than the window.

## Per-backend comparison

### `ast-60s` — placement `end` (`AST-1c` vs `AST-2a`)

#### Selected zero-pad: `AST-1c`
- Rows/scored/failed: 400/400/0 (complete: True)
- Dev goal (raw): FPR@0.85 0.2125, recall 0.8500, margin -0.0263, ROC-AUC 0.8864
- Held-out test metrics:
  - ece: 0.0861
  - ece_after_temperature: 0.0861
  - f1: 0.7838
  - fpr_at_95_recall: 0.5500
  - log_loss: 0.4749
  - pr_auc: 0.8933
  - roc_auc: 0.8756
- At dev-derived operating point (threshold 0.9737): 8 human FPs, 6 AI FNs on held-out test.
- Fill diagnostics: loop_filled 0 (human 0, ai_generated 0), short_track_padded 5, peak_normalized 283
- Config hash: `0305dbb9450d63f0892b80ea022a1c9b03b23c87811b507bd705b2ec8b7898db`

#### Loop-fill: `AST-2a`
- Rows/scored/failed: 400/400/0 (complete: True)
- Dev goal (raw): FPR@0.85 0.2125, recall 0.8500, margin -0.0263, ROC-AUC 0.8860
- Held-out test metrics:
  - ece: 0.0861
  - ece_after_temperature: 0.0861
  - f1: 0.7838
  - fpr_at_95_recall: 0.5500
  - log_loss: 0.4749
  - pr_auc: 0.8933
  - roc_auc: 0.8756
- At dev-derived operating point (threshold 0.9737): 8 human FPs, 6 AI FNs on held-out test.
- Fill diagnostics: loop_filled 5 (human 4, ai_generated 1), short_track_padded 0, peak_normalized 283
- Config hash: `55ece9940fd0dd5d9c7cc9adef60c1d0f2bb93cbd75be1e804f404671c487b0c`

**Delta (loop-fill minus zero-pad):**
- Dev human FPR@0.85: 0.0000
- Dev separation margin: 0.0000
- Dev ROC-AUC (raw): -0.0004
- Held-out test ROC-AUC: 0.0000
- Held-out test FPR@95% recall: 0.0000

### `spectttra-alpha-120s` — placement `middle` (`STA-1b` vs `STA-2a`)

#### Selected zero-pad: `STA-1b`
- Rows/scored/failed: 400/400/0 (complete: True)
- Dev goal (raw): FPR@0.85 0.2562, recall 0.8500, margin -4.9551, ROC-AUC 0.8833
- Held-out test metrics:
  - ece: 0.1373
  - ece_after_temperature: 0.1373
  - f1: 0.7895
  - fpr_at_95_recall: 0.5250
  - log_loss: 0.4597
  - pr_auc: 0.8771
  - roc_auc: 0.8806
- At dev-derived operating point (threshold -1.8243): 9 human FPs, 4 AI FNs on held-out test.
- Fill diagnostics: loop_filled 0 (human 0, ai_generated 0), short_track_padded 26, peak_normalized 291
- Config hash: `6296f0efd1ffaf09e9f1248be5b0047ef21965cc40825656de29bf6cb2b59a15`

#### Loop-fill: `STA-2a`
- Rows/scored/failed: 400/400/0 (complete: True)
- Dev goal (raw): FPR@0.85 0.2062, recall 0.8500, margin -4.9551, ROC-AUC 0.9018
- Held-out test metrics:
  - ece: 0.1435
  - ece_after_temperature: 0.1435
  - f1: 0.7848
  - fpr_at_95_recall: 0.5250
  - log_loss: 0.4600
  - pr_auc: 0.8771
  - roc_auc: 0.8806
- At dev-derived operating point (threshold -1.8243): 9 human FPs, 4 AI FNs on held-out test.
- Fill diagnostics: loop_filled 26 (human 17, ai_generated 9), short_track_padded 0, peak_normalized 291
- Config hash: `7bdec0d39d2170700742925058084cf813352ff46c5269b6c8c69019eb04c366`

**Delta (loop-fill minus zero-pad):**
- Dev human FPR@0.85: -0.0500
- Dev separation margin: 0.0000
- Dev ROC-AUC (raw): 0.0185
- Held-out test ROC-AUC: 0.0000
- Held-out test FPR@95% recall: 0.0000

### `spectttra-beta-120s` — placement `middle` (`STB-1b` vs `STB-2a`)

#### Selected zero-pad: `STB-1b`
- Rows/scored/failed: 400/400/0 (complete: True)
- Dev goal (raw): FPR@0.85 0.4500, recall 0.8500, margin -9.1741, ROC-AUC 0.8028
- Held-out test metrics:
  - ece: 0.1400
  - ece_after_temperature: 0.1400
  - f1: 0.7561
  - fpr_at_95_recall: 0.5000
  - log_loss: 0.5934
  - pr_auc: 0.8313
  - roc_auc: 0.8356
- At dev-derived operating point (threshold -4.2132): 20 human FPs, 3 AI FNs on held-out test.
- Fill diagnostics: loop_filled 0 (human 0, ai_generated 0), short_track_padded 26, peak_normalized 291
- Config hash: `0236e120f8b4cbd2e869fb92f326a9bd9c9a62024d5ad057e7d6ffbd610d5def`

#### Loop-fill: `STB-2a`
- Rows/scored/failed: 400/400/0 (complete: True)
- Dev goal (raw): FPR@0.85 0.4188, recall 0.8500, margin -9.1741, ROC-AUC 0.8217
- Held-out test metrics:
  - ece: 0.1504
  - ece_after_temperature: 0.1504
  - f1: 0.7561
  - fpr_at_95_recall: 0.5000
  - log_loss: 0.5917
  - pr_auc: 0.8354
  - roc_auc: 0.8375
- At dev-derived operating point (threshold -4.2132): 20 human FPs, 3 AI FNs on held-out test.
- Fill diagnostics: loop_filled 26 (human 17, ai_generated 9), short_track_padded 0, peak_normalized 291
- Config hash: `169f7a86e165b33ac90fc0352d45fd824371f45edae851b5246c5a00c9636ccb`

**Delta (loop-fill minus zero-pad):**
- Dev human FPR@0.85: -0.0312
- Dev separation margin: 0.0000
- Dev ROC-AUC (raw): 0.0189
- Held-out test ROC-AUC: 0.0019
- Held-out test FPR@95% recall: 0.0000

### `spectttra-gamma-120s` — placement `start` (`STG-1a` vs `STG-2a`)

#### Selected zero-pad: `STG-1a`
- Rows/scored/failed: 400/400/0 (complete: True)
- Dev goal (raw): FPR@0.85 0.5437, recall 0.8500, margin -5.7420, ROC-AUC 0.7772
- Held-out test metrics:
  - ece: 0.0833
  - ece_after_temperature: 0.0833
  - f1: 0.6835
  - fpr_at_95_recall: 0.7250
  - log_loss: 0.6169
  - pr_auc: 0.7496
  - roc_auc: 0.7356
- At dev-derived operating point (threshold -2.2436): 22 human FPs, 6 AI FNs on held-out test.
- Fill diagnostics: loop_filled 0 (human 0, ai_generated 0), short_track_padded 26, peak_normalized 293
- Config hash: `8908d71e8790c73aa1880adf83cd25dce9d32251d8de0b7ca1892c197baf48fc`

#### Loop-fill: `STG-2a`
- Rows/scored/failed: 400/400/0 (complete: True)
- Dev goal (raw): FPR@0.85 0.5000, recall 0.8500, margin -5.7420, ROC-AUC 0.7880
- Held-out test metrics:
  - ece: 0.0822
  - ece_after_temperature: 0.0822
  - f1: 0.6914
  - fpr_at_95_recall: 0.7250
  - log_loss: 0.6148
  - pr_auc: 0.7528
  - roc_auc: 0.7388
- At dev-derived operating point (threshold -2.2436): 22 human FPs, 6 AI FNs on held-out test.
- Fill diagnostics: loop_filled 26 (human 17, ai_generated 9), short_track_padded 0, peak_normalized 293
- Config hash: `3db5c9e12bb51f49cba66ace720611295bf3a1401ccd2045a567cfd9b929ac96`

**Delta (loop-fill minus zero-pad):**
- Dev human FPR@0.85: -0.0437
- Dev separation margin: 0.0000
- Dev ROC-AUC (raw): 0.0108
- Held-out test ROC-AUC: 0.0031
- Held-out test FPR@95% recall: 0.0000

