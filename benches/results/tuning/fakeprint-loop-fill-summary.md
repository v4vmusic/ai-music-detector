# Fakeprint loop-fill follow-up (T2-style, post-T3 addendum)

Compared `FP-2a` (loop-fill, placement `middle`, dev-selected) against `FP-1b` (zero-pad, same placement) using the same goal-aligned framework.

## Result

- Loop-fill affected tracks: **0** (fakeprint 10s window; no manifest track has a selected segment shorter than 10s). Zero-pad short segments: 0 (zero-pad) / 0 (loop-fill).

### Zero-pad (FP-1b)

- Dev: FPR@0.85 0.0000, recall 0.8500, margin 0.0000, ROC-AUC 0.9747
- Held-out: ece=0.2857, ece_after_temperature=0.2857, f1=0.9091, fpr_at_95_recall=0.1500, log_loss=0.6333, pr_auc=0.9756, roc_auc=0.9647
- Held-out @dev-thr: 0 human FP / 40, 5 AI FN / 40

### Loop-fill (FP-2a)

- Dev: FPR@0.85 0.0000, recall 0.8500, margin 0.0000, ROC-AUC 0.9747
- Held-out: ece=0.2857, ece_after_temperature=0.2857, f1=0.9091, fpr_at_95_recall=0.1500, log_loss=0.6333, pr_auc=0.9756, roc_auc=0.9647
- Held-out @dev-thr: 0 human FP / 40, 5 AI FN / 40

**Delta (loop-fill minus zero-pad):** no material change (no segments required filling).

