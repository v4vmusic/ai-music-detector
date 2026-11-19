# AI Music Detector — Benchmark Report

## Scope

This report compares available bench backends on the labeled manifest using seeded,
leak-free splits. MusicDET is deferred (gate D2, 2026-11-19) and is not included.

## Limitations

- The labeled set is small and genre-skewed (human tracks are podcast-derived; AI tracks
  are an untargeted sample), so results are conditional on this mix and directional
  rather than general.
- Generator identity is unknown; only binary human-vs-AI performance can be measured and
  cross-generator generalization cannot be claimed.
- Decoded windows whose peak magnitude exceeds 1.0 are conditionally peak-normalized
  to [-1.0, 1.0] (gain adjustment, with `peak_normalized_to_fit_range` warning)
  rather than clipped; results are conditional on this preprocessing.
- Backends represent the specific model revisions and preprocessing used in this bench;
  older model generations and different decoding windows may behave differently.
- Thresholds and temperature scaling are fitted on dev folds only and may not transfer to
  other distributions.

## Results

- Seed: `42`
- Split strategy: `fixed_dev_test`

### `ast-60s`

- Model: `AI-Music-Detection/ai_music_detection_large_60s` @ `5e29461bbfc41dc5fe4ed5f7fadc961aa84a7e3e`
- Config hash: `26be8796c4333cf66067d29593ce2262c7861cf729e6c2b3d734028d9740ae0b`
- Tracks: scored `400`, failed `0`
- Test metrics:
  - ece: `0.15686458503328196`
  - ece_after_temperature: `0.15686458503328196`
  - f1: `0.7631578947368421`
  - fpr_at_95_recall: `0.45`
  - log_loss: `0.5623304531236248`
  - pr_auc: `0.8809907188917006`
  - roc_auc: `0.8578125`
- Dev metrics:
  - ece: `0.03828650461672975`
  - f1: `1.0`
  - fpr_at_95_recall: `0.0`
  - log_loss: `0.039038694521076445`
  - log_loss_before_temperature: `2.2059515391570375`
  - pr_auc: `1.0`
  - roc_auc: `1.0`
- Timing:
  - peak_vram_bytes: `None`
  - total_score_seconds: `272.37816348983324`
- Calibration: temperature `5.0`, low `0.8458905146738921`, high `0.8458905146738921`

### `fakeprint`

- Model: `lofcz/ai-music-detector` @ `d2180598fed79e3f917e8050a00439982466e5c6`
- Config hash: `4aadb6053fdf57f3983fe749ab184462ba5bfdee2a63c4fdef2b3e2f10b473df`
- Tracks: scored `400`, failed `0`
- Test metrics:
  - ece: `0.32403749601092063`
  - ece_after_temperature: `0.32403749601092063`
  - f1: `0.8767123287671232`
  - fpr_at_95_recall: `1.0`
  - log_loss: `0.7955672543243145`
  - pr_auc: `0.9271101380674099`
  - roc_auc: `0.923125`
- Dev metrics:
  - ece: `0.03828650461672975`
  - f1: `1.0`
  - fpr_at_95_recall: `0.0`
  - log_loss: `0.039038694521076445`
  - log_loss_before_temperature: `2.384991978519164`
  - pr_auc: `1.0`
  - roc_auc: `1.0`
- Timing:
  - peak_vram_bytes: `None`
  - total_score_seconds: `1.854143775854027`
- Calibration: temperature `5.0`, low `0.07959758536640082`, high `0.07959758536640082`

### `spectttra-alpha-120s`

- Model: `awsaf49/sonics-spectttra-alpha-120s` @ `094b32a5545098a71c113f6ae9d5c55310564268`
- Config hash: `2e1c68ba5b9096f74da041cd5f6d3b64c666cc72f52252e464177a561074110e`
- Tracks: scored `400`, failed `0`
- Test metrics:
  - ece: `0.07033544167746428`
  - ece_after_temperature: `0.07033544167746428`
  - f1: `0.8`
  - fpr_at_95_recall: `0.725`
  - log_loss: `0.4672049721192296`
  - pr_auc: `0.8803794193714269`
  - roc_auc: `0.853125`
- Dev metrics:
  - ece: `7.626404641640112e-05`
  - f1: `1.0`
  - fpr_at_95_recall: `0.0`
  - log_loss: `7.626695466672647e-05`
  - log_loss_before_temperature: `0.5480753733056842`
  - pr_auc: `1.0`
  - roc_auc: `1.0`
- Timing:
  - peak_vram_bytes: `None`
  - total_score_seconds: `9.446880142946611`
- Calibration: temperature `1.7000000000000002`, low `0.3811434328992681`, high `0.3811434328992681`

### `spectttra-beta-120s`

- Model: `awsaf49/sonics-spectttra-beta-120s` @ `cc49422a98b5b92700876c608d0f9cdf27214a3e`
- Config hash: `d9553f67b309aed6266d4e3ae86f1af155f0481095e321486ac5941213a30dd0`
- Tracks: scored `400`, failed `0`
- Test metrics:
  - ece: `0.15047881431305135`
  - ece_after_temperature: `0.15047881431305135`
  - f1: `0.7105263157894737`
  - fpr_at_95_recall: `0.675`
  - log_loss: `0.5890106616456668`
  - pr_auc: `0.8344883293258591`
  - roc_auc: `0.810625`
- Dev metrics:
  - ece: `0.023013444231971478`
  - f1: `1.0`
  - fpr_at_95_recall: `0.0`
  - log_loss: `0.02328238776276634`
  - log_loss_before_temperature: `1.050164635321425`
  - pr_auc: `1.0`
  - roc_auc: `1.0`
- Timing:
  - peak_vram_bytes: `None`
  - total_score_seconds: `6.01362772982975`
- Calibration: temperature `4.3`, low `0.35055614819602887`, high `0.35055614819602887`

### `spectttra-gamma-120s`

- Model: `awsaf49/sonics-spectttra-gamma-120s` @ `57659d8cb04767893537bbeb89da973e2e5a0290`
- Config hash: `85a9d71164318909bbde4041a0fd15cc5f3f3453eb2856075cdecce9401d4985`
- Tracks: scored `400`, failed `0`
- Test metrics:
  - ece: `0.09901871343713302`
  - ece_after_temperature: `0.09901871343713302`
  - f1: `0.6666666666666666`
  - fpr_at_95_recall: `0.9`
  - log_loss: `0.6403759265261`
  - pr_auc: `0.732172435516212`
  - roc_auc: `0.694375`
- Dev metrics:
  - ece: `0.0038418041728636517`
  - f1: `1.0`
  - fpr_at_95_recall: `0.0`
  - log_loss: `0.003849202858127486`
  - log_loss_before_temperature: `0.788893198076873`
  - pr_auc: `1.0`
  - roc_auc: `1.0`
- Timing:
  - peak_vram_bytes: `None`
  - total_score_seconds: `5.117734093990293`
- Calibration: temperature `2.9`, low `0.4426170976924462`, high `0.4426170976924462`

## Decision

**D1 (recorded 2026-11-19): `ast-60s` selected as the provisional default backend.**

Rationale: among the measured backends, `ast-60s` offered the lowest FPR at ~95% recall
(0.45) with solid overall discrimination (ROC-AUC 0.858, PR-AUC 0.881), reducing the risk of
falsely flagging human tracks at high sensitivity relative to alternatives. `fakeprint`
achieved higher ROC-AUC/PR-AUC (0.923/0.927) and was fastest, but showed FPR@~95% recall of
1.0 and poorer calibration on this test split; `spectttra-alpha-120s` was close on AUC with
better ECE but higher FPR@~95% recall (0.725). `ast-60s` is slower (~272s total for 400
tracks in this run) than the other backends.

This choice is **provisional and conditional** on this small, genre-skewed, unknown-generator
dataset and the bench preprocessing/calibration behavior (including B1 peak normalization and
saturated dev-fit thresholds). Thresholds and calibration should be revalidated on a more
representative set before production use. See `ai/plans/plan.md` section 5 for the gate record.
