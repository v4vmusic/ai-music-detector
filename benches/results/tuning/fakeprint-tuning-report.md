# Fakeprint tuning addendum (post-T3)

**Date:** 2026-11-19  
**Status:** benching complete, **not committed** (awaiting owner review). Phase 5 committed artifacts unchanged.  
**Backend:** `fakeprint` (`lofcz/ai-music-detector`, rev `d2180598fed79e3f917e8050a00439982466e5c6`, 10 s window @ 16 kHz, CPU/ONNX, `ai_music_detector.onnx`).

## 1. Why this was run

`fakeprint` was outside the original T1/T2 tuning scope (which covered `ast-60s` and `spectttra-alpha/beta/gamma-120s`), but Phase 5/T3 goal-aligned comparison suggested it was the strongest measured option at a conservative ~80–90% AI-recall operating point for the **v4vmusic.com** use case:

- v4vmusic.com ingests music via RSS feeds (no control over submissions). All 400 tracks in `test-audio/test-songs.csv` are v4vmusic.com songs that originated from RSS feeds.
- Goal: tag AI-generated slop as spam, catch **most/all** AI, but it is **very important not to tag legitimate human songs as spam**.
- Practical policy fit: tri-state **AI / human / unsure-needs-review** (auto-spam only high-confidence AI, route the middle to review) rather than forcing a single hard threshold at catch-all recall.

This addendum benches `fakeprint` in the same manner as T1/T2 (placement sweep + loop-fill follow-up) with the same dev-only, goal-aligned selection criterion.

## 2. Protocol (same as T1/T2)

- **Manifest/split:** `test-audio/test-songs.csv` (400 local tracks: 200 human, 200 AI), seed **42**, fixed dev/test split (80/80/40/40 per class train/dev/test per `benches/_shared/eval.py`), calibration fit on dev only. Results are **directional/conditional** (genre-skewed, unknown AI generators).
- **Selection criterion (dev only, fixed):** minimize **human FPR at AI recall = 0.85** on raw dev scores (midpoint of ~80–90% acceptable band). Tie-breaks: larger separation margin at that point, higher dev ROC-AUC, config name. FPR-at-target-recall is invariant to monotonic temperature scaling; raw scores used.
- **Held-out test:** reported **once per chosen/reported config** (not used for placement selection).
- **Native window:** 10 s (`fakeprint`). Placement is duration-relative per track: `start=0`, `middle=max(0,(D-W)/2)`, `end=max(0,D-W)`. Fill applied only when selected segment is shorter than required.
- **Configs:** placement sweep `FP-1a` start / `FP-1b` middle / `FP-1c` end (`zero-pad`), then loop-fill follow-up `FP-2a` on the dev-selected best placement.

## 3. Runs and completeness

| Config | Placement | Fill | Rows | Scored | Failed | Short padded | Loop filled | Peak-normalized | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `FP-1a` | start | zero-pad | 400 | 400 | 0 | 0 | 0 | 157 | matches Phase 5 baseline behavior (fixed offset 0s ≈ start) |
| `FP-1b` | middle | zero-pad | 400 | 400 | 0 | 0 | 0 | 246 | **dev-selected** |
| `FP-1c` | end | zero-pad | 400 | 400 | 0 | 0 | 0 | 166 | |
| `FP-2a` | middle | loop-fill | 400 | 400 | 0 | 0 | 0 | 246 | follow-up on `FP-1b`; no segments < 10 s, so no-op |

No tracks in this manifest have a selected segment shorter than 10 s (0 `short_track_padded` / 0 `loop_filled` in all runs). Raw per-track JSONL (git-ignored): `benches/results/tuning/<CFG>/raw/fakeprint.jsonl`.

## 4. Placement sweep (goal-aligned, dev-selected)

| Config | Dev FPR@0.85 (raw) | Dev AI recall | Dev margin | Dev ROC-AUC (raw) | Dev rank | Held-out human FP / 40 | Held-out AI FN / 40 | Test ROC-AUC | Test PR-AUC | Test FPR@95% recall | Test ECE | Test F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `FP-1a` (start) | 0.00625 | 0.85 | -0.947496 | 0.9683 | 2 | 1 | 8 | 0.9231 | 0.9271 | 1.0000 | 0.3240 | 0.8767 |
| **`FP-1b` (middle)** | **0.00000** | **0.85** | **+0.000039** | **0.9747** | **1** | **0** | **5** | **0.9647** | **0.9756** | **0.1500** | **0.2857** | **0.9091** |
| `FP-1c` (end) | 0.10625 | 0.85 | -0.998899 | 0.9121 | 3 | 2 | 6 | 0.9244 | 0.9152 | 0.2000 | 0.2288 | 0.8675 |

**Selected:** `FP-1b` (middle, zero-pad). At the dev-derived 85%-recall raw-score operating point (threshold `4.1395e-05`, dev separation margin positive by ~3.9e-5), held-out test has **0 human false positives** and **5 AI false negatives** out of 40/40. Dev selection was unambiguous by the criterion (lowest dev FPR@0.85).

## 5. Loop-fill follow-up

Compared `FP-2a` (loop-fill, middle) vs `FP-1b` (zero-pad, middle): **identical** dev and held-out metrics (no segments required filling). Loop-fill has no effect for `fakeprint` on this manifest/window.

## 6. Comparison to Phase 5 baseline and prior tuning

Phase 5 baseline `fakeprint` (fixed offset 0s, raw `benches/results/raw/fakeprint.jsonl`) is equivalent to `FP-1a` on this split:

| Config | Dev FPR@0.85 | Held-out FP / FN @ goal point | Test ROC-AUC | Test FPR@95% recall | Test ECE | Score time (400, CPU) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 5 baseline / `FP-1a` (start, 0s) | 0.00625 | 1 / 8 | 0.9231 | 1.0000 | 0.3240 | ~1.85 s |
| **`FP-1b` (middle, selected)** | **0.00000** | **0 / 5** | **0.9647** | **0.1500** | 0.2857 | (fast, CPU) |
| `FP-1c` (end) | 0.10625 | 2 / 6 | 0.9244 | 0.2000 | 0.2288 | (fast, CPU) |

Context vs tuned in-scope backends (from `benches/results/tuning/report.md`, goal-aligned dev-derived 85% point, held-out 40/40): `ast-60s` AST-1c/AST-2a 8 FP / 6 FN (baseline ast 13/8, but FPR@95% recall 0.55 worse than baseline 0.45), `spectttra-alpha` STA-1b 9/4 (baseline 8/8), beta/gamma 20/3 and 22/6. At the **~85% recall operating point**, `FP-1b` is the best measured result on this set (fewest human false flags while keeping AI FNs low), and it also improves over baseline fakeprint/start materially.

## 7. Interpretation for v4vmusic (RSS spam-filtering)

- **Supports a conservative “protect humans” policy at ~85% AI recall:** `FP-1b` (middle 10 s window) yields 0/40 held-out human FPs and 5/40 AI FNs at the dev-derived goal point — the strongest evidence here for “catch most AI slop without auto-tagging legitimate human songs.”
- **Do not equate this with safe catch-all detection:** `FPR@95% recall` is 0.15 (6/40 human FPs in that metric regime) and **calibration is weak** (ECE ~0.286; raw scores are probabilities from the model but not well-calibrated on this split). Pushing recall toward “catch all AI” can reintroduce human false flags. The earlier observation that fakeprint “breaks if pushed to near-catch-all” remains consistent.
- **Tri-state recommendation (operational):** auto-tag as AI/spam only at a **high-confidence** threshold; treat the middle band as **unsure-needs-review**; pass clearly-human scores. This reconciles “catch most AI” with “don’t tag legitimate humans.” Thresholds for these bands should be chosen/validated from score distributions (see §8).
- **Middle placement matters for fakeprint here:** unlike baseline/start, middle reduced both held-out human FPs and AI FNs at the goal point on this set; end was worse. This is a directional finding specific to this manifest.

## 8. Next steps (no reruns required for threshold analysis)

- **Raw scores available for sweet-spot/tri-state threshold analysis:** `benches/results/tuning/FP-1b/raw/fakeprint.jsonl` (selected) and `benches/results/raw/fakeprint.jsonl` (baseline). Analyze per-track `raw_score`/label distributions on dev (to pick thresholds) and report once on held-out test to define AI / unsure / human cutoffs aligned with “protect humans.”
- **If stronger evidence needed:** validate on a larger, more representative RSS/v4vmusic sample and additional generators; consider cross-validation/stability checks (small held-out set => FP/FN counts are fragile).
- **Out of scope here:** multi-window aggregation (deferred), preprocessing changes, fakeprint model changes, or committing these benching artifacts without owner review.

## 9. Artifacts

| Artifact | Path | State |
| --- | --- | --- |
| Placement summary (JSON/MD) | `benches/results/tuning/fakeprint-placement-summary.{json,md}` | uncommitted (new) |
| Loop-fill summary (JSON/MD) | `benches/results/tuning/fakeprint-loop-fill-summary.{json,md}` | uncommitted (new) |
| This report | `benches/results/tuning/fakeprint-tuning-report.md` | uncommitted (new) |
| Raw JSONL (per config) | `benches/results/tuning/FP-1a,FP-1b,FP-1c,FP-2a/raw/fakeprint.jsonl` | git-ignored |
| Phase 5 baseline (reference) | `benches/results/raw/fakeprint.jsonl`, `benches/results/summary.json` | unchanged (raw git-ignored) |
| Prior consolidated tuning report | `benches/results/tuning/report.md`, `benches/results/tuning/summary.json` | committed (unchanged by this addendum unless owner requests integration) |

## 10. Caveats

- 400 tracks total (40 human / 40 AI held-out); single fixed seed-42 split. Differences of a few FP/FN are **not** robust — treat as directional.
- Human set genre-skewed / podcast-derived characteristics as noted in prior docs (though owner-stated all are v4vmusic.com RSS-sourced songs); AI generators unknown/unlabeled.
- Dev calibration saturated on this split for some evaluators in prior phases; goal criterion correctly uses raw dev scores, but the small split limits confidence.
- `FP-1b`’s dev-derived threshold margin is extremely small (+3.9e-5), meaning the operating point is sensitive — another reason to prefer a tri-state/review band and to revalidate thresholds on fresh data rather than hard-coding this exact cutoff.
