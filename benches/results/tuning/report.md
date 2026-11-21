# AI Music Detector — Tuning Report (Pre–Phase 6, T3)

**Date:** 2026-11-19  
**Status:** consolidated analysis complete; **T-D1/D1 resolved 2026-11-19** — owner changed D1 to `fakeprint` primary (`FP-1b`) + `spectttra-alpha-120s` second opinion (`STA-1b`); recorded in `ai/plans/plan.md` §5.

**Scope:** pre–Phase 6 benchmark tuning experiments (T0–T2) compared against the Phase 5 baseline, to support the owner’s D1 reaffirm/change decision before Phase 6. No Phase 5 committed baseline artifacts were modified.

## 1. Scope and recorded decisions

- **Primary goal:** minimize **human false flags at an explicitly acceptable AI-recall level** (address Phase 5 `ast-60s` baseline **FPR@~95% recall = 0.45**, 18/40 human test tracks). Accept **moderate AI-recall reduction (~80–90%)** if it meaningfully reduces human false flags. (Source: `ai/reference/test-tuning-discussion.md`, `ai/plans/test-tuning-plan.md` §1.)
- **Backends in scope for tuning:** `ast-60s`, `spectttra-alpha-120s`, `spectttra-beta-120s`, `spectttra-gamma-120s` (native windows 60s / 120s / 120s / 120s). `fakeprint` was **not** in the tuning sweep scope but is included in baseline comparison as measured Phase 5 evidence.
- **Experiments run:** placement sweeps `start/middle/end` with `zero-pad` for all four in-scope backends (T1: `AST/STA/STB/STG-1a/1b/1c`), then loop-fill follow-ups on each backend’s dev-selected placement (T2: `AST/STA/STB/STG-2a`). No preprocessing/feature-parameter changes; no multi-window aggregation (deferred).
- **Dataset/constraints (directional):** `test-audio/test-songs.csv`, 400 local tracks (200 human, 200 AI), seed 42, fixed dev/test split (per Phase 5/eval policy), genre-skewed human set, unknown AI generators. Results are conditional, not a general detection-rate claim.
- **Artifact separation:** tuning outputs live under `benches/results/tuning/` (raw JSONL git-ignored per config; summaries/reports committed). Phase 5 baseline remains `benches/results/calibration.json`, `benches/results/summary.json`, `benches/results/report.md` (unchanged).

## 2. Protocol and selection criterion (strict)

- **Fixed before runs (T1/T2):** minimize **human FPR at AI recall = 0.85** (raw dev scores), the midpoint of the ~80–90% acceptable tradeoff band. Tie-breaks: larger separation margin at that point, then higher dev ROC-AUC, then config name. FPR-at-target-recall is invariant to monotonic temperature scaling, so raw scores were used. (`benches/_shared/selection.py`, `benches/placement_sweep.py`, `benches/loop_fill_followup.py`.)
- **Discipline:** select best placement per backend on **dev only** (T1) and compare loop-fill follow-ups on **dev only** under the same goal framework (T2). **Held-out test reported once per chosen/reported config.** T1 selected placements: `ast-60s=end` (AST-1c), `spectttra-alpha-120s=middle` (STA-1b), `spectttra-beta-120s=middle` (STB-1b), `spectttra-gamma-120s=start` (STG-1a).
- **Calibration note:** Phase 5 dev separation saturated on this split (collapsed thresholds / saturated post-calibration dev metrics). The goal criterion uses **raw-score** dev FPR@0.85 and is therefore not the same as the evaluator’s post-calibration `dev_metrics`; this is reported explicitly in T1/T2 artifacts and below.
- **Goal-point baseline comparison (this report):** for comparability with the tuning goal, Phase 5 baseline configs are also evaluated at their **own dev-derived 85% AI-recall raw-score operating point** on the same fixed seed-42 split (computed from `benches/results/raw/<backend>.jsonl`). Held-out test FP/FN counts at that point are reported as a goal-aligned comparison metric; this does not re-select baselines and does not use test results to choose among tuning configs. This is distinct from Phase 5’s reported operating point (FPR@~95% recall / exported thresholds).

## 3. Configs run and completeness

| Track | Configs | Status | Evidence |
| --- | --- | --- | --- |
| Placement sweeps (T1), zero-pad | AST-1a/b/c, STA-1a/b/c, STB-1a/b/c, STG-1a/b/c (12 configs) | 12/12 complete, **400/400 rows, 0 failures** each | `benches/results/tuning/placement-sweep-summary.{json,md}`, raw `benches/results/tuning/<CFG>/raw/<backend>.jsonl` (git-ignored) |
| Loop-fill follow-ups (T2), on T1-selected placements | AST-2a (end), STA-2a (middle), STB-2a (middle), STG-2a (start) | 4/4 complete, **400/400 rows, 0 failures** each; STA-2a rerun cleanly after a timeout-interrupted partial run | `benches/results/tuning/loop-fill-summary.{json,md}`, raw `benches/results/tuning/<CFG>/raw/<backend>.jsonl` (git-ignored) |

Short-segment fill affected **5/400** AST tracks (4 human, 1 AI, <60s) and **26/400** per SpecTTTra config (17 human, 9 AI, <120s), label-skewed toward human.

## 4. Phase 5 baseline (reference)

Source: `benches/results/summary.json`, `benches/results/report.md`, `benches/results/calibration.json` (committed, seed 42, fixed dev/test, B1 peak normalization). D1 was provisionally recorded as `ast-60s` on the basis of lowest FPR@~95% recall among available backends; that provisional choice was superseded on 2026-11-19 by the owner decision recorded in `ai/plans/plan.md` §5.

| Backend (baseline, fixed 25s, zero-pad) | Test ROC-AUC | Test PR-AUC | Test FPR@~95% recall | Test ECE | Total score time | Test human FP / AI FN at **dev-derived 85% recall point** (goal-aligned, this report) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fakeprint` | 0.9231 | 0.9271 | **1.000** | 0.3240 | ~1.85s | **1 / 8** (of 40/40) |
| `ast-60s` (provisional D1; superseded) | 0.8578 | 0.8810 | 0.450 | 0.1569 | ~272.4s | 13 / 8 |
| `spectttra-alpha-120s` | 0.8531 | 0.8804 | 0.725 | 0.0703 | ~9.45s | 8 / 8 |
| `spectttra-beta-120s` | 0.8106 | 0.8345 | 0.675 | 0.1505 | ~6.01s | 22 / 3 |
| `spectttra-gamma-120s` | 0.6944 | 0.7322 | 0.900 | 0.0990 | ~5.12s | 25 / 7 |

Goal-point thresholds/counts above are derived from each baseline’s dev raw scores at 85% AI recall on the fixed split (raw `benches/results/raw/*.jsonl`): `fakeprint` dev FPR@0.85 0.006, `ast-60s` 0.425, `spectttra-alpha` 0.275, `spectttra-beta` 0.494, `spectttra-gamma` 0.575.

## 5. Placement sweep results (T1, goal-aligned, dev-selected)

Held-out test below is reported **once per dev-selected config** (not used for placement selection). Dev criterion values are raw-score.

| Backend | Baseline dev FPR@0.85 (fixed 25s) | Selected config (placement) | Selected dev FPR@0.85 | Δ dev FPR@0.85 vs baseline | Test human FP / AI FN @ dev 85% point | Test ROC-AUC | Test FPR@95% recall | Test ECE | Notes |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ast-60s` | 0.425 | **AST-1c (end)** | **0.2125** | **-0.2125** | **8 / 6** (baseline 13 / 8) | 0.8756 | 0.550 | 0.0861 | Fewer human FPs at ~85% recall, but **worse** FPR@95% recall than baseline (0.55 vs 0.45). |
| `spectttra-alpha-120s` | 0.275 | **STA-1b (middle)** | **0.2562** | -0.0188 | 9 / 4 (baseline 8 / 8) | 0.8806 | 0.525 | 0.1373 | Small dev gain; held-out human FPs slightly **worse** (8→9) at goal point, AI FNs fewer (8→4). Mixed. |
| `spectttra-beta-120s` | 0.494 | **STB-1b (middle)** | **0.4500** | -0.0438 | 20 / 3 (baseline 22 / 3) | 0.8356 | 0.500 | 0.1400 | Modest improvement; still high human false flags. |
| `spectttra-gamma-120s` | 0.575 | **STG-1a (start)** | **0.5437** | -0.0313 | 22 / 6 (baseline 25 / 7) | 0.7356 | 0.725 | 0.0833 | Weakest backend overall; small improvement, poor discrimination remains. |

All 12 configs scored 400/400 with 0 failures. Placement was impactful for `ast-60s` on dev (end clearly best; middle worst, dev FPR@0.85 0.4125) and mattered less/ambiguously for SpecTTTra variants. Full per-config dev rankings and distributions: `placement-sweep-summary.{json,md}`.

## 6. Loop-fill follow-up results (T2, goal-aligned comparison)

Loop-fill applied only to short segments (simple repeat, no crossfade) on each T1-selected placement. Held-out test reported once per reported config; selected zero-pad test metrics are the T1-reported values shown side-by-side.

| Backend (placement) | Zero-pad config | Loop-fill config | Δ dev FPR@0.85 (LF − ZP) | Δ dev ROC-AUC (raw) | Held-out human FP / AI FN @ goal point (ZP → LF) | Δ test ROC-AUC | Δ test FPR@95% recall | Fill affected (H / AI) |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| `ast-60s` (end) | AST-1c | AST-2a | 0.0000 | -0.0004 | 8/6 → 8/6 (unchanged) | 0.0000 | 0.0000 | 4 / 1 (5 total) |
| `spectttra-alpha-120s` (middle) | STA-1b | STA-2a | **-0.0500** | +0.0185 | 9/4 → 9/4 (unchanged) | 0.0000 | 0.0000 | 17 / 9 (26) |
| `spectttra-beta-120s` (middle) | STB-1b | STB-2a | -0.0312 | +0.0189 | 20/3 → 20/3 (unchanged) | +0.0019 | 0.0000 | 17 / 9 (26) |
| `spectttra-gamma-120s` (start) | STG-1a | STG-2a | -0.0437 | +0.0108 | 22/6 → 22/6 (unchanged) | +0.0031 | 0.0000 | 17 / 9 (26) |

**Interpretation (mixed/negative for the goal):** loop-fill improved the **dev** goal metric for SpecTTTra configs but produced **no reduction in held-out human false positives** at the dev-derived operating point (and negligible/zero test-metric change). AST showed no meaningful change. Given the small, label-skewed set of short tracks (especially 17 human / 9 AI among 26 affected SpecTTTra segments), loop-fill impact is limited and should be treated as directional, not a reliable win.

## 7. Consolidated comparison and key findings (primary goal)

- **Best goal-aligned held-out result (min human FP at ~85% AI recall, dev-derived point) among all measured configs:** **`fakeprint` baseline** — **1/40** human FP, 8/40 AI FN (test ROC-AUC 0.923, but FPR@95% recall 1.000, ECE 0.324, CPU-fast). This backend was **not** placement/fill-tuned in T1/T2.
- **Best tuned in-scope result:** **`ast-60s` AST-1c / AST-2a (end, 60s)** — **8/40** human FP, 6/40 AI FN at the goal point, improved from baseline `ast-60s` 13/40 FP; however **test FPR@95% recall worsened** (0.55 vs 0.45) and ROC-AUC gain was modest (0.876 vs 0.858). Loop-fill added no held-out benefit.
- **`spectttra-alpha-120s`:** baseline already competitive at the goal point (8/40 FP); selected STA-1b was **mixed** (9/40 FP but fewer AI FN, 4/40). STA-2a did not reduce held-out FPs. Overall discrimination similar to baseline/tuned AST but with different tradeoffs (lower FPR@95% recall performance than AST baseline in Phase 5 terms, better ECE in baseline).
- **`spectttra-beta/gamma`:** remained clearly behind on the primary goal (20 and 22 human FPs at best tuned goal point) despite small tuning gains.
- **Operating-point dependence:** several backends rank differently at ~85% recall vs ~95% recall. The provisional D1 (`ast-60s`) was justified in Phase 5 primarily by **FPR@~95% recall**; under the tuning plan’s stated goal (~80–90% recall), the evidence shifted, and the owner subsequently selected `fakeprint` + a `spectttra-alpha` second opinion (plan §5).

## 8. Limitations and caveats

- **Directional only:** 400 tracks, genre-skewed human set (podcast-derived), untargeted/unknown-generator AI set; small dev/test counts (40/40 per class in held-out test) mean individual FP/FN differences are fragile.
- **Protocol:** fixed seed-42 split; dev calibration/threshold behavior saturated (collapsed thresholds) — raw-score goal metrics avoid post-calibration saturation but the split remains small. No cross-validation of the tuning outcome beyond this protocol.
- **Comparability:** baseline runs used fixed 25s offsets and Phase 5 preprocessing (B1 peak normalization where applicable); tuning varied placement/fill only as specified. SpecTTTra/fakeprint score scales differ (logits vs probabilities) — goal-point comparisons use each config’s own dev-derived raw threshold, which is appropriate for operating-point comparison but not a single global threshold.
- **Not tested:** multi-window aggregation (deferred), fakeprint placement/fill tuning, alternative fill methods, preprocessing changes, and other datasets/generators.
- **Reproducibility:** raw per-track JSONL for all reported configs is retained locally under `benches/results/tuning/<CFG>/raw/` (git-ignored) and Phase 5 raw under `benches/results/raw/` (git-ignored); committed summaries/reports and config hashes provide traceability.

## 9. Recommendation (evidence-based, for owner decision)

**Primary goal considered:** minimize human false flags at acceptable AI recall (~80–90%).

- **Evidence favors reconsidering provisional D1 if operating at ~85% recall:** `fakeprint` baseline achieves **1/40** held-out human FPs at its dev-derived 85% recall point on this set, far fewer than any tuned in-scope config (`ast-60s` end: 8/40). On the stated tuning goal, this is the strongest measured result.
- **Reasons for caution / possible reaffirmation of `ast-60s`:** `fakeprint` performed poorly at **~95% recall** (FPR@95% recall 1.0 in Phase 5), has weak calibration (ECE 0.324), and was not evaluated under placement/fill tuning; its advantage may be operating-point- and dataset-specific. `ast-60s` remains the better choice if the product priority is **high recall (~95%)** over the ~85% tradeoff, though tuning shows end-placement improves its ~85% behavior relative to its own baseline at the cost of worse 95%-recall FPR.
- **Suggested options:** (1) **Change D1** to `fakeprint` as default for a goal-aligned ~80–90% recall operating point, with explicit caveats and a requirement to revalidate on a more representative set; (2) **Change D1** to tuned `ast-60s` **AST-1c (end, zero-pad)** if preferring the in-scope tuned model with improved ~85% behavior and accepting its 95%-recall regression vs baseline AST; (3) **Reaffirm provisional D1** as baseline `ast-60s` only if ~95% recall remains the overriding priority (note: misaligned with the tuning goal); (4) **Keep D1 pending/provisional** pending broader validation. The report does not make this decision.

## 10. Artifact traceability

| Artifact | Path |
| --- | --- |
| This report | `benches/results/tuning/report.md` |
| Consolidated machine-readable summary | `benches/results/tuning/summary.json` |
| Placement sweep summary | `benches/results/tuning/placement-sweep-summary.{json,md}` |
| Loop-fill summary | `benches/results/tuning/loop-fill-summary.{json,md}` |
| Tuning raw JSONL (git-ignored) | `benches/results/tuning/<CONFIG>/raw/<backend>.jsonl` |
| Phase 5 baseline (committed, unchanged) | `benches/results/calibration.json`, `benches/results/summary.json`, `benches/results/report.md` |
| Phase 5 raw JSONL (git-ignored) | `benches/results/raw/<backend>.jsonl` |
| Decisions/discussion | `ai/reference/test-tuning-discussion.md`, `ai/plans/test-tuning-plan.md` |

## 11. D1 review questionnaire (resolved 2026-11-19; historical record)

**T-D1 (resolved 2026-11-19):** the owner changed **D1** from the provisional `ast-60s` to **`fakeprint` primary (`FP-1b`) + `spectttra-alpha-120s` second opinion (`STA-1b`)**, recorded in `ai/plans/plan.md` §5. The questionnaire below is retained as the historical decision record.

```markdown
### 1. Which default backend/config should be D1 before Phase 6, given the primary goal (minimize human false flags at acceptable AI recall ~80–90%)?
*Goal-aligned held-out counts at each config’s dev-derived 85% recall point (40 human / 40 AI): fakeprint baseline 1 FP / 8 FN; ast-60s AST-1c end 8 / 6 (baseline ast 13 / 8, but worse FPR@95% recall 0.55 vs 0.45); spectttra-alpha STA-1b 9 / 4 (baseline 8 / 8). All results directional on a small, genre-skewed, unknown-generator set.*
- [x] **Change D1 to `fakeprint` (Phase 5 baseline, CPU, ~15 kB) for a ~80–90% recall operating point (Recommended for the stated tuning goal).** Best measured goal-point result (1 human FP) but poor at ~95% recall (FPR@95% 1.0) and weak calibration (ECE 0.324); not placement/fill-tuned; revalidate before production use. **Selected 2026-11-19, with `spectttra-alpha-120s` STA-1b as the recommended second opinion.**
- [ ] Change D1 to `ast-60s` tuned **AST-1c** (end placement, 60s, zero-pad; AST-2a equivalent at goal point). Improves ast goal-point FPs (13→8) vs baseline ast with solid ROC-AUC, but regresses FPR@95% recall (0.55 vs 0.45) and is slower (~272s/400).
- [ ] Reaffirm provisional D1 as **`ast-60s` baseline** (fixed 25s, Phase 5 config), prioritizing ~95% recall behavior over the ~80–90% tuning goal.
- [ ] Choose a different backend/config (specify): ___ (e.g. `spectttra-alpha-120s` baseline or STA-1b).
- [ ] Keep D1 provisional/pending — evidence inconclusive; require broader validation before choosing.
- [ ] Other: ___
```