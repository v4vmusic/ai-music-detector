# Benchmark test-tuning discussion (pre–Phase 6)

- **Date:** 2026-11-19
- **Context:** Phase 5 comparison and D1 evidence are complete and committed (`0feae6a`). Gate **D1 is provisionally recorded as `ast-60s`** in `ai/plans/plan.md` §5 based on `benches/results/{calibration,summary,report}.md` (400 tracks, seed 42, fixed dev/test, 5/5 backends scored 400/400).
- **Purpose of this doc:** record tuning decisions from discussion and define an experiment matrix to improve benchmark results before moving on to Phase 6.

## Goal

Experiment in benchmark tests to get the **best defensible results/evidence** before moving on to Phase 6 of the plan.

- **Primary optimization goal:** minimize **human false flags at an explicitly acceptable AI-recall level** (address the high false-positive rate observed for `ast-60s`: **FPR@~95% recall = 0.45** on the held-out test split, i.e. 18/40 human test tracks at that operating point).
- **Recall tradeoff:** accept **moderate AI-recall reduction (e.g. ~80–90%)** if it meaningfully reduces human false flags.
- **Constraint:** treat improvements as directional given the small, genre-skewed, unknown-generator dataset; understand tradeoffs (recall, calibration, speed, generalization).

## Decisions (recorded)

| Date | Decision | Notes |
| --- | --- | --- |
| 2026-11-19 | **Extend pre–Phase 6 work with benchmark tuning experiments** before starting Phase 6 implementation. | Improve measured results/confidence before Phase 6. |
| 2026-11-19 | **Primary goal: minimize human false flags at an explicitly acceptable AI-recall level.** | Accept moderate AI-recall reduction (e.g. ~80–90%) for meaningfully fewer human false flags. |
| 2026-11-19 | **Keep `ast-60s` at its native 60s window** for tuning experiments. | Do not treat 90s/120s as a knob for this checkpoint (model/extractor/spec configured for 60s, `feature_max_length=6000`). |
| 2026-11-19 | **First-pass priority:** run `ast-60s` and `spectttra-alpha-120s` placement sweeps first, then fill follow-ups on best placements. | Placement is the highest-impact initial knob; isolate effects before follow-ups. |
| 2026-11-19 | **Evaluate start/middle/end window-placement strategies** (all three, duration-relative per track). | `start=0`, `middle=max(0,(D-W)/2)`, `end=max(0,D-W)` to reduce unnecessary padding and test segment sensitivity. Proper middle/end require per-track offsets (current runner uses one fixed `--offset`). |
| 2026-11-19 | **Skip `ast-60s` preprocessing/feature-parameter experiments.** | Avoid deviating from training-compatible settings (`num_mel_bins`, `max_length`, normalization). |
| 2026-11-19 | **Keep SpecTTTra preprocessing at baseline only.** | Stay aligned with model-expected `sonics`/runner preprocessing while testing placement/fill. |
| 2026-11-19 | **Use simple loop/repeat non-zero fill (no crossfade) for short segments.** | Deterministic first non-zero-fill baseline vs zero-padding; repeat available audio to fill the window. |
| 2026-11-19 | **Apply non-zero fill only when the selected window segment is shorter than required** (would otherwise be padded). | Targets the padding issue without changing inputs that already fill the window. |
| 2026-11-19 | **Defer multi-window aggregation until after single-window placement and fill sweeps.** | Isolate single-segment/config effects first. |
| 2026-11-19 | **Include `spectttra-beta-120s` and `spectttra-gamma-120s` in tuning scope** (in addition to `ast-60s` and `spectttra-alpha-120s`). | Broader comparison across available backends despite lower Phase 5 baselines for beta/gamma. |
| 2026-11-19 | **Evaluation discipline:** strict — select configs/operating points on dev (or predefined criteria) and report held-out test once per chosen config. | Produce defensible comparisons before Phase 6; avoid tuning on held-out test results. |
| 2026-11-19 | **Be open to reaffirming or changing provisional D1 (`ast-60s`)** based on comparative tuning evidence before Phase 6. | Default should follow the best available measured results for the chosen goal. |
| 2026-11-19 | **Keep Phase 5 committed artifacts unchanged; write experiment outputs separately with clear config naming.** | Preserve baseline/D1 record and make comparisons traceable. |

## Baseline and motivation

- **Manifest:** `test-audio/test-songs.csv` — 400 tracks (200 human, 200 AI). Human tracks are podcast-derived; AI tracks untargeted; generator identity unknown. Results are directional/conditional.
- **Current `ast-60s` baseline (seed 42, fixed dev/test, 60s@25s, B1 peak normalization):** ROC-AUC 0.858, PR-AUC 0.881, **FPR@~95% recall 0.45**, F1 0.763, ECE 0.157, ~272s/400 (cuda). Calibration exported as temp 5.0, low/high 0.8459 (collapsed), with saturated dev fit.
- **SpecTTTra Phase 5 baselines (120s@25s):** alpha ROC-AUC 0.853 / PR-AUC 0.880 / FPR@~95% recall 0.725 / ECE 0.070 / ~9.4s; beta 0.811 / 0.834 / 0.675 / 0.150 / ~6.0s; gamma 0.694 / 0.732 / 0.900 / 0.099 / ~5.1s.
- **Observed issue:** at the ~95%-AI-recall operating point on the held-out test split (40 human / 40 AI), `ast-60s` produced 18/40 human false positives. Raw `ast-60s` scores are highly bimodal overall (human median ~0.841, p75 ~0.999, p25 ~0.003), so many humans are scored confidently AI-like on this set; lowering human false flags may require accepting moderate recall reduction (target range ~80–90% as a guide).
- **Per-track evidence:** Phase 5 raw results in `benches/results/raw/*.jsonl` (git-ignored), including `ast-60s.jsonl` and SpecTTTra variant files.

### Track duration facts (relevant to windowing/padding)

Measured from manifest audio (ffprobe): 400 tracks, min ~35.9s, median ~212.6s, max ~790.6s.

| Window | Tracks < window | Tracks >= window | Notes |
| ---: | ---: | ---: | --- |
| 60s (`ast-60s`) | 5 (4 human, 1 AI) | 395 | Most tracks can provide a full 60s segment with a suitable offset; <60s require fill/padding for any strategy. |
| 120s (`spectttra-*`) | 26 (17 human, 9 AI) | 374 | ~6.5% require fill/padding for 120s (label-skewed toward human in this subset). |
| 90s | 14 | 386 | (context only) |
| 180s | 116 | 284 | (context) |

Current behavior (`benches/_shared/audio.py:decode_window`): slices `[offset, offset+window)` and **zero-pads** if short, emitting `short_track_padded`; windows with peak > 1.0 are peak-normalized with `peak_normalized_to_fit_range`.

## Experiment matrix

### Scope and fixed conditions (unless an experiment varies them)

- **Backends in scope:** `ast-60s`, `spectttra-alpha-120s`, `spectttra-beta-120s`, `spectttra-gamma-120s`.
- **Manifest:** `test-audio/test-songs.csv` (400 labeled tracks).
- **Seed/splits:** seed 42; same split/eval protocol as Phase 5 (`fixed_dev_test` at current class sizes, calibration fit on dev only, metrics reported per eval methodology). Record strategy and any deviations.
- **Device:** `cuda` (where applicable), one variant/config at a time; record timings/versions/config hash.
- **Comparability:** change one factor at a time where practical; keep model revision, window length (native), and preprocessing baseline fixed except for the variable under test.
- **Artifacts:** save raw per-track JSONL per (backend, config) with unambiguous naming in a separate experiment location (do **not** overwrite Phase 5 committed artifacts in `benches/results/`).
- **Priority/order:** run `ast-60s` and `spectttra-alpha-120s` placement sweeps first, then fill follow-ups on best placements; include `spectttra-beta-120s` and `spectttra-gamma-120s` placement sweeps (and corresponding fill follow-ups) in scope as well.

### A. Window placement strategies (per-track, duration-relative)

Applies to all in-scope backends (`ast-60s`: 60s, SpecTTTra variants: 120s). Define `W = window_seconds`, `D = track duration`:

| Strategy ID | Offset rule | Rationale | Padding/fill risk |
| --- | --- | --- | --- |
| `start` | `0` | Beginning-of-track content | Low when `D >= W`; requires fill/padding when `D < W` |
| `middle` | `max(0, (D - W) / 2)` | Center segment, avoids trailing/leading bias | Minimal extra fill/padding beyond the `D < W` case |
| `end` | `max(0, D - W)` | Final segment | Minimal extra fill/padding beyond the `D < W` case |
| `fixed-25s` (baseline ref) | `25` (Phase 5 default) | Baseline for comparison to committed Phase 5 behavior | Can require more fill/padding than middle/end on shorter tracks if offset exceeds available tail |

**Note:** current runner takes a single fixed `--offset` for all tracks; `middle/end` require per-track offset computation (implementation prerequisite).

### B. Short-segment fill strategies

**Selected non-zero method:** `loop-fill` = simple loop/repeat of available audio to fill the window (**no crossfade**), deterministic.
**Scope:** apply only when the selected window segment is shorter than required (would otherwise be padded).

| Strategy ID | Behavior | Pros | Cons / caveats |
| --- | --- | --- | --- |
| `zero-pad` (baseline) | Pad remainder with zeros (current) | Simple, deterministic, current behavior | Zero/silence tail may be out-of-distribution |
| `loop-fill` (selected) | Loop/repeat available audio to fill `W` (no crossfade), only when segment < `W` | Avoids zero tail; window contains audio throughout | Artificial repetition/seams; not “real” longer content; must be evaluated empirically |

**Implementation prerequisite:** add selectable fill mode in decode/window construction and record it in metadata/warnings/config. Track/count affected cases (and existing `short_track_padded` semantics or an explicit fill warning) per config.

### C. Model-specific experiment sets

#### C1. `ast-60s` (60s native window)

| Exp | Window | Placement | Fill | Preprocessing | Purpose |
| --- | --- | --- | --- | --- | --- |
| AST-B0 | 60s | `fixed-25s` | `zero-pad` | Baseline (spec defaults) | Reference (compare to Phase 5) |
| AST-1a | 60s | `start` | `zero-pad` | Baseline | Segment sensitivity |
| AST-1b | 60s | `middle` | `zero-pad` | Baseline | Segment sensitivity / padding-minimizing placement |
| AST-1c | 60s | `end` | `zero-pad` | Baseline | Segment sensitivity |
| AST-2a | 60s | best of AST-1* (by dev/predefined criterion) | `loop-fill` | Baseline | Test non-zero fill for <60s cases (5 tracks) |

**Note:** preprocessing/feature-parameter experiments for `ast-60s` are **out of scope** (skipped by decision).

#### C2. `spectttra-alpha-120s` (120s native window)

| Exp | Window | Placement | Fill | Preprocessing | Purpose |
| --- | --- | --- | --- | --- | --- |
| STA-B0 | 120s | `fixed-25s` | `zero-pad` | Baseline (`sonics-1`, runner std-norm as today) | Reference (compare to Phase 5) |
| STA-1a | 120s | `start` | `zero-pad` | Baseline | Segment sensitivity |
| STA-1b | 120s | `middle` | `zero-pad` | Baseline | Segment sensitivity / padding-minimizing placement |
| STA-1c | 120s | `end` | `zero-pad` | Baseline | Segment sensitivity |
| STA-2a | 120s | best of STA-1* (by dev/predefined criterion) | `loop-fill` | Baseline | Compare loop-fill vs zero-pad for <120s cases (26 tracks total) |

#### C3. `spectttra-beta-120s` (120s native window)

| Exp | Window | Placement | Fill | Preprocessing | Purpose |
| --- | --- | --- | --- | --- | --- |
| STB-B0 | 120s | `fixed-25s` | `zero-pad` | Baseline | Reference (compare to Phase 5) |
| STB-1a | 120s | `start` | `zero-pad` | Baseline | Segment sensitivity |
| STB-1b | 120s | `middle` | `zero-pad` | Baseline | Segment sensitivity / padding-minimizing placement |
| STB-1c | 120s | `end` | `zero-pad` | Baseline | Segment sensitivity |
| STB-2a | 120s | best of STB-1* (by dev/predefined criterion) | `loop-fill` | Baseline | Compare loop-fill vs zero-pad for <120s cases |

#### C4. `spectttra-gamma-120s` (120s native window)

| Exp | Window | Placement | Fill | Preprocessing | Purpose |
| --- | --- | --- | --- | --- | --- |
| STG-B0 | 120s | `fixed-25s` | `zero-pad` | Baseline | Reference (compare to Phase 5) |
| STG-1a | 120s | `start` | `zero-pad` | Baseline | Segment sensitivity |
| STG-1b | 120s | `middle` | `zero-pad` | Baseline | Segment sensitivity / padding-minimizing placement |
| STG-1c | 120s | `end` | `zero-pad` | Baseline | Segment sensitivity |
| STG-2a | 120s | best of STG-1* (by dev/predefined criterion) | `loop-fill` | Baseline | Compare loop-fill vs zero-pad for <120s cases |

**SpecTTTra notes:** keep native 120s windows and baseline preprocessing for all variants.

#### C5. Follow-up (deferred)

| Exp | Idea | Purpose |
| --- | --- | --- |
| AGG-1 | Multi-window aggregation (e.g. combine start/middle/end scores) for best single-config backend(s) | Reduce segment noise / improve robustness (requires implementation); **defer until after single-window placement and fill sweeps**. |

## Evaluation protocol and metrics

- **Primary goal metric:** minimize human false flags at an explicitly acceptable AI-recall level. Use a predefined selection criterion aligned with this goal (e.g. lowest human FPR at an acceptable AI-recall operating point, with **~80–90% AI recall** as the target tradeoff range for “meaningfully fewer human false flags”).
- **Reported metrics:** ROC-AUC, PR-AUC, FPR at the chosen operating point(s) (including, as relevant, FPR@~95% recall for comparability to baseline), AI recall, precision/F1 as applicable, and ECE (plus post-calibration ECE where applicable). Report operating-point tradeoffs/curves as needed.
- **Thresholding:** compare configs at consistent, goal-aligned operating points. “Keep threshold same” is useful for A/B only when score scale/calibration is comparable; otherwise prefer target-recall or dev-selected thresholds evaluated on held-out test.
- **Calibration:** fit temperature/thresholds on **dev only** (no test leakage). Phase 5 dev separation saturated on this split (collapsed thresholds); treat calibration as conditional and inspect stability.
- **Diagnostics per config:** score distributions by label, counts/lists of short-segment fill/padding cases and affected tracks (and any fill warnings), and false-positive/false-negative lists on the test split (and optionally overall) for qualitative review.
- **Selection discipline (strict):** define candidate configs → select best on dev using a predefined criterion (aligned with the primary goal) → **report held-out test once per chosen config**. Avoid repeated tuning/selection on held-out test results. Given the small, skewed, unknown-generator set, treat improvements as directional until validated on more representative data.

## Implementation prerequisites (to run this matrix)

Current bench behavior supports fixed per-run offset and zero-padding only. Likely needed before full execution:

1. **Per-track window placement**: support `start/middle/end` (duration-relative) strategies in `benches/run_bench.py` / decode path (compute track duration or derive offset per entry) and record the strategy in config/output.
2. **Configurable short-segment fill**: extend `benches/_shared/audio.py` (and callers/warnings/metadata) to support `zero-pad` vs `loop-fill` (simple loop/repeat, no crossfade) applied only when the selected segment is shorter than required, with explicit labeling.
3. **Experiment bookkeeping**: consistent output naming in a separate experiment location (not overwriting Phase 5 `benches/results/{calibration,summary,report}.*` or baseline raw files unintentionally), plus a comparison workflow that preserves traceability.
4. **Tests (TDD per contract)**: add/adjust tests first for any behavior change — e.g. per-track offset strategy resolution, loop-fill behavior/scope and warnings/metadata, and config/experiment recording.

## Open questions

- **Loop-fill implementation detail:** loop/repeat the available audio to fill the window with no crossfade (decided at intent level); exact low-level handling of edge cases may be determined during implementation as long as behavior is deterministic and recorded.
- **Multi-window aggregation method:** deferred; if pursued later, choose aggregation approach (e.g. mean/max/vote across start/middle/end) and evaluation criterion at that time.
- **D1 outcome:** provisional D1 is `ast-60s`; reaffirm or change before Phase 6 based on comparative tuning evidence aligned with the primary goal (owner decision, informed by results).

## Next steps

1. **Finalize execution plan:** start with placement sweeps — `AST-1a/b/c` and `STA-1a/b/c` (priority), plus `STB-1a/b/c` and `STG-1a/b/c` — then run `*-2a` loop-fill follow-ups on the best placement per backend (selected on dev by the predefined goal-aligned criterion).
2. **Implement prerequisites (TDD):** per-track placement strategies and configurable loop-fill (scoped to short segments), with clear config/metadata and experiment output separation.
3. **Run experiments and compare:** collect raw/summary results, apply strict dev-select/test-once reporting, and compare against Phase 5 baselines.
4. **Decide before Phase 6:** document findings and reaffirm or update D1 (and chosen operating point/calibration approach) based on the evidence.
