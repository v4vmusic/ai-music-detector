# AI Music Detector — Test Tuning Plan (Pre–Phase 6)

## How agents use this plan

The owner assigns one phase to one fresh agent. Read **sections 1–3** and **only your assigned phase section**. Do not implement other phases; note anything they need in your report. The coding contract (`ai/contracts/coding-contract.md`) is binding, and `ai/prompts/implementer.md` is the working prompt unless the owner specifies otherwise. Tick checkboxes in your phase section when done, then tick the phase in the overview. Decision gates are recorded in section 5 by the owner.

This plan is a pre–Phase 6 tuning track. It implements the experiments and evidence defined in `ai/reference/test-tuning-discussion.md`. **Do not start Phase 6 of `ai/plans/plan.md` work here** except to record a D1 reaffirmation/change in that plan’s section 5 when the owner makes that decision.

## 1. Project context and decisions

**Goal:** run defensible benchmark tuning experiments to get the best measured results/evidence before moving on to Phase 6 of `ai/plans/plan.md`.

- **Primary optimization goal:** minimize **human false flags at an explicitly acceptable AI-recall level** (addressing `ast-60s` Phase 5 baseline **FPR@~95% recall = 0.45** on the held-out test split, 18/40 human test tracks).
- **Recall tradeoff:** accept **moderate AI-recall reduction (e.g. ~80–90%)** if it meaningfully reduces human false flags.
- **Constraint:** results are **directional/conditional** (small, genre-skewed, unknown-generator manifest `test-audio/test-songs.csv`, 400 tracks: 200 human / 200 AI, seed 42).

**Current state / baseline:** Phase 5 complete and committed (`0feae6a`). Gate **D1 is provisionally recorded as `ast-60s`** in `ai/plans/plan.md` §5. Phase 5 artifacts remain the committed baseline: `benches/results/calibration.json`, `benches/results/summary.json`, `benches/results/report.md` (raw git-ignored in `benches/results/raw/`). Tuning experiments must **not** overwrite these baseline artifacts.

**Recorded decisions (2026-11-19, see `ai/reference/test-tuning-discussion.md`):**
- Extend pre–Phase 6 work with benchmark tuning experiments before Phase 6.
- Keep `ast-60s` at its native **60s** window (no 90s/120s knob for this checkpoint).
- Evaluate **start/middle/end** window-placement strategies (all three), **duration-relative per track**: `start=0`, `middle=max(0,(D-W)/2)`, `end=max(0,D-W)`.
- First-pass priority: run `ast-60s` and `spectttra-alpha-120s` placement sweeps first, then fill follow-ups on best placements. Include **`spectttra-beta-120s` and `spectttra-gamma-120s`** in tuning scope as well.
- Skip `ast-60s` preprocessing/feature-parameter experiments; keep SpecTTTra preprocessing at baseline only.
- Non-zero fill: **simple loop/repeat** of available audio to fill the window (**no crossfade**), deterministic, applied **only when the selected window segment is shorter than required** (would otherwise be padded). Compare against `zero-pad` baseline.
- Defer multi-window aggregation until after single-window placement and fill sweeps.
- **Evaluation discipline (strict):** define candidate configs → select best on **dev** using a predefined criterion aligned with the primary goal (e.g. lowest human FPR at acceptable AI-recall, target range ~80–90%) → report **held-out test once per chosen config**. No tuning/selection on held-out test results.
- Keep Phase 5 committed artifacts unchanged; write experiment outputs separately with clear config naming.
- Be open to reaffirming or changing provisional D1 based on comparative tuning evidence before Phase 6 (owner decision).

**Backends in scope:** `ast-60s`, `spectttra-alpha-120s`, `spectttra-beta-120s`, `spectttra-gamma-120s`.

## 2. Model, data, and implementation facts

- **Manifest:** `test-audio/test-songs.csv` (400 local tracks, 200 `human`, 200 `ai_generated`; paths resolved relative to manifest). Extra columns ignored; bench uses local paths only.
- **Split/eval:** seed **42**, `fixed_dev_test` at current class sizes (per `benches/_shared/eval.py` policy), calibration fit on dev only. Phase 5 notes: dev separation saturated on this split (collapsed thresholds) — treat calibration as conditional and inspect stability.
- **Current decode behavior:** `benches/_shared/audio.py` decodes full file via ffmpeg (16 kHz mono float32), slices `[offset, offset+window)`, zero-pads short segments with `short_track_padded`, rejects tracks < 5 s, and peak-normalizes out-of-range windows with `peak_normalized_to_fit_range`.
- **Track durations (ffprobe, 400 tracks):** <60s: 5 (4 human, 1 AI); <120s: 26 (17 human, 9 AI); median ~212.6s. Loop-fill affects only cases where the selected segment is shorter than required.
- **Runner today:** `benches/run_bench.py` takes a single fixed `--offset` for all entries and writes per-track JSONL with `path,label,seed,config_hash,versions,timings,warnings` and score fields on success. Default offsets: `ast-60s` 25s (60s window), SpecTTTra 25s (120s window), `fakeprint` 0s.
- **Experiment matrix (source of truth):** `ai/reference/test-tuning-discussion.md` § “Experiment matrix” (placement A, fill B, model sets C1–C5, priority/order).

## 3. Environment and binding rules

- **Environment:** Python 3.10.12, `uv`, ffmpeg, PyTorch 2.6.0+cu124, CUDA 12.4, RTX 3070 Ti (~7.8 GB free VRAM typical). Weights cached in `AIDETECT_CACHE_DIR` (default Hugging Face cache). Never commit weights, audio, or raw experiment output.
- **Contract rules (binding):** TDD red → green → refactor for every behavior change/bug fix; DRY by rule of three; offline deterministic default tests with fakes (GPU/model/network tests marked and run explicitly); ffmpeg via argument lists with `shell=False`; no deletion of files not created by you; conventional commits (one commit per phase unless owner directs otherwise); no push without permission.
- **Benchmark integrity:** criteria/selection rule defined before comparing configs; strict dev-select then report held-out test once per chosen config; report negative/mixed results plainly; seeded runs; raw per-track JSONL records model revision, versions, config hash, timings, and experiment config (placement/fill/strategy) for traceability.
- **Artifact separation:** do not overwrite Phase 5 committed `benches/results/calibration.json`, `summary.json`, `report.md`. Store tuning outputs separately with clear config naming (e.g. under `benches/results/tuning/...` or equivalent git-ignored/explicitly managed locations per repo ignore rules); keep experiment raw JSONL traceable.
- **Owner interaction:** markdown checkbox questionnaires only (no question-asking tool); blank answer is not consent; “don’t make changes yet” means discuss only.

## 4. Phase overview

- [x] **Phase T0 — Tuning harness prerequisites** — per-track placement strategies, loop-fill mode, config/metadata & experiment output support, tests (TDD)
- [x] **Phase T1 — Placement sweep experiments** — run AST/STA/STB/STG start/middle/end sweeps (zero-pad), evaluate and select best placement per backend (dev, goal-aligned)
- [ ] **Phase T2 — Loop-fill follow-up experiments** — run `*-2a` loop-fill on best placement per backend, evaluate
- [ ] **Phase T3 — Comparison, report, and D1 review** — consolidate results vs Phase 5 baseline, produce tuning report, owner gate for D1 reaffirm/change

## 5. Decision gates

- **T-D1 (after Phase T3):** owner reaffirms or changes provisional **D1** (`ast-60s`) before Phase 6 based on comparative tuning evidence aligned with the primary goal (minimize human false flags at acceptable AI-recall, ~80–90% tradeoff range). Record the decision as a dated line in `ai/plans/plan.md` §5 (and summarize in the tuning report). If evidence is inconclusive, record that D1 remains provisional with rationale.

No agent decides T-D1 alone.

---

## 6. Phases

### Phase T0 — Tuning harness prerequisites

**Status:** [x] done  **Depends on:** Phase 5 complete (baseline committed); `ai/reference/test-tuning-discussion.md` decisions recorded

**Goal:** extend bench decode/runner capabilities so placement and fill experiments can be run reproducibly and traced, with TDD coverage and no change to Phase 5 committed artifacts.

**In scope:**
- **Per-track window placement strategies:** support `start`, `middle`, and `end` (duration-relative per track, `W` = backend window seconds, `D` = track duration: `start=0`, `middle=max(0,(D-W)/2)`, `end=max(0,D-W)`) in the bench scoring/decode path. Expose selection via runner CLI/API as appropriate (exact interface left to implementer) while preserving existing fixed-offset behavior for compatibility where needed. Ensure the selected segment avoids unnecessary padding when `D >= W`.
- **Configurable short-segment fill:** extend `benches/_shared/audio.py` (and callers/metadata/warnings as needed) to support `zero-pad` (baseline) and `loop-fill` (simple loop/repeat of available audio to fill `W`, **no crossfade**, deterministic) applied **only when the selected window segment is shorter than required**. Record fill mode and affected cases clearly (warnings/metadata/row fields) so outputs distinguish fill behavior.
- **Config/metadata & traceability:** include experiment configuration in outputs/config hash as appropriate (at minimum placement strategy and fill mode, plus existing seed/model/revision/versions/window/offset details) so per-config runs are unambiguous and comparable.
- **Experiment output separation:** support/standardize writing tuning outputs to a separate experiment location with clear config naming, without overwriting Phase 5 `benches/results/calibration.json|summary.json|report.md` or baseline raw files unintentionally. Document the convention.
- **Documentation updates:** update `benches/README.md` (and any relevant CLI help) for new placement/fill options and experiment output conventions.
- **Tests (TDD):** behavior tests for placement offset resolution (start/middle/end, including short tracks and boundary cases), loop-fill behavior/scope (fills only when segment shorter than required, deterministic, no unintended change when full segment available), and config/metadata/warning recording. Use synthetic WAV fixtures/fakes; no model weights, GPU, or network required for default tests.

**Out of scope:** running full 400-track GPU experiments (T1/T2), multi-window aggregation (deferred), `ast-60s` preprocessing/feature-param experiments, SpecTTTra preprocessing changes, loop-fill variants beyond simple loop/repeat no crossfade, changes to Phase 5 committed result artifacts, and Phase 6 library work.

**Test-first checklist:**
- [x] Placement resolution: `start/middle/end` compute correct per-track offsets for representative durations (`D < W`, `D == W`, `D > W`) and do not introduce unnecessary padding when a full window fits.
- [x] Loop-fill: when the selected segment is shorter than `W`, output length equals `W`, content is deterministically loop/repeated (no crossfade), and fill is explicitly recorded; when the segment already fills `W`, samples/mode are unchanged relative to baseline behavior (except intentional metadata additions).
- [x] Zero-pad baseline behavior remains correct and existing short-track/padding semantics are preserved or explicitly and compatibly extended.
- [x] Runner/outputs record placement strategy and fill mode (and relevant config) so runs are traceable and config hashes distinguish configs.
- [x] Existing bench tests continue to pass (update tests only where behavior is intentionally extended and specified).

**Implementation notes:** prefer small, explicit helpers (apply rule of three). Keep backend scoring I/O-free; decode/window policy remains in shared bench code. Do not change model revisions/preprocessing baselines except as required to support placement/fill. Avoid micromanaging internal structure, but ensure deterministic, testable behavior.

**Artifacts:** code/tests under `benches/` (e.g. `benches/_shared/audio.py`, `benches/run_bench.py`, and related shared modules/tests), updated `benches/README.md`, and any experiment output convention docs as needed.

**Human testing:** on a small synthetic/local manifest, run at least one backend/config smoke with each placement strategy and with `loop-fill` on a short/forced-short case; confirm per-track offsets/fill warnings/metadata and that outputs are written to the separate experiment location (not overwriting Phase 5 committed artifacts). Run CC §2 checks.

**Done criteria:**
- [x] checklist tests written red then green
- [x] CC §2 checks pass (`uv run pytest -q`, `uv run ruff check . && uv run ruff format --check .`, `uv run mypy benches`)
- [x] placement and loop-fill behaviors are documented and traceable in outputs
- [x] no Phase 5 committed artifacts modified

**For later phases:**
- For T1/T2: use `run_bench.py --experiment <NAME>` with `--placement {start,middle,end}` and optionally `--fill loop-fill`; `--offset` is fixed-placement only and rejects combinations with placement. Config names like `AST-1b`/`STA-2a` are validated as `[A-Za-z0-9][A-Za-z0-9._-]*`.
- For T1/T2: raw JSONL lands at `benches/results/tuning/<NAME>/raw/<backend>.jsonl` (git-ignored); `--compare --experiment <NAME>` writes `summary.json`/`calibration.json`/`report.md` beside it, with `--from-raw <dir>` supported. Evaluating rows containing `experiment` into `benches/results/` is rejected.
- For T1/T2: row field `experiment` holds `config_name`, `placement`, `fill_mode` and `offset_seconds` only for fixed placement; scored rows also hold resolved `offset_seconds`. Short-segment warnings are `short_track_padded` (zero-pad) or `loop_filled` (loop-fill); both can coexist with `peak_normalized_to_fit_range`. Count fill-affected tracks from these warnings.
- For T1/T2: config hashes now include the experiment config, so Phase 5 hashes (computed before this field) will not match a baseline rerun; compare tuning configs only against each other, and record the new hashes in comparison artifacts.
- For T1/T2: reuse `resolve_offset_seconds()` in `benches/_shared/placement.py` for placement math and `ExperimentConfig`/path helpers in `benches/_shared/experiment.py`; `decode_window()` owns window/fill policy and backends remain I/O-free. No model downloads or preprocessing changes were part of T0.
- For T2: loop-fill with a fixed offset beyond a track's end raises `AudioDecodeError` (no audio to repeat); duration-relative placements never hit that case. Multi-window aggregation remains deferred to a later phase.

---

### Phase T1 — Placement sweep experiments

**Status:** [x] done  **Depends on:** Phase T0

**Goal:** execute and evaluate single-window placement sweeps (`start/middle/end`, `zero-pad`) for all in-scope backends, then select the best placement per backend on dev using the predefined goal-aligned criterion.

**In scope:**
- **Configs to run (placement sweeps, zero-pad):** `AST-1a/b/c`, `STA-1a/b/c`, `STB-1a/b/c`, `STG-1a/b/c` as defined in `ai/reference/test-tuning-discussion.md` (native windows: AST 60s, SpecTTTra 120s; baseline preprocessing; fill `zero-pad`).
- **Priority:** run `ast-60s` and `spectttra-alpha-120s` sweeps first, then `spectttra-beta-120s` and `spectttra-gamma-120s` sweeps (all in scope).
- **Execution:** full-manifest runs on `test-audio/test-songs.csv` (400 tracks) with seed 42 and consistent settings per config; save raw per-track JSONL per (backend, config) in the separate experiment location with clear naming. Load/run one GPU model variant at a time as appropriate; record timings/versions/revisions/config hashes.
- **Evaluation/selection (strict):** for each backend, evaluate configs under the same split protocol (seed 42, fixed dev/test as per Phase 5 policy) with calibration fit on dev only. Using a **predefined criterion aligned with the primary goal** (minimize human false flags at acceptable AI-recall, target tradeoff range ~80–90%), select **one best placement per backend on dev**. Report held-out test metrics **once per selected config** (and record the criterion used). Optionally compute/report reference metrics (ROC-AUC/PR-AUC, FPR@~95% recall, ECE, etc.) for comparability, but do not use held-out test results to choose among placements.
- **Diagnostics:** capture per-config score distributions by label and short-segment/padding-affected counts; identify false positives/false negatives relevant to the chosen operating point(s) where useful for comparison.
- **Comparison artifact:** produce a tuning comparison summary for placement sweeps (markdown and/or JSON in the experiment location, clearly named) documenting configs run, selection criterion, dev-selected placements, test results, and caveats. Do not overwrite Phase 5 committed artifacts.

**Out of scope:** loop-fill follow-ups (`*-2a`, T2), multi-window aggregation, code changes unrelated to running/reporting these experiments (if tooling gaps are found, note them; minimal fixes must follow TDD), and D1 decision.

**Test-first checklist:**
- [x] If any analysis/evaluation helper code is added or changed, its behavior is covered by tests first (e.g. goal-aligned operating-point/FPR-at-target-recall selection) using synthetic data; pure experiment execution and report writing require no new behavior tests.
- [x] For each completed config, raw JSONL has 400 rows with expected fields/config identifiers and no silent failures (failed decodes/rows, if any, are recorded and reported).

**Implementation notes:** prefer reusing `benches/_shared/eval.py` and existing runner/eval flows; extend only if needed for goal-aligned selection/reporting and do so with TDD. Treat all results as directional. If a backend/config run fails or is unstable, record it plainly and continue with remaining configs.

**Artifacts:** per-config raw JSONL and evaluation outputs under the separate experiment location, and a placement-sweep comparison summary (e.g. `benches/results/tuning/placement-sweep-summary.{md,json}` or equivalent documented path).

**Human testing:** verify GPU/device availability and cached weights as needed; spot-check a few per-track outputs/config identifiers for one config per backend; review the selection criterion and that held-out test was reported once per selected config.

**Done criteria:**
- [x] all in-scope placement configs run or explicitly reported as failed with reasons
- [x] best placement per backend selected on dev using a predefined goal-aligned criterion; held-out test reported once per selected config
- [x] placement-sweep comparison artifact written with clear config naming and caveats
- [x] CC §2 checks pass if any code/tests changed; Phase 5 committed artifacts unchanged

**For later phases:**
- Criterion (fixed before runs, dev-only): lowest human FPR at AI recall 0.85 on raw dev scores; tie-break larger separation margin, then dev ROC-AUC, then config name. Helper: `benches/_shared/selection.py`; CLI: `benches/placement_sweep.py`.
- Dev-selected placements: `ast-60s`=`end` (AST-1c), `spectttra-alpha-120s`=`middle` (STA-1b), `spectttra-beta-120s`=`middle` (STB-1b), `spectttra-gamma-120s`=`start` (STG-1a). T2 should use these with `*-2a` names.
- Artifacts: `benches/results/tuning/placement-sweep-summary.{json,md}`; raw under `benches/results/tuning/<CFG>/raw/`. All 12 configs scored 400/400, 0 failures.
- Held-out test for selected configs (ROC-AUC / FPR@95 recall): AST-1c 0.8756 / 0.55; STA-1b 0.8806 / 0.525; STB-1b 0.8356 / 0.50; STG-1a 0.7356 / 0.725. At each dev-derived 85%-recall point, held-out human FPs: 8/40, 9/40, 20/40, 22/40.
- Placement helped dev FPR@0.85 vs Phase 5 fixed-25s for all backends (AST 0.425→0.213, STA 0.275→0.256, STB 0.494→0.450, STG 0.575→0.544). At the matched dev-derived threshold, held-out human FPs baseline→selected are AST 13→8, STA 8→9, STB 22→20, STG 25→22; AST-1c test FPR@95 recall (0.55) is still worse than baseline (0.45), so the gain is at the lower-recall point. T3 owns the formal baseline comparison.
- Dev raw-score separation is not saturated for FPR@0.85 even though the evaluator's post-calibration dev metrics saturate; no tie-break was needed. Report raw-score dev values, not `dev_metrics`, when comparing placements.

---

### Phase T2 — Loop-fill follow-up experiments

**Status:** [ ] not started  **Depends on:** Phase T1

**Goal:** evaluate the selected non-zero fill strategy (`loop-fill`, simple repeat, no crossfade, applied only when the selected segment is shorter than required) on the dev-selected best placement for each backend.

**In scope:**
- **Configs to run:** `AST-2a`, `STA-2a`, `STB-2a`, `STG-2a` as defined in `ai/reference/test-tuning-discussion.md`, each using the **best placement selected in Phase T1** for that backend (by the predefined goal-aligned dev criterion) and `loop-fill`.
- **Execution:** full-manifest runs (400 tracks, seed 42) with consistent settings; save per-config raw JSONL and metadata in the separate experiment location with clear naming. Record loop-fill usage and affected cases (notably <60s tracks for AST: 5; <120s for SpecTTTra: 26, label-skewed 17 human / 9 AI).
- **Evaluation (strict):** apply the same split protocol and calibration discipline (dev-only fit). Compare `*-2a` against the corresponding selected zero-pad placement config for each backend using the same goal-aligned criterion/operating-point framework; report held-out test once per chosen/reported config as appropriate without reusing test results for selection beyond what was predefined. Include diagnostics on fill-affected tracks.
- **Comparison artifact:** produce/update a loop-fill follow-up comparison summary (markdown and/or JSON) documenting selected placements, configs run, fill-affected counts, metrics/tradeoffs, and caveats. Do not overwrite Phase 5 committed artifacts.

**Out of scope:** multi-window aggregation (deferred), additional fill methods, preprocessing experiments, broader placement re-sweeps (unless a blocking issue requires rerun — document and justify), and D1 decision.

**Test-first checklist:**
- [ ] If any analysis/evaluation helper code is added or changed, its behavior is covered by tests first using synthetic data; pure experiment execution and report writing require no new behavior tests.
- [ ] For each completed config, raw JSONL records loop-fill configuration and affected-case signaling consistently with T0 behavior.

**Implementation notes:** reuse Phase T1 evaluation approach for comparability. Loop-fill impact may be limited by the number of short tracks; report observed impact plainly (including “no meaningful change” if applicable).

**Artifacts:** per-config raw JSONL and evaluation outputs under the separate experiment location, and a loop-fill follow-up comparison summary (e.g. `benches/results/tuning/loop-fill-summary.{md,json}` or equivalent documented path).

**Human testing:** spot-check that loop-fill was applied only to short segments as intended (via warnings/metadata and a few known short tracks) and that outputs are separate from Phase 5 artifacts.

**Done criteria:**
- [ ] all in-scope loop-fill configs run or explicitly reported as failed with reasons
- [ ] loop-fill vs selected zero-pad placement compared per backend with goal-aligned metrics and diagnostics
- [ ] loop-fill comparison artifact written with clear config naming and caveats
- [ ] CC §2 checks pass if any code/tests changed; Phase 5 committed artifacts unchanged

**For later phases:** *(agent fills this in: configs run, affected-track counts, key metrics vs selected placements, artifact paths, and any run issues.)*

---

### Phase T3 — Comparison, report, and D1 review

**Status:** [ ] not started  **Depends on:** Phase T1, Phase T2

**Goal:** consolidate tuning evidence, compare it against Phase 5 baselines, and produce a clear recommendation package for the owner to reaffirm or change provisional D1 before Phase 6.

**In scope:**
- **Consolidated analysis:** gather results from Phase 5 baseline (`benches/results/calibration.json`, `summary.json`, `report.md`) and tuning experiments (T1/T2 artifacts). Compare backends/configs on the primary goal (minimize human false flags at acceptable AI-recall, ~80–90% tradeoff range) while reporting ROC-AUC/PR-AUC, FPR@~95% recall (for comparability), calibration/ECE, timings, and key diagnostics (including fill impact and notable false positives where informative).
- **Tuning report:** write a committed tuning report (e.g. `benches/results/tuning/report.md` or equivalent clearly named path) summarizing: scope/decisions (reference `ai/reference/test-tuning-discussion.md`), configs run, selection criterion and strict dev-select/test-once protocol, per-backend best results, comparison to Phase 5, limitations/caveats, and a recommendation on whether to **reaffirm or change provisional D1 (`ast-60s`)**. Include negative/mixed results plainly.
- **Machine-readable summary (optional but recommended):** write/update a consolidated tuning summary JSON alongside the report if useful for traceability.
- **D1 review package for owner:** present the evidence and an explicit question/checkbox questionnaire (markdown) for the owner to reaffirm D1 as `ast-60s` or choose a different backend/config as the default before Phase 6. Do **not** record the final gate decision without owner confirmation.
- **Plan/doc updates (if owner decides in-session):** if the owner records T-D1/D1 during this phase, update `ai/plans/plan.md` §5 (and any directly affected status notes) accordingly and note the evidence paths; otherwise leave a clear “pending owner decision” state in the tuning report.

**Out of scope:** starting Phase 6 implementation, multi-window aggregation experiments, additional model runs beyond T1/T2 unless explicitly requested by owner (would require plan adjustment), and changing Phase 5 committed baseline artifacts.

**Test-first checklist:**
- [ ] If any analysis/consolidation helper code is added or changed, its behavior is covered by tests first using synthetic data; writing the report/questionnaire requires no new behavior tests.

**Implementation notes:** be precise about which numbers come from held-out test (reported once per chosen config) vs dev/selection. Keep claims proportional to evidence and dataset limitations. If results are inconclusive, recommend retaining provisional D1 with rationale rather than forcing a change.

**Artifacts:** committed tuning report (and optional consolidated tuning summary JSON) in a clearly named location, plus the owner questionnaire; updated `ai/plans/test-tuning-plan.md` status/notes and, if decided, `ai/plans/plan.md` §5.

**Human testing:** owner reviews the tuning report and answers the D1 review questionnaire.

**Done criteria:**
- [ ] tuning results consolidated and compared against Phase 5 baseline with goal-aligned metrics and caveats
- [ ] tuning report (and optional summary) written with clear traceability to experiment artifacts
- [ ] owner presented with a D1 reaffirm/change questionnaire; T-D1 recorded in `ai/plans/plan.md` §5 if and only if owner decides
- [ ] CC §2 checks pass if any code/tests changed; Phase 5 committed artifacts unchanged

**For later phases:** *(agent fills this in: final compared results, recommendation, artifact paths, and the recorded T-D1/D1 outcome or pending state.)*
