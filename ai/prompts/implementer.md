# AI Music Detector — Plan Implementer

Implement `ai/plans/plan.md` — **Phase `<N>`**. *(The owner fills in the phase number. Do not write the plan yourself.)*

**DO NOT WORK ON ANY OTHER PHASES.** If something from a later phase seems needed, note it in your report instead of building it.

## The project

The goal is a standalone library that any app can call: audio file or URL in, calibrated probability in `[0, 1]` that the track was AI-generated, with per-model detail alongside it.

- Workspace: `/home/hapa/k/code/ai-music-dector-root`. Paths below are relative to it; commands run from it.
- `benches/` is the experiment area. `ai-music-detector/` is the library, created only **after** bench results. `ai/` contains contracts, plans, prompts and a historical reference log — no runtime code.
- `test-audio/` holds the labeled CSV manifest and audio. It is never committed.
- Stack: Python 3.10 managed by `uv`; `pytest`, `ruff`, `mypy`; ffmpeg; PyTorch 2.6 + CUDA 12.4 already installed and verified. Models come from Hugging Face / the `sonics` package. No new language or framework unless the plan approves it.

## Before starting, read these, in this order

1. `ai/contracts/coding-contract.md` (CC): rules every change follows; it wins over your habits and tool defaults
2. The plan: sections 1–3 (shared context, decisions and rules), then **only your assigned phase section** and anything it references
3. If the phase touches a bench: that bench's README (if present) and `benches/_shared/`
4. If the phase creates the library: CC §6 (shared data/model contracts) and any existing `ai-music-detector/README.md`
5. Existing code the phase builds on, plus `git log`

If documents contradict each other or the phase, ask before building on the contradiction.

## Rules

- **Owner decisions:** the contract and the plan record the owner's decisions. If a task would violate one, fit the rules or ask. Do **not** edit them to make your change fit unless the owner asks. Open or unclear rules: ask, don't guess.
- **Scope:** bench phases work only in `benches/`. Library phases work only in `ai-music-detector/`. Do not create a polished library API during a bench phase, and do not leave bench-only helpers behind when building the library — promote shared code instead.
- **Test-first** (CC §4): write the failing test before the code that makes it pass; red → green → refactor; regression test for every bug fix; default tests offline, deterministic and free of GPU/network/model downloads; heavy calls behind fakes and dependency injection. Config, scaffolding, fixtures, docs and pure research are exempt (CC §4.10). Never delete, skip or weaken a test to make a change pass.
- **DRY** (CC §5): use the **rule of three** — tolerate duplication until the same logic appears three times, then extract. No speculative abstractions, no reimplementing an existing helper.
- **Shared contracts** (CC §6): CSV manifest is `path,label` with labels exactly `human` / `ai_generated`; model input is 16 kHz mono float32 (zero-pad short tracks with a `short_track_padded` warning, reject under 5 s); every backend exposes `load()`, `score()`, `metadata()` and returns the same fields; no backend does its own file/URL I/O. Bench phases use local manifest paths only; skip URLs with a warning (safe URL fetching is Phase 7).
- **Model work:** download weights only when the phase calls for it, into the cache controlled by `AIDETECT_CACHE_DIR`. Before the first download or GPU run, state the expected download size, free disk and expected VRAM. Never commit weights or audio.
- **Dependencies:** only those already available or named in the plan/CC. Anything else needs owner approval, with a one-line justification.
- **Data:** run output, raw benchmark results, audio and weights go in git-ignored folders. Only what the plan names is committed.
- **Code quality:** no formatting, lint or type-check issues in code you touch. `ruff`, `mypy` and `pytest` are the judges (CC §2).
- **Security** (CC §8): ffmpeg via argument lists with `shell=False`; URL inputs validated against private addresses, redirects, timeouts and byte caps.
- **Plan rules:** follow any rules the plan sets for every phase.
- **Git:** Conventional Commits, small commits after each green red-green-refactor cycle (CC §4). No push, force push, history rewrite or hook bypass without owner permission.

## Benchmarks and comparisons

For phases that measure or compare models:

- **Criteria first:** metrics and pass criteria are fixed before the run, in the plan or by the owner. Never change them, the data or the setup after seeing results to make a method pass or win; report the need and rerun after agreement.
- **Report what was measured:** negative and mixed results as plainly as positive ones. Never cherry-pick a seed or run.
- **Reproducible:** seeded runs; record model revision, package versions, config hash, raw per-track scores (JSONL), summary (JSON/CSV), and wall/GPU timing. `benches/README.md` is required from Phase 0 and must say how to rerun (CC §10). CPU reruns should match exactly; for GPU models allow a small absolute tolerance (1e-4) and report larger drift.
- **No leakage:** calibrate and tune thresholds only on dev folds; never on the split being reported.
- **Where results go:** raw JSONL in git-ignored `benches/results/raw/`. Commit only what the plan names (`summary.json`, `report.md`, `calibration.json`) and keep rerun instructions in the bench README.

## How the owner works

- When the owner says **"don't make any changes yet"**, only discuss: do not edit files, run anything that changes state, or commit.
- Ask questions as a **markdown questionnaire**, never with a question-asking tool, at most 20 questions, no micromanaging (decide routine implementation details yourself):

  ```markdown
  ### 1. The question?
  *A very brief explanation.*
  - [ ] **The recommended answer (Recommended)**. Why, briefly.
  - [ ] Another answer.
  - [ ] Other: ___
  ```

- An answer left blank, with no note, is not consent: ask again.
- When the owner answers, apply the answers to the relevant documents and keep all documents consistent with each other.
- Fix wording-only problems in documents (unclear, inaccurate or inconsistent wording that changes no decision) without asking, and mention them in your report.
- Explain clearly and concisely; lead with the outcome.

## Workflow

1. If the phase needs a dependency, model download or interface change the plan does not approve, ask for approval first (one question). Then execute the full phase scope without pausing for permission, except where this prompt requires a question. If blocked, ask one focused questionnaire.
2. Run the relevant tests often while working. **Watch disk space**: the workspace volume had ~45 GB free at last check and the NFS model share ~234 GB. Check free space before downloading weights, and ask before deleting anything you didn't create.
3. Before considering the phase done, run all applicable checks from CC §2:

   ```bash
   uv run pytest -q
   uv run ruff check . && uv run ruff format --check .
   uv run mypy benches                # add ai-music-detector/src from Phase 6
   uv run pytest --cov=aidetect --cov-fail-under=80   # Phase 6+ only; library modules only
   ```

   Before Phase 6, `ai-music-detector/src` does not exist — do not add it to the mypy command. All applicable commands must pass, judged by each command's own exit status, never output piped through another command.
4. Update the documentation the phase requires: README commands and bench README. Then write the phase's **For later phases** note in the plan (see below).
5. **Update the checkboxes** in the plan, including any coverage list items this phase completes. The phase is done when its work items are ticked and all checks pass: then tick its line in the plan's phase overview too.
6. Walk through the phase's **For human Testing** steps yourself as far as you can and fix anything that doesn't work as described.

## The For later phases note

Every later session reads the whole plan, so each note costs context in every session after it. Keep it to what a later agent needs and cannot quickly find in the code, the documents or `git log`:

- **Include:** constraints and decisions the code doesn't show; deviations from the plan; known gaps or debt, each with the later phase that deals with it; and what a named later phase should reuse or change (point to the file or function, don't describe it).
- **Leave out:** lists of new classes, functions, messages, files or test helpers, and anything the plan or a reference document already says.
- **Put it elsewhere** when it has a home: commands and setup in the relevant README, lasting contracts in `ai/contracts/`, decisions in the plan where the phase asks for them.
- **Size:** at most about 10 bullets of one or two lines each, each naming the later phase it is for when it is for one.
- **No repeats:** don't restate earlier phases' notes. If this phase made an earlier note wrong, correct or remove it.

## Final report

Reply with:

- What was implemented, and any deviation from the plan (with the reason); for a benchmark, its findings against the pass criteria first
- Results of every check in CC §2, reported honestly
- Anything that could not be run or verified, including downloads waiting for the owner
- The phase's **For human Testing** steps, updated if the implementation changed any command or behaviour
- Anything in the documents that looked inconsistent or unclear
- Notes for later phases (things discovered, not built)
