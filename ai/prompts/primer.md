# AI Music Detector — Agent Primer

You are joining work on **AI Music Detector**, a standalone Python library that answers one question: given an audio file or URL, what is the calibrated probability that the track was AI-generated? Callers also get per-model detail and a conservative tri-state label (`human` / `uncertain` / `ai_generated`). The project is benchmark-first: model experiments live in `benches/`, and `ai-music-detector/` was created only after those results informed the design. Before doing anything else, familiarize yourself with the project as described below. Do not change any files while priming.

The project is one git repo, `/home/hapa/k/code/ai-music-dector-root/ai-music-detector`. Paths below are relative to it; commands run from it. There is no second repo and no proof-of-concept to protect.

## 1. Read these documents, in this order

1. `ai/contracts/coding-contract.md` (CC): binding rules every change follows. It wins over your habits and tool defaults.
2. `ai/plans/plan.md`: sections 1–3 (context, decisions, binding rules), then the phase overview (§4) and decision gates (§5), then **only the phase you are assigned**. Do not build other phases; note what they need in your report.
3. `ai/prompts/implementer.md`: the working prompt for phase work, including owner interaction and the final report.
4. If the work touches tuning: `ai/plans/test-tuning-plan.md`. Tuning is a finished pre–Phase 6 track whose results feed the D1 decision.
5. If the work touches a bench: `benches/README.md` and `benches/_shared/`. If it touches the library: `ai-music-detector/README.md` (API, install profiles, output responses).
6. The root `README.md` for orientation, then the code, `git log`, and the plan checkboxes. The documents describe the target; the code, checkboxes and history show how far along it is.

`ai/reference/` (`discussion.md`, `test-tuning-discussion.md`) is a **historical log, not a source of truth** (CC §9): read it for background, never build from it. If it disagrees with the contract or plan, the contract and plan win. There is no glossary, vision document or architecture document — the files above are the whole rulebook.

## 2. Things that are easy to get wrong

These are all in the documents above; they are repeated here because they matter most.

- **Authority:** the contract and the plan record the owner's decisions, including gates D1/D2 in plan §5. Fit your change to them or ask; do not edit them to make your change fit. The intended consumer, v4vmusic, is a separate repo and out of scope: plan §1 excludes worker queues, databases, RSS ingestion, feed heuristics, web/UI filters and HTTP wrappers. Do not add a Docker or HTTP-service scaffold unless the owner asks.
- **Terminology and output shape:** the distribution is `ai-music-detector`, the import name is `aidetect`. A single-model call returns a `Result`; the two-stage call returns a `CascadeResult`. `probability` is calibrated in `[0, 1]`; `raw_score` is model-specific (fakeprint returns a sigmoid value, SpecTTTra a logit) and must not be compared across models or treated as a probability. The label is always derived from `probability` and packaged thresholds, with inclusive edges falling to `uncertain`. In the cascade only stage 1 (fakeprint FP-1b) can auto-label `ai_generated`; stage 2 (SpecTTTra STA-1b) routes flagged tracks to `uncertain` and is skipped when stage 1 auto-labels. Install profiles are `cpu-only` (numpy + onnxruntime, no torch/CUDA/transformers/sonics) and `full` (every backend, required for `--cascade`). The CLI writes JSON to stdout, diagnostics and errors to stderr, and exits non-zero on errors; scoring flags follow the `score` subcommand.
- **Shared contracts (CC §6):** manifest is `path,label` with labels exactly `human` / `ai_generated`; model input is 16 kHz mono `float32` in `[-1.0, 1.0]`; short tracks are zero-padded with a `short_track_padded` warning (or `loop_filled` if explicitly configured) and tracks under 5 s are rejected; backends expose `load()`, `score()` and `metadata()` and never do their own file/URL I/O.
- **Test-first (CC §4):** write the failing test before the behaviour, red → green → refactor, with a regression test for every bug fix. Default tests are offline, deterministic and free of GPU/network/model downloads; heavy calls sit behind fakes and dependency injection. Config, scaffolding, fixtures, docs and pure research are exempt (CC §4.10), but every behaviour change still needs a red test. Never delete, skip or weaken a test to make a change pass.
- **DRY (CC §5):** use the rule of three — tolerate duplication until the same logic appears three times, then extract. Promote shared bench code into the library; never reimplement a bench helper in the library or vice versa.
- **Benchmark integrity (CC §10):** fix metrics and pass criteria before a run, never retune thresholds on the split being reported, and report negative and mixed results as plainly as positive ones. Seed runs and record model revision, package versions, config hash, raw per-track JSONL and timings. Results are directional: the labeled set is small, genre-skewed and of unknown generator identity, so cross-generator generalization cannot be claimed.
- **Model work:** weights download on first use into the cache controlled by `AIDETECT_CACHE_DIR` and are never committed. Check disk space before downloads (the local volume had ~45 GB free and the NFS model share ~234 GB — verify) and state expected download size, free disk and VRAM before the first GPU run. Never delete files you did not create.
- **Security (CC §8):** invoke ffmpeg via argument lists with `shell=False`; allow only `http`/`https` URLs, rejecting private, loopback and link-local addresses and capping redirects, timeouts and bytes; validate that decoded output is actually audio before scoring.
- **Dependencies:** only those already available or named in the contract/plan; anything else needs owner approval with a one-line justification. Heavy optional dependencies go behind extras. There is one root `pyproject.toml` and a committed `uv.lock`; all commands run via `uv run`.
- **Git:** branch `master`, Conventional Commits, **one commit per phase only after owner review** (CC §9). Keep work uncommitted while the phase is in progress, then show the full diff and proposed message and wait for explicit approval. No push, force push, history rewrite or hook bypass without permission; keep unrelated pre-existing changes out of a commit.
- **Verification (CC §2):** run every applicable command and report results by each command's own exit status, never output piped through another command:
  ```bash
  uv run pytest -q
  uv run ruff check . && uv run ruff format --check .
  uv run mypy benches ai-music-detector/src
  uv run pytest --cov=aidetect --cov-fail-under=80
  ```
  Report honestly anything you could not run. There is no `make check` in this repo.

## 3. How the owner likes to work

- When the owner says **"don't make any changes yet"**, only discuss: do not edit files, run anything that changes state, or commit.
- Discussion and building are separate phases; build one part at a time.
- When you have questions, write them as a **markdown questionnaire**, never with a question-asking tool, at most 20 questions, without micromanaging (decide routine implementation details yourself):

  ```markdown
  ### 1. The question?
  *A very brief explanation.*
  - [ ] **The recommended answer (Recommended)**. Why, briefly.
  - [ ] Another answer.
  - [ ] Other: ___
  ```

- An answer left blank, with no note, is not consent: ask again.
- When the owner answers, apply the answers to the relevant documents and keep all documents consistent with each other.
- Fix wording-only problems in documents (unclear, inaccurate or inconsistent wording that changes no decision) without asking, and mention them in your reply.
- Explain clearly and concisely; lead with the outcome.
- Plans record decisions and what works, never failed tests; bench reports and READMEs may record findings. Do not move decisions into `ai/reference/`.
- Stay inside the assigned scope. If a task requires changing the contract, the repository layout or a shared interface, stop and ask instead of improvising (CC §9).

## 4. When you are done priming

Reply with:

1. A short summary of your understanding of AI Music Detector: what it does, how it is built (bench-first, backends, D1 cascade), and its current state as derived from the plan checkboxes, code and `git log`.
2. Contradictions between documents that would change what gets built or how, each quoting both places. "None" is a good answer; leave out wording slips.
3. Confirmation that you will follow the coding contract.

Then wait for your task.
