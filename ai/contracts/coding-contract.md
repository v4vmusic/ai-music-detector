# Coding Contract

**Applies to:** every agent and human contributor in `/home/hapa/k/code/ai-music-dector-root`.
**Status:** binding. Read this file and `ai/reference/discussion.md` before writing any code.
**Changes:** only with explicit approval from the project owner. Never silently reinterpret a rule.

This contract fixes conventions *now* so that work from different agents remains interchangeable. It is not an implementation plan.

---

## 1. Repository shape and sequencing

```
/home/hapa/k/code/ai-music-dector-root/
├── pyproject.toml              # single source of truth for deps + tool config
├── uv.lock                     # committed lockfile
├── benches/                    # model experiments and benchmark harness (first)
│   ├── _shared/                # reusable bench helpers
│   ├── backends/               # one module per candidate model
│   └── tests/                  # tests for bench logic
├── test-audio/                 # labeled audio + CSV manifest; never committed
├── ai-music-detector/          # the library, created only after bench results
│   └── src/aidetect/           # import name: aidetect
└── ai/                         # docs, decisions, contracts only — no runtime code
    ├── contracts/
    └── reference/
```

- **Bench first, library second.** Model testing happens in `benches/`. The library in `ai-music-detector/` is created only after bench results inform the design.
- `ai/` is documentation only. Do not put Python runtime code there.
- The library is distributed as `ai-music-detector` and imported as `aidetect`.
- One workspace-level `pyproject.toml` at the root defines dependencies and tool settings for both `benches/` and the library. Do not create competing config files.
- Use `.venv` at the workspace root, managed by `uv`. All commands run through `uv run`.

## 2. Enforced commands

A task is not done until all applicable commands pass:

```bash
uv run pytest -q                                          # unit + integration-by-default
uv run ruff check . && uv run ruff format --check .       # lint and format
uv run mypy benches ai-music-detector/src                 # static types
uv run pytest --cov=aidetect --cov-fail-under=80          # library only, once it exists
```

Settings live in `pyproject.toml` (line length 100, double quotes, trailing commas, sorted imports). Do not override them per-file without a comment explaining why.

## 3. Style and structure

- Python 3.10. Every module starts with `from __future__ import annotations`.
- `snake_case` for modules/functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants, `_leading_underscore` for private names.
- One primary concept per module. Split a module when it exceeds ~300 lines.
- Explicit public surface: `__all__` in every package `__init__.py`.
- All function signatures are fully annotated. No untyped `def`. No `Any` unless accompanied by a comment justifying it.
- Prefer dataclasses for structured data. No mutable module-level globals.
- No `print()` in library code. CLI code prints JSON to stdout and diagnostics to stderr.
- Use `logging` with `logger = logging.getLogger(__name__)`. Never configure logging in library modules.
- Timestamps are UTC ISO 8601 strings. Probabilities are plain `float` in `[0.0, 1.0]`; never `NaN`, never sentinel values. Missing data is `None`.

## 4. TDD rules (non-negotiable)

1. **Write the failing test first.** No production code without a test that fails for the intended reason.
2. **Red → green → refactor.** Write the smallest implementation that passes, then clean up while tests stay green.
3. **Bug fixes get a regression test first.**
4. **Tests describe behaviour, not implementation.** Test observable outputs and contracts, not private internals.
5. **Default tests are deterministic and offline.** No network, no GPU, no real model downloads. Use fixtures, fakes, and dependency injection around heavy calls.
6. **Slow, GPU, or network tests** are marked `@pytest.mark.integration` or `@pytest.mark.slow` and excluded from the default run. They must still be runnable explicitly.
7. **Coverage gate:** 80% line coverage on library modules once `aidetect` exists. Bench scripts are not coverage-gated, but their shared helpers in `benches/_shared/` are tested.
8. Do not use `pytest.skip` to hide a failing test. Do not weaken assertions to make a test pass.
9. Commit after each green red-green-refactor cycle, not a pile of unrelated changes at the end.

Agent loop for every task: read contract → write test → run it and confirm failure → implement → run full suite + lint + types → refactor → commit.

## 5. DRY rules

1. Single source of truth for constants: model IDs, sample rates, window lengths, thresholds, label strings. No magic numbers in logic.
2. **Rule of three:** tolerate duplication until the same logic appears three times, then extract a shared helper. Do not abstract earlier on speculation.
3. **Never reimplement a bench helper in the library (or vice versa).** When the library is created, promote shared bench code into `aidetect/` and have benches import it.
4. Model metadata lives in one registry; backends reference registry keys rather than duplicating IDs and preprocessing parameters.
5. Prefer composition and data-driven tables over chains of duplicated conditionals.
6. Generated files, test fixtures, and intentionally duplicated benchmark baselines are exempt — but mark them clearly.

## 6. Shared data and model contracts

All agents producing or consuming data must obey these shapes.

### Label manifest (CSV)

```csv
path,label
/music/human/track001.mp3,human
https://example.com/feed/ep12.mp3,ai_generated
```

- Header required. `label` is exactly `human` or `ai_generated`.
- `path` is absolute, relative to the manifest's directory, or `http(s)://`.
- Extra columns are ignored. Duration is derived from audio, not supplied.

### Audio handed to models

- `16_000` Hz, mono, `float32` NumPy array, values in `[-1.0, 1.0]`.
- Each backend declares its own required window length; the shared decoder produces windows, never the backend.
- Short audio is zero-padded or rejected by an explicit, tested policy — never silently truncated.

### Backend interface (defined in `benches/` first, promoted to the library)

- Each backend exposes `load()`, `score()` and `metadata()`.
- `score()` returns: raw score/logit (`float`), calibrated probability when available (`float | None`), model id, model revision, window description, warnings, timings.
- No backend imports another backend. No backend performs its own file/URL I/O.
- Loading one model must not leak GPU memory into the next; unload or isolate per process.

## 7. Errors, dependencies, and artifacts

- Exception hierarchy rooted at `AidetectError`, with specific subclasses such as `AudioDecodeError`, `ModelLoadError`, `InvalidManifestError`, `NetworkAudioError`.
- Fail fast with actionable messages. No bare `except`, no silent fallback that changes a score.
- Add a dependency only with a one-line justification in the commit message; prefer the standard library; heavy/optional dependencies go behind extras.
- `uv.lock` is committed. No unpinned transitive surprises.
- Never commit audio files, model weights, credentials, or raw benchmark output. Add these paths to `.gitignore`.
- Model weights download on first use into a cache controlled by `AIDETECT_CACHE_DIR` (default: the Hugging Face cache). Not into the repo.

## 8. Security

- `ffmpeg` is always invoked with an argument list and `shell=False`. User input is never interpolated into a shell string.
- URL inputs: allow only `http`/`https`, reject private/loopback/link-local addresses, cap redirects, set connect/read timeouts, cap downloaded bytes.
- External hosts are treated as untrusted: validate that decoded output is actually audio before scoring.

## 9. Git and agent conduct

- Small, focused commits with conventional prefixes: `feat:`, `fix:`, `test:`, `refactor:`, `docs:`, `chore:`, `bench:`.
- One logical change per commit, with its tests. Do not bundle unrelated fixes.
- Stay inside the assigned scope. If a task requires changing this contract, the repository layout, or shared interfaces, stop and ask instead of improvising.
- Never overwrite or delete another agent's work to "clean up". Resolve conflicts by asking.
- No force pushes, no history rewriting, no destructive git commands without explicit permission.
- Architecture decisions go in `ai/reference/discussion.md`, not in code comments or commit messages alone.

## 10. Bench-specific rules

- Bench scripts are standalone: `uv run python benches/<name>.py --manifest <csv> --out <dir>`.
- Save raw per-track scores as JSONL and summaries as JSON/CSV. Record model revision, package versions, config hash, seed, and wall/GPU timing.
- Deterministic seeds everywhere. Never tune thresholds or calibrate on the split being reported.
- Benchmark results are evidence, not production code. Do not promote a model wrapper into the library until its bench tests are green and its limitations are documented.

## 11. Definition of done

A task is complete when:

- [ ] A failing test was written before the implementation.
- [ ] `uv run pytest -q`, `ruff check`, `ruff format --check`, and `mypy` pass.
- [ ] Coverage gate passes for library code (once it exists).
- [ ] No new dependency, large file, secret, or generated artifact was committed.
- [ ] Existing shared behaviour was reused; the rule of three was applied before creating new abstractions.
- [ ] Relevant docs/contracts were updated, and decisions were recorded in `ai/reference/discussion.md`.
- [ ] The commit is small, conventionally named, and contains only this task.
