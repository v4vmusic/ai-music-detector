# AI Music Detector — Implementation Plan

## How agents use this plan

The owner assigns one phase to one fresh agent. Read **sections 1–3** and **only your assigned phase section**. Do not implement other phases; note anything they need in your report. The coding contract (`ai/contracts/coding-contract.md`) is binding, and `ai/prompts/implementer.md` is the working prompt. Tick checkboxes in your phase section when done, then tick the phase in the overview. Decision gates are recorded in section 5 by the owner.

## 1. Project context and decisions

**Goal:** a standalone library any app can call. Audio file or URL in; calibrated probability in `[0, 1]` that the track was AI-generated, with per-model detail alongside it (`model_id`, `model_revision`, per-window scores, warnings, timings). A tri-state label (`human` / `uncertain` / `ai_generated`) is derived from the calibrated probability, never stored as an independent truth.

**Sequence:** benchmark models in `benches/` first; create `ai-music-detector/` only after results. Bench code is scratch but still tested. Never reimplement a bench helper in the library — promote it. The library is distributed as `ai-music-detector` and imported as `aidetect`. `ai/` holds docs only.

**SONICS dataset: skipped.** No download, no training in this plan. It may be revisited later for cross-generator testing; the writable NFS path `/mnt/chronos/kmmedia/Archives/ai/models` exists if that changes.

**Labels:** CSV at `test-audio/manifest.csv` (or another owner-named path) with exactly two columns `path,label`; labels exactly `human` or `ai_generated`; paths absolute, relative to the manifest directory, or `http(s)://`; extra columns ignored; duration derived from audio. Supply may be small (~20–40 per class): if fewer than ~50 per class, use stratified k-fold and report variation across folds; do not pad one class with duplicates; treat results as directional. Generator identity is unknown, so only binary performance can be measured — cross-generator generalization cannot be claimed.

**Out of scope:** worker queue, database, RSS ingestion, feed heuristics, web/UI filters, HTTP wrapper.

## 2. Model candidates

| Model | Facts affecting implementation |
| --- | --- |
| `awsaf49/sonics-spectttra-{alpha,beta,gamma}-120s` | Authors' 120s F1 0.97 / 0.92 / 0.88. 16 kHz, 128 mel bands, 20 Hz–8 kHz. Needs `pip install git+https://github.com/awsaf49/sonics`. Weights 75 / 86 / 98 MB; ~3.9 / 2.3 / 1.6 GB VRAM. Load one variant at a time. |
| `AI-Music-Detection/ai_music_detection_large_60s` | AST fine-tune, 60s window, 90.8M params, 363 MB, transformers compatible, no metrics in model card. |
| `lofcz/ai-music-detector` | Logistic regression over 3585 fakeprint values; 16 kHz, FFT 8192, 1–8 kHz, hull area 10 bins. ONNX ~14.8 kB, CPU. Author-reported accuracy is on their own distribution. |
| `Chaolei98/MusicDET` | ICML 2026, MIT repo, **no published trained weights**. Zero-shot variant trains only on real music; output is a likelihood. See Phase 4. |

Benchmark all *available* models; "all four" is qualified by these constraints.

## 3. Environment and binding rules

- Python 3.10.12, `uv` 0.6.5, ffmpeg, PyTorch 2.6.0+cu124, CUDA 12.4, cuDNN 9.1, RTX 3070 Ti with ~7.8 GB free VRAM. `torch` ~1.5 GB + `nvidia` libs ~3.7 GB installed. Missing at start: `onnxruntime` (Phase 1) and `sonics` (Phase 2).
- Local disk ~45 GB free (96% used); NFS model share ~234 GB free. Check before downloads; never delete files you did not create.
- Weights download on first use into `AIDETECT_CACHE_DIR` (default Hugging Face cache). Never commit weights, audio or raw benchmark output.
- Contract rules: TDD red → green → refactor, test first for every behaviour and bug fix; DRY by the **rule of three**; offline deterministic default tests with fakes; GPU/network tests marked and run explicitly; ffmpeg via argument lists with `shell=False`; URL inputs reject private addresses and cap redirects, timeouts and bytes; conventional commits; no push without permission.
- Benchmark integrity: criteria fixed before a run; report negative results plainly; no tuning on the reported split; seeded runs; raw per-track JSONL with model revision, versions, config hash and timings.
- Owner interaction: markdown checkbox questionnaires, never a question-asking tool; blank answer is not consent; "don't make changes yet" means discuss only.

## 4. Phase overview

- [ ] **Phase 0 — Bench foundations** — no models or downloads; manifest, audio adapter, shared types
- [ ] **Phase 1 — Fakeprint backend** — ONNX, CPU, cheapest model first
- [ ] **Phase 2 — SpecTTTra backends** — alpha/beta/gamma 120s
- [ ] **Phase 3 — AST backend** — Hugging Face AST, 60s
- [ ] **Phase 4 — MusicDET feasibility** — research spike, gate D2
- [ ] **Phase 5 — Benchmark harness and report** — k-fold, metrics, calibration, gate D1
- [ ] **Phase 6 — Library foundation** — create `ai-music-detector/`, promote shared code
- [ ] **Phase 7 — Library adapters and CLI** — URL/bytes inputs, CLI, docs

## 5. Decision gates

- **D2 (after Phase 4):** owner decides whether MusicDET is dropped, attempted via zero-shot training on our human tracks, or deferred. No agent decides alone.
- **D1 (after Phase 5):** owner chooses the default backend from measured results. Phase 6 does not start until D1 is recorded here.

Record a gate decision as one dated line with the choice; the benchmark report holds the evidence.

---

## 6. Phases

### Phase 0 — Bench foundations

**Status:** [ ] not started  **Depends on:** nothing

**Goal:** a runnable `benches/` skeleton with shared code for later backends: manifest loading, audio window decoding, shared result types. No models, no downloads, no GPU, no network.

**In scope:** root `pyproject.toml` with uv-managed dev deps (pytest, pytest-cov, ruff, mypy) and tool config; `uv.lock`; `.gitignore` for `.venv/`, `test-audio/`, `benches/results/`, caches. `benches/_shared/manifest.py` (load `path,label` CSV, validate labels, resolve relative paths against CSV directory). `benches/_shared/audio.py` (local file → ffmpeg → 16 kHz mono float32, offset and window length; short-track policy explicit). `benches/_shared/types.py` (`AudioWindow`, `ModelOutput`, `ModelMetadata` per contract §6). `benches/_shared/registry.py` (model keys only). A script to decode one file and print shape and duration.

**Out of scope:** model wrappers, URL inputs, metrics, the library.

**Test-first checklist:**
- [ ] Manifest rejects a bad label, accepts a valid row, resolves relative paths, ignores extra columns.
- [ ] Decode window math: sample count, offset, mono shape, float32 dtype, value range; generated WAV fixture only.
- [ ] Short audio follows one documented policy (zero-pad or reject) and the test asserts it.
- [ ] Non-audio input raises `AudioDecodeError`, not an empty array.
- [ ] Shared dataclasses reject out-of-range probabilities and missing required fields.

Generate fixtures in test code with the standard `wave` module; commit no binary audio.

**Implementation notes:** one tool config in the root `pyproject.toml`; use buffers rather than temp files where practical, clean up any temp file; apply the rule of three before adding helpers.

**Artifacts:** `pyproject.toml`, `uv.lock`, `.gitignore`, `benches/_shared/*.py`, `benches/tests/`, decode script.

**Human testing:** create a 10 s synthetic WAV in `/tmp`; decode a 3 s window at 2 s offset and confirm shape `(48000,)`, float32; run all CC §2 checks.

**Done criteria:** [ ] every checklist test was red then green; [ ] CC §2 checks pass; [ ] no model weights, audio or network calls added.

**For later phases:** *(agent fills this in: ≤10 short bullets naming the later phase each item is for.)*

---

### Phase 1 — Fakeprint backend

**Status:** [ ] not started  **Depends on:** Phase 0

**Goal:** a fakeprint backend in `benches/backends/` scoring a decoded window into `ModelOutput`, plus a single-backend bench runner writing JSONL. Install `onnxruntime` (approved here). Weights ~15 kB, CPU only.

**In scope:** `benches/backends/fakeprint.py` with `load()`, `score()`, `metadata()`; feature extraction exactly: 16 kHz, FFT 8192, 1–8 kHz band, hull area 10 bins, 3585 values; injectable ONNX session so tests load no weights; `benches/run_bench.py` with `--backend`, `--manifest`, `--out`, one JSONL row per track with model id, revision, config hash and timing; one real-weights smoke run after tests pass (report its ~15 kB size before running).

**Out of scope:** other models, calibration, metrics, URL inputs, the library.

**Test-first checklist:**
- [ ] Deterministic synthetic signal gives 3585 features, correct dtype, bounded values.
- [ ] Silence and a pure tone give different but valid feature vectors.
- [ ] Fake ONNX session with fixed logit yields expected probability, model id and metadata.
- [ ] Wrong sample rate or stereo raises the shared error, not a raw numpy/ONNX exception.
- [ ] Runner writes one JSONL row per manifest entry and records a warning for a failed decode without stopping.

**Implementation notes:** backend receives an `AudioWindow` and performs no file/URL I/O; keep band/FFT constants in one function; promote sigmoid conversion only when a third backend needs it (rule of three).

**Artifacts:** `benches/backends/fakeprint.py`, `benches/run_bench.py`, tests, git-ignored `benches/results/raw/fakeprint.jsonl`.

**Human testing:** run the bench runner on a manifest; confirm probabilities in `[0, 1]`, revisions present, and identical scores on a seeded rerun.

**Done criteria:** [ ] checklist tests red then green; [ ] CC §2 checks pass; [ ] weights not committed; smoke size reported first.

**For later phases:** *(agent fills this in.)*

---

### Phase 2 — SpecTTTra backends

**Status:** [ ] not started  **Depends on:** Phase 0, Phase 1 runner pattern

**Goal:** three SpecTTTra backends behind the shared interface, each scoring a 120 s window. Install `sonics` from GitHub (approved here). Weights 75 / 86 / 98 MB; authors report 3.9 / 2.3 / 1.6 GB VRAM. Load one variant at a time.

**In scope:** `benches/backends/spctttra.py` parameterised by variant key; registry keys for alpha/beta/gamma 120s; 120 s window at configurable offset (default 25 s) using Phase 0's short-track policy; injectable classifier so unit tests use no weights/GPU; one GPU smoke run per variant downloading one at a time, reporting size, free disk and free VRAM before each; JSONL in Phase 1's shape.

**Out of scope:** 5 s variants, training, fine-tuning, SONICS dataset, the library.

**Test-first checklist:**
- [ ] Registry keys map to valid model ids and window lengths; unknown key raises `ModelLoadError`.
- [ ] Offset and duration reach the decoder unchanged, including the 25 s default.
- [ ] Fake classifier with known logit gives expected probability, variant id and revision.
- [ ] Metadata records sample rate 16000, window length 120 and preprocessing version.
- [ ] Load/unload leaves no module-level model handle (fake loader tracks calls).

**Implementation notes:** keep `HFAudioClassifier.from_pretrained` in one place; reuse Phase 1's probability conversion when a third backend needs it (rule of three); GPU tests marked `@pytest.mark.gpu` and excluded from default runs.

**Artifacts:** `benches/backends/spctttra.py`, tests, git-ignored JSONL per variant.

**Human testing:** score one 120 s file per variant; confirm plausible probabilities and no OOM; confirm `nvidia-smi` memory returns to baseline between variants; seeded rerun matches.

**Done criteria:** [ ] checklist tests red then green; [ ] CC §2 checks pass; [ ] download size and VRAM reported; [ ] no weights committed; one variant loaded at a time.

**For later phases:** *(agent fills this in.)*

---

### Phase 3 — AST backend

**Status:** [ ] not started  **Depends on:** Phase 0, Phase 1 runner pattern

**Goal:** an AST backend using the Hugging Face model with a 60 s window, behind the same interface. Weights 363 MB; no published metrics, so treat results as evidence only.

**In scope:** `benches/backends/ast.py`; registry key; 60 s window at configurable offset; injectable pipeline so unit tests load no model; one GPU smoke run (report size, disk and VRAM first); JSONL in Phase 1's shape.

**Out of scope:** training, other AST checkpoints, the library.

**Test-first checklist:**
- [ ] Registry key maps to the right model id and 60 s window.
- [ ] Label-to-probability mapping is verified explicitly against the model's `id2label`; a swapped mapping fails the test.
- [ ] Fake pipeline with known logits yields expected probability and metadata.
- [ ] Wrong input shape or sample rate raises the shared error.
- [ ] Two consecutive `load()` calls do not duplicate the model in memory (fake loader tracks calls).

**Implementation notes:** backend receives an `AudioWindow`; only one backend module imports `transformers`; reuse shared probability conversion per rule of three.

**Artifacts:** `benches/backends/ast.py`, tests, git-ignored `benches/results/raw/ast.jsonl`.

**Human testing:** score a file with AST and fakeprint; compare outputs qualitatively; seeded rerun matches; `nvidia-smi` memory returns to baseline.

**Done criteria:** [ ] checklist tests red then green; [ ] CC §2 checks pass; [ ] download size reported; [ ] weights not committed.

**For later phases:** *(agent fills this in.)*

---

### Phase 4 — MusicDET feasibility

**Status:** [ ] not started  **Depends on:** Phase 0

**Goal:** decide whether MusicDET can be benchmarked without SONICS. Research spike only — no training, no dataset downloads.

**In scope:** read the MusicDET repository code and paper; establish checkpoint availability, exact preprocessing (16 kHz mono, 4 s crops, STFT `n_fft=512`, hop 160), whether the `--only_real` path can train on our human-labeled tracks, the minimum viable data size, and rough time/disk cost; write `benches/musicdet-feasibility.md` with a recommendation. Produce **gate D2** for the owner.

**Out of scope:** training, model downloads, SONICS, adding a backend, the library.

**Test-first checklist:** if any helper code is written, its behaviour is test-first; pure research needs no tests. Do not fabricate a benchmark result.

**Implementation notes:** cite sources read; separate verified facts from inference; report uncertainty plainly. Do not edit this plan — the owner records D2 in section 5.

**Artifacts:** `benches/musicdet-feasibility.md`.

**Human testing:** read the report and answer the D2 questionnaire (drop / attempt zero-shot / defer).

**Done criteria:** [ ] report contains verified facts, sources and a recommendation; [ ] no training or dataset download; [ ] owner asked for D2.

**For later phases:** *(agent fills this in.)*

---

### Phase 5 — Benchmark harness and report

**Status:** [ ] not started  **Depends on:** Phases 1–3; Phase 4 findings; labeled CSV in `test-audio/`

**Goal:** one reproducible comparison of all available backends on the owner's labeled set, with calibration and a recommendation. If the CSV is not present yet, build and test the harness with synthetic fixtures and leave the real run as a human step.

**In scope:** unified `benches/run_bench.py` (all backends, folds/seed flags); stratified k-fold when fewer than ~50 per class, otherwise a fixed dev/test split; metrics ROC-AUC (primary), PR-AUC, FPR at ~95% recall, F1 at chosen threshold, ECE after temperature scaling; timing and peak VRAM; save raw per-track JSONL plus `benches/results/summary.json`; write `benches/results/report.md` with limitations (small set, unknown generators, older model generations). Produce **gate D1** for the owner.

**Out of scope:** choosing the default model (owner decides), the library, threshold hardening beyond dev data.

**Test-first checklist:**
- [ ] Splitting is stratified, seeded and leak-free; tests use synthetic labels.
- [ ] Each metric matches a hand-computed expected value on a tiny synthetic score set.
- [ ] Temperature scaling improves or leaves unchanged log-loss on a synthetic dev set and never touches test scores.
- [ ] Summary output contains model revisions, versions, config hash, seed and timing.
- [ ] A failing backend run is recorded and excluded without corrupting the summary.

**Implementation notes:** criteria are fixed before the run; do not change them after seeing results; report negative results; never cherry-pick a seed; raw JSONL stays git-ignored, only the summary/report the owner names is committed.

**Artifacts:** `benches/run_bench.py`, `benches/_shared/eval.py`, tests, `benches/results/summary.json`, `benches/results/report.md`.

**Human testing:** run the full benchmark on `test-audio/manifest.csv`; check the report's limitations section; verify a seeded rerun produces the same summary.

**Done criteria:** [ ] checklist tests red then green; [ ] CC §2 checks pass; [ ] criteria documented before results; [ ] owner asked for D1.

**For later phases:** *(agent fills this in.)*

---

### Phase 6 — Library foundation

**Status:** [ ] not started  **Depends on:** gate D1

**Goal:** create `ai-music-detector/` with a clean, pip-installable library containing only what worked in the benches.

**In scope:** `ai-music-detector/src/aidetect/` with public API (`Detector.from_pretrained`, `score_file`, `score_url`, `score_bytes`), `Result` dataclass, calibration and threshold logic, one default backend chosen at D1; promote shared bench code rather than reimplementing it; add build metadata to the existing root `pyproject.toml` pointing at `ai-music-detector/src` (no second pyproject); lazy heavy imports; `ai-music-detector/README.md`; package tests with a fake backend. Coverage gate applies from this phase.

**Out of scope:** URL safety, CLI, HTTP wrapper, worker/DB/RSS work.

**Test-first checklist:**
- [ ] Public API with a fake backend returns probability in `[0, 1]`, label, model id/revision, windows, warnings, timings.
- [ ] Threshold boundaries produce `human` / `uncertain` / `ai_generated` as specified, including exact edges.
- [ ] Calibration transform is the same fitted object used at inference; no dev data leaks into scoring.
- [ ] `import aidetect` does not import torch or any model library (lazy imports).
- [ ] Unsupported backend or model id raises the shared error with an actionable message.

**Implementation notes:** `__all__` in every package `__init__.py`; fully annotated signatures; no mutable module globals; no `print()` in library code.

**Artifacts:** `ai-music-detector/src/aidetect/`, tests, README, root pyproject build config.

**Human testing:** `uv pip install -e .` in a fresh environment; import `aidetect` and score a local file with the default backend; confirm no torch import on plain import.

**Done criteria:** [ ] checklist tests red then green; [ ] CC §2 checks including 80% coverage pass; [ ] no bench-only helper reimplemented.

**For later phases:** *(agent fills this in.)*

---

### Phase 7 — Library adapters and CLI

**Status:** [ ] not started  **Depends on:** Phase 6

**Goal:** finish the callable surface: safe URL and bytes inputs, CLI, cache configuration and documentation.

**In scope:** `aidetect` URL adapter (http/https only, private/loopback/link-local rejection, redirect cap, connect/read timeouts, byte cap, audio validation); bytes adapter; CLI `aidetect score <path|url> --json` with JSON to stdout and diagnostics to stderr; `AIDETECT_CACHE_DIR` documented; README updated with commands and limitations.

**Out of scope:** HTTP service, worker/DB/RSS/UI work.

**Test-first checklist:**
- [ ] URL adapter rejects private IPs, loopback, non-http schemes, too many redirects, oversized bodies and timeouts using a local fake server/fakes — no external network.
- [ ] URL adapter validates that downloaded bytes are audio before decoding.
- [ ] Bytes adapter produces the same `Result` shape as a local file with identical fixture audio.
- [ ] CLI emits valid JSON with a probability in `[0, 1]`; errors go to stderr with a non-zero exit code.
- [ ] Cache directory override is honoured and documented.

**Implementation notes:** ffmpeg via argument lists with `shell=False`; external hosts are untrusted; no `print()` in library code; rule of three before new helpers.

**Artifacts:** adapters, CLI, tests, updated README.

**Human testing:** score a local file and one owner-supplied URL with `--json`; confirm diagnostics on stderr and a clean JSON object on stdout.

**Done criteria:** [ ] checklist tests red then green; [ ] CC §2 checks pass; [ ] no external network calls in default tests; [ ] README matches behaviour.

**For later phases:** *(agent fills this in; name the later plan for HTTP wrapper / worker work if relevant.)*
