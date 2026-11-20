# AI Music Detector — Discussion Log

**Status:** discussion phase only. No project code has been written.
**Last updated:** 2026-11-18

## Goal

A standalone library that any app can call: given an audio file or a URL for an audio file, return a probability from 0 to 1 that the track was AI-generated. Per-model detail (sub-scores, model version, warnings, timings) should travel alongside the probability so callers can choose how to use it.

## Environment (checked 2026-11-18)

- Python 3.10.12
- ffmpeg available
- `uv` 0.6.5
- NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8192 MiB (driver 595.99.02)
- Workspace `/home/hapa/k/code/ai-music-dector-root/ai-music-detector` currently contains only empty `ai-music-detector/` and `ai/` directories
- PyTorch 2.6.0+cu124, CUDA build 12.4, cuDNN 9.1; GPU visible with ~7.8 GB free VRAM
- Installed Python stack: `torch` (1.5 GB) + `nvidia` CUDA libs (3.7 GB) ≈ 5.2 GB; `transformers` 4.48.3, `accelerate`, `sklearn`, `safetensors`, `pandas`, `librosa`, `soundfile`, `scipy` present. Missing: `onnxruntime` (needed for fakeprint ONNX) and the `sonics` package (needed for SpecTTTra)
- Disk: 938 GB volume, 45 GB free (96% used) — fine for model weights, not for large training datasets

## Decisions made

1. **First working slice:** scoring core + CLI/API. Benchmark *all four* candidate models first, then choose which one works best as the default. These are one slice in practice: build a pluggable core, then a benchmark CLI that drives all four backends through it.
2. **Runtime:** Python library + CLI. A thin HTTP service can be added later as a language-agnostic adapter without changing the core.
3. **Integration:** standalone for now. No v4vmusic codebase, DB, RSS ingestion, or listener-filter work in this project yet. Keep interfaces such that a catalog worker can be added later.
4. **Labeled data:** not gathered yet; will live in `/home/hapa/k/code/ai-music-dector-root/ai-music-detector/test-audio/`. Supply of AI tracks may be limited (200 is difficult), so the set may be small. The label manifest will be CSV (format below).
5. **Generator identity is unknown** for the labeled tracks, so evaluation can only measure binary human-vs-AI performance. Honest cross-generator generalization testing is not possible with these labels; this limitation must be stated in any benchmark result.
6. **SONICS: skip for now.** No download or training in this phase. It may be revisited later for training or cross-generator generalization testing (see the SONICS section below).
7. **Sequence: bench first, library second.** Model testing happens in a separate `benches/` folder at the workspace root. Only after bench results are in do we create `ai-music-detector/` as a library, informed by what we learned. No detailed implementation plan has been agreed yet.
8. **Coding contract:** `ai/contracts/coding-contract.md` is binding for all agents. It fixes tooling, style, TDD and DRY rules, shared data/model contracts, security rules, and the definition of done. Changes require owner approval.

## Model research (verified online 2026-11-18)

| Model | Verified facts | Caveats |
| --- | --- | --- |
| `awsaf49/sonics-spectttra-{alpha,beta,gamma}-120s` | Real models from SONICS (ICLR 2025). Authors' table: 120s F1 = 0.97 / 0.92 / 0.88 for alpha/beta/gamma. 16 kHz, 128 mel bands, 20 Hz–8 kHz. Installed via `pip install git+https://github.com/awsaf49/sonics`, not a plain `transformers` pipeline. | The brief's claim that "alpha overfits more than beta/gamma" is not in the model card; alpha actually has the highest reported F1. Treat variant choice as an empirical question. |
| `Chaolei98/MusicDET` | Real: ICML 2026 paper (arXiv:2605.18072), MIT-licensed GitHub repo. Zero-shot / one-class normalizing-flow model trained only on real music; inference produces a *likelihood*, low likelihood = AI. | Need to inspect the repo for released checkpoints and exact preprocessing before wiring a backend. Likelihood must be mapped to a probability via calibration on our data. |
| `AI-Music-Detection/ai_music_detection_large_60s` | Real HF model: AST fine-tune of `MIT/ast-finetuned-audioset-10-10-0.4593`, 60-second window, 90.8M params (~363 MB), `transformers` compatible. | Model card reports no evaluation metrics (`model-index` results are empty). Do not assume accuracy figures from the brief. |
| `lofcz/ai-music-detector` | Real HF model + MIT GitHub repo. Logistic regression over a 3585-dimension "fakeprint" vector; 16 kHz, FFT 8192, 1–8 kHz band; ONNX model is ~14.8 kB, effectively CPU-only. Author-reported 99.88% accuracy on their held-out test set. | Numbers are self-reported on the author's distribution. README explicitly says it needs retraining when generator architectures change, so it is likely brittle to unknown/new generators. |

## Disk footprint (checked 2026-11-18)

| Model | Weights on disk |
| --- | --- |
| SpecTTTra alpha-120s | 75,315,274 bytes (~75 MB) |
| SpecTTTra beta-120s | 86,309,962 bytes (~86 MB) |
| SpecTTTra gamma-120s | 97,849,930 bytes (~98 MB) |
| AST `ai_music_detection_large_60s` | 363,148,344 bytes (~363 MB) |
| lofcz fakeprint (ONNX + safetensors + npz) | ~45 KB total |
| **All of the above** | **~623 MB** |

- MusicDET has **no published trained checkpoints** (GitHub releases list is empty); the repo provides training code only. Its `spec-nf` model is small but needs training on FakeMusicCaps/SONICS. The SSL backbone variants require `facebook/wav2vec2-xls-r-300m` (1.27 GB) or `m-a-p/MERT-v1-330M` (1.26 GB HF weights / 3.99 GB fairseq file). Benchmarking MusicDET may therefore require finding community weights or training it — unresolved. Local training is especially unlikely: only 45 GB of disk is free, while the repo describes ~150 GB for SONICS alone.
- PyTorch + CUDA are already installed: `torch` 1.5 GB + `nvidia` CUDA libraries 3.7 GB (~5.2 GB total). ffmpeg is present. Missing packages for the planned stack: `onnxruntime`, `sonics`.
- Plan: download weights on first use into a local cache; do not commit them to the repo.

## SONICS dataset (checked 2026-11-18)

**Decision: skip for now.** Revisit later if we need cross-generator testing or want to train MusicDET; the NFS path `/mnt/chronos/kmmedia/Archives/ai/models` is available for a future download.

- Dataset from the same authors as SpecTTTra: >97k songs / 4,751 hours, >49k synthetic tracks generated with Suno and Udio.
- HF repo `awsaf49/sonics` ships fake audio in 10 zip parts totaling 32,195,437,925 bytes (~32.2 GB / 30 GiB), plus CSVs: `fake_songs.csv` 132 MB, `train.csv` 152 MB, `test.csv` 57 MB, `valid.csv` 9.8 MB, `real_songs.csv` 82 MB.
- **Real songs are NOT included**: only metadata with YouTube IDs; the real audio must be sourced separately.
- Fake metadata includes `source` (Suno/Udio), `algorithm` (e.g. `chirp-v3.5`), `label` (`full fake`/`half fake`/`mostly fake`), `duration`, `skip_time`, and train/val/test splits. Generator identity is therefore known here, unlike our own catalog.
- License: code and models MIT; **dataset CC BY-NC 4.0 (non-commercial only)** — a concern if v4vmusic is commercial. Paper-reported reproduction setup: 150 GB disk, 48 GB GPU, 32 GB RAM (authors' hardware).
- Generators are older (Suno v3/v3.5-era, early Udio), so models trained on it may learn artifacts that newer generators no longer have. The HF dataset viewer currently fails on a CSV schema error, though raw files are downloadable.
- NFS target `/mnt/chronos/kmmedia/Archives/ai/models` is writable, 11 TB total, **234 GB free (98% used)** — enough for the zips and extraction, but not a large margin.
- Possible middle path: MusicDET's zero-shot variant is trained **only on real music**, so our own human-labeled tracks could potentially train it without SONICS (needs verification in the repo). No SONICS download would then be required for that comparison.

## Label manifest format (CSV)

- Required columns: `path`, `label`.
- `path` may be an absolute file path, a path relative to the CSV's directory, or an HTTP(S) URL. No separate "type" column is needed.
- `label` must be exactly `human` or `ai_generated`.
- Only these two columns are used. Extra columns, if any, are ignored. Generator identity and duration are not supplied in the manifest; duration is derived from the audio itself with ffprobe/ffmpeg when needed.
- UTF-8, standard CSV quoting; a header row is required.

Example:

```csv
path,label,source,notes
/music/human/track001.mp3,human,local,
https://example.com/feed/ep12.mp3,ai_generated,feed-x,
```

If the labeled set is small (fewer than ~50 per class), use stratified k-fold cross-validation instead of a single holdout, report variation across folds, and treat results as directional rather than definitive. Prefer matching the number of human tracks to the AI tracks rather than padding one class.

## Design notes carried forward (not yet decided in detail)

- **Calibration:** a sigmoid is not a calibrated probability. Plan is a stratified dev split for fitting temperature scaling per model, with a held-out test split for reporting. The tri-state label (`human` / `ai_generated` / `uncertain`) should be derived from thresholds on the calibrated probability, not stored as an independent truth.
- **Class balance:** means the ratio of human to AI tracks in the labeled set. A heavily skewed set (e.g. 90% AI) makes plain accuracy misleading; roughly 50/50 is easiest to interpret, and ROC-AUC / FPR-at-fixed-recall remain usable either way.
- **Generator labels missing:** the manifest contains only path and label. Domain-shift analysis is limited to what can be inferred from paths/URLs and the audio itself; cross-generator generalization cannot be measured.
- **Benchmark metrics:** ROC-AUC (primary, threshold-free), PR-AUC, FPR at ~95% recall, F1 at chosen threshold, ECE after calibration; plus peak VRAM, seconds/track, and audio-seconds/sec. Save raw per-track scores, HF commit hashes, package versions, config hash, and seed for reproducibility.
- **Fair comparison:** shared versioned audio adapter; each backend declares its own window length/sample rate. Run backends one at a time with model unload or process isolation so 8 GB VRAM is not exceeded.
- **Proposed public API sketch:** `Detector.from_pretrained(id)`, `score_file(path)`, `score_url(url)`, `score_bytes(data)` returning `p_ai`, `label`, `model_id`, `model_revision`, `windows`, `warnings`, `timings`.
- **Deferred to later phases:** URL SSRF protections (scheme/private-IP checks, redirect limits, byte caps, timeouts), in-memory ffmpeg window slicing, retry/backoff, worker queue and DB schema, RSS feed heuristics. Feed heuristics, when built, should only reorder the queue and never skip a track, otherwise wrong tags never get corrected.

## Open questions / next discussion steps

- Approximate final count once `test-audio/` is populated; AI-track supply may be limited. If fewer than ~50 per class, benchmark with stratified k-fold cross-validation and report uncertainty instead of a single holdout.
- Acceptance criteria: what AUC / false-positive rate is good enough to ship as a default?
- Confirm whether model weights may be downloaded on first use into a local cache (rather than committed to the repo).
- MusicDET handling: drop it, attempt zero-shot training on our human tracks, or defer it explicitly.
- Where downloaded weights are cached: default Hugging Face cache with an env override, or `/mnt/chronos/kmmedia/Archives/ai/models`.
- Go-ahead required before any implementation starts.
