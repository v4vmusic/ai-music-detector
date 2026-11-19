# Bench harness

Model experiments live here. Shared helpers cover manifest loading, audio window decoding
and result types. Backends include fakeprint (CPU, ONNX, ~15 kB of weights) and GPU-based
SpecTTTra and AST; MusicDET has no backend (see the feasibility note below).

## Setup

From the workspace root (Python 3.10, `uv`, `ffmpeg` required):

```bash
uv sync
```

This creates the root `.venv` and installs the pinned runtime dependencies (NumPy, onnxruntime,
`torch`/`torchaudio` 2.6.0, `sonics`, `transformers` 4.48.3 and their dependencies) and dev tools
from `uv.lock`. `sonics` is pinned to commit `9156ffad151f797c71556923c4a02fa01fa8fc91` on GitHub.
A CUDA GPU is only needed for the SpecTTTra and AST backends; CPU fakeprint runs need no GPU.

## Manifest format

The owner-designated manifest is `test-audio/test-songs.csv`. It requires a `path,label` header:

```csv
path,label
/music/human/track001.mp3,human
track002.wav,ai_generated
```

`label` is exactly `human` or `ai_generated`. A relative `path` is resolved against the
directory containing the manifest. Extra columns are ignored. `http(s)://` rows are skipped
with a warning; URL fetching is not bench work. Audio under 5 s is rejected; a track shorter
than the requested window is zero-padded with a `short_track_padded` warning. If a decoded
window's peak magnitude exceeds 1.0, samples are conditionally peak-normalized by `1/peak`
(preserving waveform shape, no clipping) to fit `[-1.0, 1.0]` and a
`peak_normalized_to_fit_range` warning is emitted; windows already within range are unchanged.

## Decode one file

```bash
uv run python benches/decode.py --path /path/to/track.wav --offset 2 --window 3
```

`--offset` defaults to 0 seconds; `--window` is required. The command prints one JSON object
to stdout with `path`, `sample_rate`, `shape`, `duration_seconds` and `warnings`. Errors are
written to stderr with a non-zero exit code.

## Bench command convention

Bench scripts are standalone and invoked from the workspace root:

```bash
uv run python benches/<name>.py --manifest <csv> --out <dir>
```

### Score one backend

```bash
uv run python benches/run_bench.py \
  --backend fakeprint \
  --manifest test-audio/test-songs.csv \
  --out benches/results/raw/fakeprint.jsonl \
  --seed 42
```

Use `--manifest test-audio/test-songs.csv` for bench runs. Its extra metadata columns are
ignored and URL rows are skipped with a warning. Fakeprint scores a 10 s window from the start
of each track; SpecTTTra scores a 120 s window at 25 s by default; AST scores a 60 s window at
25 s by default.

Each local manifest row produces one JSONL line. Scored rows contain the shared `ModelOutput`
fields plus `path`, `label`, `seed`, `config_hash` and package `versions`; `timings.score_seconds`
is wall time and `gpu_seconds` is `null` for this CPU backend. A failed decode produces a row
with `warnings`, `error`, `seed`, `config_hash` and `versions` but no score fields, and the run
continues.

### Compare all available backends

To score all available backends, write raw per-backend JSONL into `benches/results/raw/`, and
produce `calibration.json`, `summary.json` and `report.md` in `benches/results/`:

```bash
uv run python benches/run_bench.py \
  --compare \
  --manifest test-audio/test-songs.csv \
  --results-dir benches/results \
  --seed 42 \
  --device cuda
```

Run GPU backends one variant at a time in separate processes when scoring individually; the
`--compare` runner loads and unloads backends sequentially and records a failing backend run in
the summary without corrupting other backends' results.

To evaluate from existing raw JSONL results without running models (each file must be named
`<backend-key>.jsonl`, e.g. `benches/results/raw/fakeprint.jsonl`):

```bash
uv run python benches/run_bench.py \
  --compare \
  --manifest test-audio/test-songs.csv \
  --results-dir benches/results \
  --from-raw benches/results/raw \
  --seed 42
```

`--folds <k>` sets k for stratified k-fold when fewer than ~50 samples per class are available
(default: 5). With at least ~50 per class, a fixed seeded dev/test split is used. Calibration
(temperature scaling and low/high thresholds) is fitted on dev data only; metrics are reported on
held-out/test data. If calibration is degenerate or dev data is unavailable, thresholds fall back
to `0.35 / 0.65` with a warning in `calibration.json`.

Fakeprint weights are pinned by revision and SHA-256 in `benches/_shared/registry.py`. They are
downloaded once into `AIDETECT_CACHE_DIR` (default: the Hugging Face cache) and never committed.
Rerun with the same `--seed` and manifest to reproduce scores exactly; the `config_hash` records
the model metadata, seed and package versions. CPU reruns should match exactly; for GPU models
allow a small absolute tolerance (1e-4) and record larger drift.

## SpecTTTra backends (GPU)

The three 120 s variants are separate backends. **Run one variant per process** so GPU memory is
released between models; each invocation exits after writing its JSONL. With `--device cuda`
(default) a GPU is required; `--device cpu` works but is much slower.

```bash
uv run python benches/run_bench.py \
  --backend spectttra-alpha-120s \
  --manifest /tmp/smoke.csv \
  --out benches/results/raw/spectttra-alpha-120s.jsonl \
  --seed 42 --offset 25 --device cuda
```

Use `spectttra-beta-120s` or `spectttra-gamma-120s` for the other variants.
`--offset` is optional: the per-backend default is 25 s, pinned in
`benches/_shared/registry.py` (`SPECTTRA_SPECS`). Window duration comes from the registry
(120 s), and short tracks follow the shared zero-pad / 5 s reject policy. Each row records the
actual offset in `window` (e.g. `120s@16000Hz@25s`) and the model revision.

Pinned weights (downloaded on first use into `AIDETECT_CACHE_DIR`, never committed):

| Backend | HF revision | Weights |
| --- | --- | --- |
| `spectttra-alpha-120s` | `094b32a5545098a71c113f6ae9d5c55310564268` | 75 MB |
| `spectttra-beta-120s` | `cc49422a98b5b92700876c608d0f9cdf27214a3e` | 86 MB |
| `spectttra-gamma-120s` | `57659d8cb04767893537bbeb89da973e2e5a0290` | 98 MB |

Authors report 3.9 / 2.3 / 1.6 GB VRAM for training these variants; inference on one 120 s
window uses substantially less. Check `nvidia-smi` between runs — memory should return to
baseline. Re-running with the same seed and manifest should match within an absolute tolerance
of 1e-4; record any larger GPU drift. The unit tests use an injected classifier and need no
weights, GPU or network.

## AST backend (GPU)

The Hugging Face AST fine-tune scores a 60 s window at 25 s by default:

```bash
uv run python benches/run_bench.py \
  --backend ast-60s \
  --manifest /tmp/smoke.csv \
  --out benches/results/raw/ast.jsonl \
  --seed 42 --offset 25 --device cuda
```

Pinned revision `5e29461bbfc41dc5fe4ed5f7fadc961aa84a7e3e`; `model.safetensors` is 363.1 MB and
downloads once into `AIDETECT_CACHE_DIR`. Inference on one 60 s window used roughly 1–2 GB VRAM;
`nvidia-smi` returned to the 63 MiB baseline after the process exited. A seeded two-track rerun
had 0.0 probability drift; record any larger GPU drift.

The checkpoint has no `preprocessor_config.json`. Its training notebook used the
`MIT/ast-finetuned-audioset-10-10-0.4593` AST feature extractor with `max_length=6000`, 128 mel
bins, 16 kHz and mean/std `-4.2677393` / `4.5689974`; `run_bench._load_ast_pipeline()` builds that
extractor from `AST_SPEC` so no base-model download is needed. The pinned config's `id2label` is
`{0: human, 1: ai_generated}`; the backend resolves the AI class by label, not by class index.
Unit tests inject a pipeline and need no weights, GPU or network.

## MusicDET feasibility

Phase 4 researched `Chaolei98/MusicDET`; see [musicdet-feasibility.md](musicdet-feasibility.md). No trained
checkpoint is published and there is no MusicDET bench command or backend. Gate D2 was recorded on
2026-11-19 as **defer**, so Phase 5 compares only the four available candidate model families.

## Staging the AI audio (download script)

`scripts/download_ai_songs.py` downloads the AI half of the staging set and appends verified
rows to `test-audio/test-songs.csv`. The bench runner never fetches URLs itself. Set
`V4VMUSIC_DATABASE_URL` (or pass `--database-url`), then preview or download:

```bash
uv run --with psycopg2-binary python scripts/download_ai_songs.py --dry-run
uv run --with psycopg2-binary python scripts/download_ai_songs.py
```

`--with psycopg2-binary` provides a throwaway dependency without changing `pyproject.toml`.
The script uses seeded selection (default seed 42), a 300-song candidate pool, serial requests
with a 1 s delay, capped redirects/bytes/timeouts, `Retry-After`-aware retries, and ffprobe
verification. Failed URLs are logged to `test-audio/ai-download-errors.csv`. Reruns skip songs
already in the CSV. `--dry-run` performs no downloads. All paths are git-ignored.

The manifest now holds 200 `human` and 200 `ai_generated` rows. It is the same
`test-audio/test-songs.csv` file with extra metadata columns, which the loader ignores.

## Result locations

- Raw per-track JSONL: `benches/results/raw/` (git-ignored).
- Committed, named outputs: `benches/results/calibration.json`, `summary.json`, `report.md`.
- Audio and model weights are never committed; weights download on first use into the cache
  controlled by `AIDETECT_CACHE_DIR` (default: the Hugging Face cache).

## Checks

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run mypy benches
```
