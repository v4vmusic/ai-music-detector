# Bench harness

Model experiments live here. Shared helpers cover manifest loading, audio window decoding
and result types. The first scoring backend is fakeprint (CPU, ONNX, ~15 kB of weights);
there are no GPU calls yet.

## Setup

From the workspace root (Python 3.10, `uv`, `ffmpeg` required):

```bash
uv sync
```

This creates the root `.venv` and installs the pinned runtime dependencies (NumPy, onnxruntime)
and dev tools from `uv.lock`.

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
than the requested window is zero-padded with a `short_track_padded` warning.

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

The first runner is `benches/run_bench.py`:

```bash
uv run python benches/run_bench.py \
  --backend fakeprint \
  --manifest test-audio/test-songs.csv \
  --out benches/results/raw/fakeprint.jsonl \
  --seed 42
```

Use `--manifest test-audio/test-songs.csv` for bench runs. Its extra metadata columns are
ignored and URL rows are skipped with a warning. The backend scores a 10 s window from the
start of each track.

Each local manifest row produces one JSONL line. Scored rows contain the shared `ModelOutput`
fields plus `path`, `label`, `seed`, `config_hash` and package `versions`; `timings.score_seconds`
is wall time and `gpu_seconds` is `null` for this CPU backend. A failed decode produces a row
with `warnings`, `error`, `seed`, `config_hash` and `versions` but no score fields, and the run
continues.

Fakeprint weights are pinned by revision and SHA-256 in `benches/_shared/registry.py`. They are
downloaded once into `AIDETECT_CACHE_DIR` (default: the Hugging Face cache) and never committed.
Rerun with the same `--seed` and manifest to reproduce scores exactly; the `config_hash` records
the model metadata, seed and package versions.

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
