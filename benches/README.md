# Bench harness

Model experiments live here. Phase 0 provides only the shared skeleton: manifest loading,
audio window decoding, result types and a decode utility. There are no models, weights,
downloads or GPU calls in this phase.

## Setup

From the workspace root (Python 3.10, `uv`, `ffmpeg` required):

```bash
uv sync
```

This creates the root `.venv` and installs the pinned dev tools from `uv.lock`.

## Manifest format

`test-audio/manifest.csv` (or any owner-named path) is `path,label` with a header:

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

Phase 0 has no scoring runner; Phase 1 adds the first one (`benches/run_bench.py`) using this
interface. Every runner must use a fixed seed, write raw per-track JSONL, and record the model
revision, package versions, config hash and wall/GPU timing.

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

The staging list now holds 200 `human` and 200 `ai_generated` rows. `test-audio/manifest.csv`
remains the manifest for bench runs and is still pending.

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
