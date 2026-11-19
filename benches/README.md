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
