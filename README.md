# AI Music Detector

A standalone Python library that scores a local audio file, in-memory audio bytes or an http(s)
URL and returns a calibrated probability that the track was AI-generated. Each result carries
per-model detail (model id and revision, window, warnings, timings) and a conservative tri-state
label: `human`, `uncertain` or `ai_generated`.

The intended consumer is a music catalogue or feed that wants to catch likely AI tracks without
wrongly labelling legitimate human tracks as AI. This repository contains the library, its
benchmark evidence and its documentation — not a web service, database or feed ingestion.


## Repository layout

| Path | What it is |
| --- | --- |
| `ai-music-detector/` | The installable library, imported as `aidetect`. See [the library README](ai-music-detector/README.md) for the full API, output fields, CLI reference and example JSON. |
| `benches/` | Model experiments, benchmark harness and results. Start with [the bench README](benches/README.md). |
| `scripts/` | Data preparation helper (`download_ai_songs.py`) and its tests. |
| `test-audio/` | Owner-supplied labeled audio and `test-songs.csv`. Git-ignored. |
| `ai/` | Documentation only: [coding contract](ai/contracts/coding-contract.md), [implementation plan](ai/plans/plan.md), [agent primer](ai/prompts/primer.md), [implementer prompt](ai/prompts/implementer.md) and a historical discussion log. No runtime code. |
| `pyproject.toml`, `uv.lock` | One workspace-level source of truth for dependencies and tool configuration, with a single root `.venv`. |

## Quick start

Requires Python 3.10, `uv` and `ffmpeg` on `PATH`. From the repo root:

```bash
uv sync                                            # install the dev environment (all backends)
uv run aidetect score track.mp3 --json
uv run aidetect score track.mp3 --cascade --json   # two-stage policy; needs the full install
uv run aidetect score --help
```

Scoring flags come after the `score` subcommand. JSON goes to stdout; diagnostics and errors go
to stderr; errors exit non-zero. Without `--json`, stdout is one short `label<TAB>probability=...`
line.

The same call from Python:

```python
from aidetect import Detector

detector = Detector.from_pretrained()  # FP-1b: 10 s middle window, CPU/ONNX
result = detector.score_file("track.mp3")
print(result.label, result.probability)
```

See [the library README](ai-music-detector/README.md) for `score_bytes`, `score_url`,
`CascadeDetector`, output field meanings and example payloads.

## Install profiles

| Profile | Installs | Use |
| --- | --- | --- |
| `cpu-only` | numpy + onnxruntime (default fakeprint) | CPU-only host or container; no torch, CUDA, SONICS or transformers. |
| `full` | every backend | Required for `--cascade`, whose second stage uses SpecTTTra. |
| `onnx`, `gpu`, `ast` | individual backend stacks | Granular subsets; the profiles above delegate to these. |

```bash
uv pip install -e '.[cpu-only]'
uv pip install -e '.[full]'
```

`uv sync` for contributors installs `full` through the dev dependency group. Model weights are
fetched on first use into `AIDETECT_CACHE_DIR` (default: the Hugging Face cache), never into the
repo. Plain `import aidetect` does not import any model library.

## How the detection policy works

Models were benchmarked in `benches/` before the library was built; shared code was promoted into
`aidetect` rather than reimplemented. The D1 decision recorded in [the plan](ai/plans/plan.md#5-decision-gates)
selects fakeprint `FP-1b` as the primary stage for all tracks and SpecTTTra `STA-1b` as a second
opinion for tracks the primary stage did not auto-label `ai_generated`. Only the primary stage can
produce an `ai_generated` label: a second-stage flag routes the track to `uncertain` for review.

The label is always derived from the calibrated probability and packaged thresholds, never stored
as independent ground truth. Results are directional: the labelled set is small, genre-skewed and
of unknown generator identity, so cross-generator generalization cannot be claimed. Revalidate
thresholds on representative data before enforcing any label automatically.

## Data and artifacts

`test-audio/` (audio and manifest), `benches/results/raw/`, tuning `raw/` directories, model
weights and tool caches are git-ignored. Only the summaries and reports named in the plan are
committed. Never commit audio, weights, credentials or raw per-track results.

## Checks

All applicable commands must pass before a change is considered done (CC §2). They use each
command's own exit status, with no output piped through another command:

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run mypy benches ai-music-detector/src
uv run pytest --cov=aidetect --cov-fail-under=80
```

Default tests are offline, deterministic and free of GPU/network/model downloads; heavy calls sit
behind fakes and dependency injection.

## Scope and contributing

Out of scope for this repo (plan §1): worker queues, databases, RSS ingestion, feed heuristics,
web/UI filters and HTTP wrappers. No Docker image or HTTP service is included.

The [coding contract](ai/contracts/coding-contract.md) and [plan](ai/plans/plan.md) bind all work.
Read [the primer](ai/prompts/primer.md) first when joining; follow test-first red → green →
refactor, keep changes uncommitted until the phase is reviewed, use Conventional Commits, and do
not push without the owner's permission.
