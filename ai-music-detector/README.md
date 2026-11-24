# aidetect

`aidetect` is the library half of this workspace. It provides calibrated, tri-state
AI-generated music detection over local audio files, in-memory audio bytes or http(s) URLs. The
benchmark harness lives in [`benches/`](../benches/README.md); this package promotes its shared
helpers and backends rather than reimplementing them.

## Install and import

Two supported installation profiles:

```bash
uv pip install -e '.[cpu-only]'   # CPU only: fakeprint via ONNX
uv pip install -e '.[full]'       # every backend (needed for --cascade)
```

`cpu-only` installs just `numpy` and `onnxruntime`; the ~15 kB fakeprint weights are fetched
directly from Hugging Face with the standard library. It is the right choice for a CPU-only
host or container — no torch, CUDA wheels, SONICS or transformers are downloaded. `--cascade`
requires `full` (or both the `onnx` and `gpu` extras): stage 1 uses ONNX fakeprint and stage 2
uses SpecTTTra. On a cpu-only install it fails with a clear error instead of importing the GPU
stack.

The granular extras remain available if you want a subset:

```bash
uv pip install -e .            # library only (numpy)
uv pip install -e '.[onnx]'    # default fakeprint backend
uv pip install -e '.[gpu]'     # SpecTTTra backends
uv pip install -e '.[ast]'     # AST backend
```

Plain `import aidetect` does not import `torch`, `onnxruntime`, `transformers`, `sonics` or
`huggingface_hub`. Model libraries and weights are loaded only when a detector actually needs
them. The pinned FP-1b ONNX weights are bundled as package data and load offline without a cache
or network request. Other weights are cached under `AIDETECT_CACHE_DIR` when set, otherwise under
the default Hugging Face cache. Scoring needs `ffmpeg` on `PATH` (also inside the container if you
deploy there); it performs the real audio decode for files, bytes and URLs.

## Python API

### Detectors

```python
Detector.from_pretrained(
    model_id=None, *,
    config_name=None,
    backend=None,
    calibration=None,
    cache_dir=None,
    device="cpu",
) -> Detector
```

Builds a detector from packaged config and calibration, loading weights lazily. With no
arguments it uses the D1 default `FP-1b` (fakeprint, 10 s middle window, CPU/ONNX). Select
another packaged operating point with `config_name` (`STA-1b`, `AST-1c`, `STB-1b`, `STG-1a`)
or `model_id`; giving both checks that the id belongs to that config. SpecTTTra configs need
the `gpu` extra and `AST-1c` needs the `ast` extra. `cache_dir` overrides `AIDETECT_CACHE_DIR`,
and `device` defaults to `"cpu"`. `backend` and `calibration` are advanced injection points
used by offline tests.

| Method | Returns | Description |
| --- | --- | --- |
| `score_file(path)` | `Result` | Decode one configured window from a local file. |
| `score_bytes(data)` | `Result` | Decode the same window from in-memory audio bytes. |
| `score_url(url, *, fetcher=None)` | `Result` | Fetch remote audio safely, then score it (see [URL safety](#url-safety)). |

`Detector.flag_threshold` exposes the packaged raw-score cutoff used by the cascade for its
stage flags. A standalone `Detector` does not use that cutoff: its `Result.label` comes from the
calibrated probability and the packaged low/high thresholds.

```python
CascadeDetector.from_pretrained(
    *,
    primary=None,
    second_opinion=None,
    primary_config_name="FP-1b",
    second_opinion_config_name="STA-1b",
    cache_dir=None,
    device="cpu",
) -> CascadeDetector
```

Builds the D1 two-stage detector described below. It has the same `score_file`, `score_bytes`
and `score_url` methods, returning `CascadeResult`. `score_url` fetches the bytes once and
scores both stages from that same payload.

### Examples

**Score a file, bytes or a URL** — all three adapters return the same `Result` shape:

```python
from pathlib import Path
from aidetect import Detector

detector = Detector.from_pretrained()
results = [
    detector.score_file("track.mp3"),
    detector.score_bytes(Path("track.mp3").read_bytes()),
    detector.score_url("https://example.com/track.mp3"),
]
for result in results:
    print(result.label, f"{result.probability:.4f}", result.model_id)
```

**Use the conservative cascade** (stage 2 runs only when stage 1 did not flag):

```python
from aidetect import CascadeDetector

cascade = CascadeDetector.from_pretrained()
result = cascade.score_file("track.mp3")

print(result.label)  # human | uncertain | ai_generated
print(result.primary.label)
if result.second_opinion is not None:  # None when stage 1 auto-labelled ai_generated
    print(result.second_opinion.label)
```

**Select another packaged model**:

```python
from aidetect import Detector

detector = Detector.from_pretrained(config_name="STA-1b")
# equivalently: Detector.from_pretrained(model_id="awsaf49/sonics-spectttra-alpha-120s")
result = detector.score_file("track.mp3")
print(result.model_id, result.window)
```

**Handle failures**:

```python
from aidetect import (
    AidetectError,
    AudioDecodeError,
    Detector,
    ModelLoadError,
    NetworkAudioError,
)

try:
    result = Detector.from_pretrained().score_url("http://127.0.0.1/track.mp3")
except NetworkAudioError as exc:
    print(f"unsafe or unusable URL: {exc}")
except (AudioDecodeError, ModelLoadError) as exc:
    print(f"audio/model failure: {exc}")
except (AidetectError, ValueError, OSError) as exc:
    print(f"expected failure: {exc}")
```

**Serialize for an app or service** — this is the payload the CLI writes with `--json`:

```python
import json
from dataclasses import asdict
from aidetect import Detector

detector = Detector.from_pretrained()  # keep one instance; weights load once
payload = asdict(detector.score_bytes(audio_bytes))  # audio_bytes: bytes from upload/cache
print(json.dumps(payload, sort_keys=True))
```

**Reuse a detector for a batch** instead of constructing one per track:

```python
from aidetect import UNCERTAIN_LABEL, Detector

detector = Detector.from_pretrained()
review = [path for path in paths if detector.score_file(path).label == UNCERTAIN_LABEL]
```

### Output responses

All three adapters return a frozen `Result`; the cascade returns a frozen `CascadeResult`.
The CLI's `--json` output is `dataclasses.asdict()` of that same object, so JSON keys match the
field names below. The example values below were generated from a synthetic tone and injected
scorers: they show the exact shape, not expected scores for real music.

**`Result` fields**

| Field | Meaning |
| --- | --- |
| `label` | Tri-state outcome: `human`, `uncertain` or `ai_generated`. For a standalone detector it is derived from `probability` and the packaged low/high thresholds, never stored as independent truth. The cascade may override the final label; see `CascadeResult`. |
| `probability` | Calibrated score in `[0, 1]` after temperature scaling. Compare with the packaged low/high thresholds for this model. |
| `raw_score` | Backend score before calibration. Its scale is model-specific (fakeprint returns a sigmoid value; SpecTTTra returns a logit), so do not compare it across models or treat it as a probability. |
| `model_id` | Pinned model id that produced the score. |
| `model_revision` | Pinned model revision/commit. |
| `window` | Decoded window as `<seconds>s@<rate>Hz[@<offset>s]`, e.g. `10s@16000Hz` or `120s@16000Hz@0s`. |
| `placement` | Window placement used: `fixed`, `start`, `middle` or `end`. |
| `warnings` | Array of strings from decoding and calibration, e.g. `short_track_padded`, `loop_filled`, `peak_normalized_to_fit_range` or `calibration_fallback: ...`. The CLI also prints these to stderr. |
| `timings` | Object with `score_seconds`, `load_seconds` and `gpu_seconds`; fields that do not apply are `null`. |

**`CascadeResult` fields**

| Field | Meaning |
| --- | --- |
| `label` | Conservative final label and authoritative cascade outcome. Only stage 1 can produce `ai_generated`; a stage-2 flag or any disagreement between the two stages' calibrated labels yields `uncertain`. `human` requires both stages to agree on `human` and stage 2 not to flag. |
| `primary` | The stage-1 `Result` (always present). |
| `second_opinion` | The stage-2 `Result`, or `null` when stage 1 auto-labelled `ai_generated`. |
| `warnings` | Warnings from both stages, de-duplicated and merged in order. |

For a standalone `Result`, `human` means the calibrated probability fell below the low
threshold, not proof of human authorship; `ai_generated` means it exceeded the high threshold,
not ground truth. With inclusive edges, `low <= probability <= high` yields `uncertain`. For the
packaged FP-1b calibration the two thresholds coincide, so the single-model `uncertain` band is
effectively a single value — the cascade below is the recommended conservative workflow.

The cascade uses a different rule for its stage decisions: stage 1's auto-AI flag and stage 2's
flag compare each model's **raw score** with its packaged `flag_threshold` in `tuning.json`.
`primary.label` and `second_opinion.label` remain the single-model calibrated labels and are not
rewritten. If those two labels disagree, `CascadeResult.label` is `uncertain`; use
`CascadeResult.label`, not `primary.label`, when applying the cascade policy.

**Example: single-model `Result`** (`aidetect score track.mp3 --json`):

```json
{
  "label": "ai_generated",
  "model_id": "lofcz/ai-music-detector",
  "model_revision": "d2180598fed79e3f917e8050a00439982466e5c6",
  "placement": "middle",
  "probability": 0.7357,
  "raw_score": 0.9940,
  "timings": {
    "gpu_seconds": null,
    "load_seconds": null,
    "score_seconds": 0.0067
  },
  "warnings": [],
  "window": "10s@16000Hz"
}
```

**Example: cascade routed to `uncertain`** — stage 1 did not flag, stage 2 did:

```json
{
  "label": "uncertain",
  "primary": {
    "label": "human",
    "model_id": "lofcz/ai-music-detector",
    "model_revision": "d2180598fed79e3f917e8050a00439982466e5c6",
    "placement": "middle",
    "probability": 0.0594,
    "raw_score": 0.000001,
    "timings": {
      "gpu_seconds": null,
      "load_seconds": null,
      "score_seconds": 0.01
    },
    "warnings": [],
    "window": "10s@16000Hz"
  },
  "second_opinion": {
    "label": "ai_generated",
    "model_id": "awsaf49/sonics-spectttra-alpha-120s",
    "model_revision": "094b32a5545098a71c113f6ae9d5c55310564268",
    "placement": "middle",
    "probability": 0.6695,
    "raw_score": 1.2,
    "timings": {
      "gpu_seconds": null,
      "load_seconds": null,
      "score_seconds": 0.01
    },
    "warnings": ["short_track_padded"],
    "window": "120s@16000Hz@0s"
  },
  "warnings": ["short_track_padded"]
}
```

**Example: cascade auto-labelled `ai_generated`** — stage 2 is skipped:

```json
{
  "label": "ai_generated",
  "primary": {
    "label": "ai_generated",
    "model_id": "lofcz/ai-music-detector",
    "model_revision": "d2180598fed79e3f917e8050a00439982466e5c6",
    "placement": "middle",
    "probability": 0.7148,
    "raw_score": 0.99,
    "timings": {
      "gpu_seconds": null,
      "load_seconds": null,
      "score_seconds": 0.01
    },
    "warnings": [],
    "window": "10s@16000Hz"
  },
  "second_opinion": null,
  "warnings": []
}
```

Without `--json`, stdout is one line:

```
human	probability=0.0594
```

For a cascade this line shows the final cascade `label` with the **primary** stage's
`probability`; use `--json` to get both stages.

### Errors

Expected failures derive from `AidetectError`: `AudioDecodeError` (undecodable or invalid
audio), `ModelLoadError` (weights/config problems) and `NetworkAudioError` (unsafe,
unreachable or non-audio URL sources). Invalid arguments raise `ValueError`; missing local
files raise `OSError`.

## CLI

The console script takes one local path or http(s) URL. Diagnostics and errors go to stderr;
stdout stays machine-readable:

```bash
uv run aidetect score track.mp3 --json
uv run aidetect score https://example.com/track.mp3 --json
uv run aidetect score track.mp3 --cascade --json   # D1 two-stage conservative policy
```

`aidetect --help` and `aidetect score --help` list every flag with examples.

- `--json` writes exactly one JSON object (the serialized `Result` or `CascadeResult`) and a
  trailing newline to stdout. Without it, stdout is one short `label<TAB>probability=...` line.
  See [Output responses](#output-responses) for field meanings and example payloads.
- Decode warnings (for example `short_track_padded`) are written to stderr as diagnostics and
  are also present in the result's `warnings` array.
- Expected failures (`AidetectError`, invalid values, I/O errors) write `aidetect: <message>` to
  stderr and exit with status 1; stdout stays empty. Argument errors use argparse's status 2.
- `--cascade` keeps D1 semantics: only stage one can auto-label `ai_generated`; a stage-two flag
  or a disagreement between the two stages' calibrated labels routes the track to `uncertain`.
  Its JSON has nested `primary` and (unless stage one flagged) `second_opinion` results instead of
  a top-level `probability`. It requires the `full` install (or both the `onnx` and `gpu` extras).

## Conservative two-stage policy

The default single-model detector applies packaged temperature calibration and tri-state
thresholds: `human` below the low threshold, `ai_generated` above the high threshold, and
`uncertain` at or between them. Thresholds are inclusive edges for `uncertain`.

The recommended D1 workflow is the explicit cascade:

```python
from aidetect import CascadeDetector

cascade = CascadeDetector.from_pretrained()
result = cascade.score_file("track.mp3")
```

Stage 1 runs `fakeprint` FP-1b on a middle 10 s window. Only stage 1 can produce
`ai_generated`, using its packaged raw-score cutoff; `primary.label` remains the single-model
calibrated label. Stage 2 runs `spectttra-alpha-120s` STA-1b on a middle 120 s window for tracks
stage 1 did not flag: a stage-2 raw flag routes to `uncertain` for review, and `human` requires
both stages' calibrated labels to be `human`. Any disagreement between the two stages' labels
is `uncertain`, because it needs human review. There is no union rule that auto-labels a track
`ai_generated` because either model alone flagged it. `CascadeResult` records both stage results
(except that a stage-1 auto-AI result has no second-opinion stage).

`cascade.score_url(url)` fetches the bytes once and runs both stages on that same audio.

## Packaged configuration

- `FP-1b` — default primary: `fakeprint`, 10 s middle window, zero-pad, CPU/ONNX.
- `STA-1b` — second opinion: `spectttra-alpha-120s`, 120 s middle window, zero-pad.
- `AST-1c`, `STB-1b`, `STG-1a` — other dev-selected backends, selectable by model id.

Temperature and the F1-derived single-model low/high tri-state thresholds are packaged in
`aidetect/resources/calibration.json`. The goal-aligned raw-score `flag_threshold`s used by the
cascade are packaged separately in `aidetect/resources/tuning.json`; they are not re-fitted from
the single-model calibration. If calibration is missing or degenerate, the library uses fixed
`0.35 / 0.65` thresholds with a warning.

## URL safety

`score_url` and `score ... <url>` apply these defaults before any audio decoding:

- Only `http`/`https` schemes; missing hosts, bad ports and unresolvable or private, loopback,
  link-local, reserved, multicast, unspecified and IPv4-mapped special addresses are rejected
  with `NetworkAudioError`.
- Redirects (maximum 5) are validated hop by hop, including relative `Location` values.
  Proxies are disabled so the connection uses the validated address.
- Connect/read timeout: 15 s per request. Responses are capped at 80 MiB (`Content-Length` is
  checked early; streamed bytes are capped regardless).
- Downloaded bytes must match a known audio container/frame signature before ffmpeg runs;
  ffmpeg then performs the real decode/validation. Non-audio payloads raise `NetworkAudioError`.
- `Detector.score_url(..., fetcher=...)` accepts an injected fetcher for offline tests; a custom
  fetcher must enforce the same contract.

Bench manifests still use local paths only and skip URL rows with a warning (CC §6); URL fetching
is a library adapter, not bench behaviour.

## Cache configuration

Non-bundled model weights are cached on first use under `AIDETECT_CACHE_DIR` when set, otherwise
under the default Hugging Face cache:

```bash
export AIDETECT_CACHE_DIR=/path/to/writable/cache
uv run aidetect score track.mp3 --json
```

The fakeprint loader checks that cache first, then the pinned FP-1b asset bundled with the
package, and only then downloads from Hugging Face. That makes CPU-only deployments self-contained
without requiring a writable model cache for FP-1b. The override applies to every model loader. An
explicit `cache_dir=` passed to `from_pretrained()` takes precedence over the environment
variable. URL responses are not written to this cache.

## Limitations

Thresholds were derived on a small, genre-skewed 400-track set with unknown generators and are
conditional on the documented windows, placements, fill mode and preprocessing. Fakeprint
calibration is weak, and threshold margins are sensitive. Treat this as a shadow/review tool,
revalidate on representative data before enforcement, and do not interpret `human` as proof of
human authorship.

URL fetches validate resolved addresses before connecting but do not pin the connection to that
address, so a hostile DNS server could still rebind between validation and connection. The
signature check is a cheap pre-filter, not a full file-type verifier; ffmpeg decides whether the
payload is decodable. No retry/backoff is performed for transient URL failures.
