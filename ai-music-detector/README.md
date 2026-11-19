# aidetect

`aidetect` is the library half of this workspace. It provides calibrated, tri-state
AI-generated music detection over local audio files, in-memory audio bytes or http(s) URLs. The
benchmark harness lives in [`benches/`](../benches/README.md); this package promotes its shared
helpers and backends rather than reimplementing them.

## Install and import

```bash
uv pip install -e .            # library only (numpy)
uv pip install -e '.[onnx]'    # default fakeprint backend
uv pip install -e '.[gpu]'     # SpecTTTra backends
uv pip install -e '.[ast]'     # AST backend
```

Plain `import aidetect` does not import `torch`, `onnxruntime`, `transformers`, `sonics` or
`huggingface_hub`. Model libraries and weights are loaded only when a detector actually needs
them. Weights are cached under `AIDETECT_CACHE_DIR` when set, otherwise under the default
Hugging Face cache.

## Quick start

```python
from aidetect import Detector

detector = Detector.from_pretrained()  # D1 default: fakeprint FP-1b
result = detector.score_file("track.mp3")
print(result.label, result.probability, result.window, result.placement)
```

All three input adapters return the same `Result` shape:

```python
result = detector.score_bytes(audio_bytes)  # in-memory audio
result = detector.score_url("https://example.com/track.mp3")  # safe remote fetch
```

The result includes `raw_score`, calibrated `probability`, the tri-state `label`
(`human`, `uncertain`, `ai_generated`), pinned model id/revision, window description, placement,
warnings and timings.

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
- Decode warnings (for example `short_track_padded`) are written to stderr as diagnostics and
  are also present in the result's `warnings` array.
- Expected failures (`AidetectError`, invalid values, I/O errors) write `aidetect: <message>` to
  stderr and exit with status 1; stdout stays empty. Argument errors use argparse's status 2.
- `--cascade` keeps D1 semantics: only stage one can auto-label `ai_generated`; a stage-two flag
  routes the track to `uncertain`. Its JSON has nested `primary` and (unless stage one flagged)
  `second_opinion` results instead of a top-level `probability`.

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
`ai_generated`, using its packaged raw-score cutoff. Stage 2 runs `spectttra-alpha-120s`
STA-1b on a middle 120 s window for tracks stage 1 did not flag: a stage-2 flag routes to
`uncertain` for review, and no flag routes to `human`. There is no union rule that auto-labels
a track `ai_generated` because either model alone flagged it. `CascadeResult` records both
stage results (except that a stage-1 auto-AI result has no second-opinion stage).

`cascade.score_url(url)` fetches the bytes once and runs both stages on that same audio.

## Packaged configuration

- `FP-1b` — default primary: `fakeprint`, 10 s middle window, zero-pad, CPU/ONNX.
- `STA-1b` — second opinion: `spectttra-alpha-120s`, 120 s middle window, zero-pad.
- `AST-1c`, `STB-1b`, `STG-1a` — other dev-selected backends, selectable by model id.

Temperature, low/high tri-state thresholds and raw flag cutoffs are packaged in
`aidetect/resources/calibration.json` and `aidetect/resources/tuning.json`. If calibration is
missing or degenerate, the library uses fixed `0.35 / 0.65` thresholds with a warning.

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

Model weights are cached on first use under `AIDETECT_CACHE_DIR` when set, otherwise under the
default Hugging Face cache:

```bash
export AIDETECT_CACHE_DIR=/path/to/writable/cache
uv run aidetect score track.mp3 --json
```

The override applies to every model loader. An explicit `cache_dir=` passed to
`from_pretrained()` takes precedence over the environment variable. URL responses are not written
to this cache.

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
