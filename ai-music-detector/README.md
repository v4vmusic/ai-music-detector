# aidetect

`aidetect` is the library half of this workspace. It provides calibrated, tri-state
AI-generated music detection over local audio files or in-memory audio bytes. The benchmark
harness lives in [`benches/`](../benches/README.md); this package promotes its shared helpers
and backends rather than reimplementing them.

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

`score_bytes(data)` produces the same shape from in-memory audio. `score_url()` is deliberately
not part of Phase 6; it is planned for the adapter phase.

The result includes `raw_score`, calibrated `probability`, the tri-state `label`
(`human`, `uncertain`, `ai_generated`), pinned model id/revision, window description, placement,
warnings and timings.

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

## Packaged configuration

- `FP-1b` — default primary: `fakeprint`, 10 s middle window, zero-pad, CPU/ONNX.
- `STA-1b` — second opinion: `spectttra-alpha-120s`, 120 s middle window, zero-pad.
- `AST-1c`, `STB-1b`, `STG-1a` — other dev-selected backends, selectable by model id.

Temperature, low/high tri-state thresholds and raw flag cutoffs are packaged in
`aidetect/resources/calibration.json` and `aidetect/resources/tuning.json`. If calibration is
missing or degenerate, the library uses fixed `0.35 / 0.65` thresholds with a warning.

## Limitations

Thresholds were derived on a small, genre-skewed 400-track set with unknown generators and are
conditional on the documented windows, placements, fill mode and preprocessing. Fakeprint
calibration is weak, and threshold margins are sensitive. Treat this as a shadow/review tool,
revalidate on representative data before enforcement, and do not interpret `human` as proof of
human authorship.
