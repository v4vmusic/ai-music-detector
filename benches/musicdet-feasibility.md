# MusicDET feasibility (Phase 4)

**Date:** 2026-11-19. **Status:** research spike complete; gate D2 recorded 2026-11-19 — **defer** (see end).

**Question:** can `Chaolei98/MusicDET` be benchmarked without downloading SONICS or training a model on a generator dataset?

**Bottom line:** there is no trained checkpoint to download, so benchmarking requires local training. The
`spec-nf` zero-shot variant *can* be trained on our human-labeled tracks alone, needs no SONICS download
and no SSL backbone weights, and is cheap in compute and disk. However, our human set is roughly 240×
smaller than the authors' real-music set and the model degrades sharply on compressed audio like ours.
**Recommendation: defer MusicDET for now**; if cross-generator generalization later becomes a goal, run a
bounded, held-out zero-shot spike rather than treating a locally trained model as comparable to the
bench-validated backends.

Facts below are marked **[verified]** (read from the paper or repository code, or measured locally) or
**[inference]** (my judgement, not stated by the authors).

## Sources read

- Paper: "MusicDET: Zero-Shot AI-Generated Music Detection", ICML 2026, arXiv:2605.18072
  (<https://arxiv.org/abs/2605.18072>, full text read at <https://arxiv.org/html/2605.18072>).
- Repository: <https://github.com/Chaolei98/MusicDET>, MIT licence, default branch `main`, shallow-cloned
  to `/tmp/opencode/musicdet-src` for reading (clone HEAD `087a758955a638dca68ef0dcbb81107fce48d60f`; nothing from it was added to this
  repo). Files read: `README.md`, `config.py`, `train.py`, `dataset.py`, `model.py`, `model_factory.py`,
  `feature_extraction.py`, `preprocess_audio.py`, `test.py`, `evaluate.py`, `NF_model/glow_model.py`,
  `requirements.txt`.
- GitHub API: `GET /repos/Chaolei98/MusicDET/releases` → `[]`; `GET .../issues?state=all` → `[]`
  (checked 2026-11-19).
- Hugging Face API model search for `MusicDET` and `spec-nf` (checked 2026-11-19): no MusicDET/checkpoint
  matches.
- Local synthetic cost probe: `SpecNF` instantiated and timed with random tensors on our RTX 3070 Ti,
  torch 2.6.0+cu124, CUDA 12.4. Script in `/tmp/opencode/musicdet_cost_probe.py`, not committed.
  No weights, audio, labels or datasets were downloaded; no model was fitted and no checkpoint saved.

## Checkpoint availability

- **[verified]** No trained MusicDET checkpoint is published. GitHub Releases and Issues are both empty;
  the README gives training commands, a Google Drive link for *label CSVs* (not weights), and download
  commands for the wav2vec2-XLS-R-300m / MERT-v1-330M **SSL backbones** used by the
  `fr/ft-w2v2aasist` and `fr/ft-mertaasist` variants only.
- **[verified]** A locally trained model is a plain PyTorch `state_dict`
  (`anti-spoofing_feat_model.pt`) plus `args.json` in an output folder (`train.py:230`, `test.py:54-80`).
  `test.py` rebuilds the architecture from `args.json`; there is no HF-hub integration or single-file
  inference API. A bench backend would need its own load + preprocess + score wrapper (later phase work).
- **[inference]** Searching for community weights would be speculative and unauditable: even if a
  third-party `.pt` appeared, it would need its matching `args.json` and a provenance/trust review.

## Exact preprocessing (verified in code, matches the paper)

- Audio is resampled to **16 kHz mono**. The optional `preprocess_audio.py` writes 16 kHz mono float32
  WAV; its default path upsamples to 48 kHz then downsamples with `librosa`'s `soxr_hq`
  (`always_updown=True`). `dataset.py` can also load source audio directly at 16 kHz.
- `dataset.py` loads with `librosa.load(sr=16000, mono=True)` and crops or zero-pads to
  `TARGET_LEN = 64600` samples = **4.0375 s** at 16 kHz. Training uses a random crop offset; evaluation
  uses the centered crop. Short tracks are zero-padded; there is no 5 s rejection rule in this repo
  (our shared decoder's `AudioDecodeError` under 5 s is stricter and compatible).
- Each crop is **RMS-normalized** (`audio / sqrt(mean(audio²) + 1e-8)`), not left as plain `[-1, 1]`
  float audio.
- `SpecNF` computes a power spectrogram with `n_fft=512`, `hop_length=160`, `win_length=512`,
  `power=2`, `normalized=False`, takes `log(S + 1e-10)`, applies two stride-2 conv layers, then
  `x[:, :, 1:, 1:]` → `(B, 32, 64, 100)` into a 2-band Glow-style normalizing flow with `K=2` flow
  steps per band (CLI defaults `K=2, L=1, R=5.0`; paper §4.1 uses the same `K=2`, 2 bands, prior mean
  `μ_real=5`). SpecAugment masks time/frequency regions during training.
- Score at inference is **negative log-likelihood** (`-nll`); **lower means more likely AI-generated**.
  It is not a probability and needs our Phase 5 calibration before it can be a `Result.probability`.
- **[inference]** Our shared decoder (`benches/_shared/audio.py`) produces 16 kHz mono float32 via ffmpeg
  with no 48 kHz round-trip and no RMS normalization. A future backend must either reproduce MusicDET's
  resampling/RMS steps or deliberately document the deviation; do not assume the shared window is enough.

## Can `--only_real` train on our human tracks?

- **[verified]** Yes in principle, with no SONICS data and no SSL backbone weights. `--only_real` is a
  global CLI flag (`config.py:113`). `train.py:73-105` applies it to the **training** split only:
  - `task=fakemusiccaps` filters the training protocol to rows whose label is `real`;
    `task=sonics` filters the training CSV to `target == 0`.
  - The dev/validation split is always loaded with both classes, so an EER can be computed each epoch.
  - With all-zero training labels the flow uses the normal prior; `test.py` also scores every track with
    label `0`, so zero-shot train and inference are consistent.
- **[verified]** The datasets are hard-wired to the authors' file layouts. The simplest route is
  `task=fakemusiccaps` with generated protocol text files (`<filename> real|fake`, one per line, resolved
  against `--fakemusiccaps_audio`). The `task=sonics` CSV route needs `filepath,target` columns and its
  `normalize_sonics_filepath()` rewrites any path containing `real_songs_wav/` or `fake_songs_wav/`.
  Our `test-audio/test-songs.csv` uses `path,label` with `human`/`ai_generated`, so a small converter plus
  held-out fold assignment is required; no upstream code edit is needed.
- **[verified]** The dev protocol **must contain both classes**: `train.py:219-222` computes EER from
  `scores[labels==0]` and `scores[labels==1]`, so a real-only dev file would break validation.
- **[verified]** `train.py` deletes `--out_fold` if it exists unless `--continue_training` is given —
  always point it at a fresh, git-ignored directory.
- **[inference]** Any spike must train on human tracks from the training fold only and evaluate on a
  held-out fold (human + AI). Training on the same 200 human tracks we report on would leak.

## Minimum viable data size

- **[verified]** The paper gives no minimum training-set size or data-size ablation. Its scale is
  FakeMusicCaps = 27,605 generated tracks from ~5.5k real prompts, and SONICS real = **48,090 songs from
  9,096 artists**; training is 10 epochs, Adam, lr `5e-4`, batch 64.
- **[inference]** Our 200 human tracks (~160 per 5-fold training fold, ≈640 s of audio at 4.04 s/crop)
  are ~240× smaller than the SONICS real subset. A 5.6M-parameter one-class density model on that little
  audio is likely to memorize the crop distribution; a locally trained result would be **directional
  only**, with no defensible claim about cross-generator generalization. More real audio (or a much
  smaller/regularized model) would be needed for a meaningful comparison.

## Rough time, disk and VRAM cost

- **[verified, local probe]** `SpecNF(K=2, L=1, R=5.0)` has **5,616,368** trainable parameters
  (~22.5 MB fp32 state dict); forward+backward at batch 16 × 64,600 samples took **0.056 s/step** with
  **665 MiB** peak allocated VRAM; batch-12 forward took 0.0133 s (~900 tracks/s excluding audio I/O).
  No weights were downloaded and nothing was saved.
- **[verified, paper Table 3]** The authors report 8.13M params, 0.11 GB GPU memory, 4.09 GFLOPs and
  516 music/s on an RTX 4090 (batch 12). The parameter-count difference from our local probe is real and
  unexplained (likely a code/config revision difference) — both figures are reported, not reconciled.
- **[inference]** At 160 human tracks/fold, batch 16 gives ~10 steps/epoch, so 10 epochs ≈ 100 steps
  ≈ 6 s of GPU compute per fold (~30 s for 5 folds). Wall time will be dominated by thousands of
  `librosa` 4 s decodes, so budget minutes, not seconds. Pre-converting full tracks to 16 kHz float32
  WAV is optional (`preprocess_audio.py`); for 200 typical 3–5 min songs it would use roughly 2–4 GB of
  our 37 GB free disk. Inference weights are ~22 MB — negligible versus the other backends.

## Risks and unknowns

- **[verified, paper Table 6]** MusicDET is brittle to the kind of processing our catalog uses: at 64 kbps
  MP3 its EER rises from 4.51% to **41.75%**; pitch shifting → 44.73%, white noise → 44.11%, AAC 64 kbps
  → 35.85%, Opus 64 kbps → 22.15%. Our human tracks are gPodder-sourced podcast MP3s of unknown bitrate.
- **[verified]** The repository needs no new packages for the spec-nf path in our environment: all repo
  modules import under torch 2.6.0 / transformers 4.48.3 / numpy 1.26.4 / librosa 0.11.0; `einops` and
  `pytorch_wavelets` are absent but not needed on that path. Its `requirements.txt` pins different
  versions (Python 3.9, torch 2.7.1, transformers 4.46.3, librosa 0.10.2), so a spike may hit minor
  version drift. The authors' exact label CSVs are behind Google Drive and were not inspected.
- **[inference]** Our human set is genre-skewed and our AI set has unknown generators, so even a correct
  zero-shot run could not reproduce the paper's cross-generator tables. Likelihood→probability mapping
  and tri-state thresholds belong to Phase 5; do not hard-code a conversion here.

## Recommendation and gate D2

Recommend **defer**: keep Phase 5's D1 comparison to the four candidate model families with published weights. MusicDET
would add a training step, a new I/O wrapper, a calibration decision and a known MP3-robustness risk for
a result we could only call directional. If the owner instead wants the cross-generator angle pursued now,
the cheap bounded option is a zero-shot `spec-nf` spike on held-out human folds using generated protocol
files; that is a new phase of work, not part of this report.

**Owner decision (2026-11-19): defer MusicDET.** Keep the D1 comparison to the four available candidate
model families; reopen only with a new owner decision. Recorded in `ai/plans/plan.md` §5.

### 1. Gate D2 — what should we do with MusicDET?
*No checkpoint exists, so any benchmark requires local training; the owner decides, not this agent.*
- [x] **Defer MusicDET, keep D1 to the four available backends (Recommended)**. Cheapest, avoids an
      unevaluable 200-track model; revisit if cross-generator testing becomes a goal.
- [ ] Attempt a bounded zero-shot spike on our human tracks. Feasible and cheap (~22 MB weights,
      minutes), but results would be directional and need a new backend + held-out protocol.
- [ ] Drop MusicDET entirely. No further work; note it as unavailable in the final benchmark report.
- [ ] Other: ___
