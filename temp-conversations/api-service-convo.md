# API Service — Conversation Record

**Date:** 2026-11-21
**Status:** design discussion; no service code written.
**Library repo:** `<workspace>/ai-music-detector` (stays library-only).
**Remote:** `git@github.com:v4vmusic/ai-music-detector.git`.

This file is a working record of the API-service discussion. Decide later whether it is
committed or kept local.

## 1. Goal

Build an HTTP API service, packaged as a Docker container, around the `aidetect` library.
A site such as v4vmusic.com submits an audio file URL and gets back whether the track is
`ai_generated`, `human` or `uncertain`.

The service must be a thin adapter over the library. It must not reimplement detection,
URL safety or calibration logic.

## 2. Decisions from the first questionnaire

| # | Question | Answer |
| --- | --- | --- |
| 1 | Add the API service to the library plan? | **No.** The library plan is finished; keep the library repo library-only. A new plan will cover the service. |
| 2 | Where should the service code live? | **Undecided.** Owner asked for the best method. |
| 3 | How should it be packaged if in this repo? | **Undecided.** Follows question 2. |
| 4 | Default detection policy? | **D1 cascade checked**, but owner noted there is no GPU on the server and "we will not install the --cascade requirements". This conflicts with running the cascade; see §4. |
| 5 | Runtime/image? | **CPU-only server, no GPU.** "See previous answer." |
| 6 | Inputs for v1? | **URL only:** `POST /v1/score` with `{"url": "..."}`. |
| 7 | Sync or async? | **Synchronous** request/response with a timeout; no queue/DB worker. |
| 8 | URL-fetch safety? | **Reuse `aidetect.url` plus an explicit host allowlist and/or network egress rules.** |
| 9 | HTTP stack? | **FastAPI + uvicorn.** |
| 10 | Authentication? | **API key header; rate limiting at the reverse proxy.** |
| 11 | Response body? | **Library JSON as-is, plus an `X-Request-ID` header.** |

## 3. New requirements from the owner

- **Bulk URL submission:** accept up to `X` URLs in one request.
- **Result database:** persist results so an already-checked URL is not checked again.
- **Conversation record:** this file.

## 4. Blocking contradiction: cascade vs CPU-only

The owner selected the D1 cascade but also said there is no GPU and the cascade requirements
will not be installed.

Facts:

- The cascade's stage 2 is `spectttra-alpha-120s` (`STA-1b`). It needs `torch`, `torchaudio`,
  `sonics` and `huggingface_hub`. These can run on CPU; a GPU is not strictly required.
- The library's `gpu` extra installs the default (CUDA) `torch` wheels, which are large. There is
  no CPU-only SpecTTTra extra today.
- Stage 2 is **not a rare fallback**. Stage 1 (`fakeprint FP-1b`) auto-labels only high-confidence
  AI; it does not flag most human tracks. So stage 2 runs on almost every human track and on the
  AI tracks stage 1 missed. On a CPU-only server this is the dominant cost.
- Without `torch`/`sonics`, the service can only use `Detector` (fakeprint), not
  `CascadeDetector`. Fakeprint's packaged low/high thresholds coincide (`0.0796 / 0.0796`), so
  `uncertain` is effectively only the exact threshold value. The conservative review band is
  lost.

Options:

1. **Fakeprint-only with a new raw-score tri-state band (recommended for CPU-only).** Use the
   existing FP-1b raw scores to choose a high-confidence AI cutoff, a human cutoff and an
   `uncertain` band between them. Fast, small image, no torch. Requires a short dev-only tuning
   analysis and an owner decision; it is not the D1 cascade.
2. **Cascade on CPU.** Install CPU `torch` + `sonics`, keep D1 semantics, measure latency. Larger
   image and potentially slow synchronous responses because stage 2 runs on most tracks.
3. **Fakeprint-only with existing calibrated thresholds.** Smallest/fastest, but `uncertain` is
   effectively empty, so it does not meet the conservative tri-state goal.
4. **Cascade on a GPU host elsewhere / async.** Not available on this server; adds a queue or
   network hop.

The CPU policy must be decided before the API response shape is fixed: option 1 returns a
`Result` with a new label policy, option 2 returns a `CascadeResult`.

## 5. Recommendations

### 5.1 Repository and dependency method

**Recommended for v1:** a sibling repo, `ai-music-detector-api`, with a **path dependency** on
`../ai-music-detector`. Build the Docker image from the parent directory so the build context
contains both repos.

- Pros: no publishing step, no git credentials in Docker, library and service stay separate,
  simplest for a solo/local setup.
- Cons: the service repo cannot build alone; CI must check out both repos side by side.
- When to change: if the service needs independent CI/deployment, switch to a **published
  versioned wheel** (cleanest long-term) or a **pinned git dependency** with a BuildKit SSH mount
  or CI token.

Not recommended: git submodule or a vendored wheel.

### 5.2 Database and cache

**Recommended for v1:** SQLite in a mounted volume, WAL mode, one uvicorn worker.

- No separate database service; fits the single-container CPU deployment.
- Store the canonical library result JSON plus denormalized `label`, `probability`, `raw_score`,
  `model_id`, `model_revision`, `checked_at`, `expires_at`.
- Cache key: normalized URL **plus a policy/config version** so a threshold or model change does
  not serve stale labels.
- Invalidation: URL-keyed cache with a TTL (for example 30 days) and a `force` flag to bypass.
  URLs can point to different audio over time; a URL-only cache without a TTL can be stale.
- Failures: cache for a short TTL (for example 1 hour) or not at all; decide.
- Concurrency: one scoring lock/semaphore for CPU; one uvicorn worker for v1. Move to PostgreSQL
  and a shared lock only if multiple instances are needed.
- Privacy: store URLs and results, never audio. Avoid logging full query strings if they may
  contain tokens.

### 5.3 Bulk endpoint

- `POST /v1/score/batch` with a JSON array of URLs (or a file upload; decide).
- Recommended default limit: 20 URLs per request, configurable. Lower it for CPU cascade; raise
  it for fakeprint-only.
- Deduplicate URLs within a batch and return the same result for each occurrence.
- Per-item status so one bad URL does not fail the batch:
  `{"url", "status": "ok|cached|error", "result"|"error"}`.
- Bounded concurrency; bulk is synchronous only if the worst-case latency is acceptable. If not,
  bulk should be async submit/poll, which requires a queue and is a larger change.

### 5.4 API and operations

- Single response: library JSON as-is, `X-Request-ID` header, `X-Cache: HIT|MISS` header.
- Authentication: `X-API-Key` from `AIDETECT_API_KEYS` (comma-separated), constant-time compare;
  fail closed if no keys are configured. Health endpoint unauthenticated.
- Error mapping: invalid/unsafe URL → 4xx; upstream fetch/decode failure → 502/504; too many
  URLs → 413/422; internal model failure → 500.
- Docker: `python:3.10-slim` + `ffmpeg`, non-root user, mounted `AIDETECT_CACHE_DIR` and SQLite
  volume, healthcheck that does not load models, one uvicorn worker.
- Tests: offline FastAPI tests with a fake detector/fetcher and temporary SQLite; no network or
  GPU in the default suite. A Docker smoke test can check `/health`; a real URL score is hard to
  test offline because `aidetect.url` rejects loopback/private addresses.

### 5.5 New plan

The service repo should get its own `ai/plans/plan.md` and a coding contract (copy/adapt the
library's, or reference it). Suggested phases:

- Phase 0: repo scaffold, `pyproject.toml`, settings, dependency on the library, CI.
- Phase 1: single-URL endpoint, auth, error mapping, request IDs, health.
- Phase 2: SQLite cache (schema, hit/miss, TTL, force, policy version).
- Phase 3: bulk endpoint (format, limit, dedupe, partial failures, concurrency).
- Phase 4: Docker image and deployment docs.
- Phase 5: hardening (rate-limit docs, logging, metrics, retention).

## 6. Questionnaire transcript

### 1. Should we add the API service to the plan as a new phase?
- [ ] Yes — add a Phase 8.
- [ ] Yes, but as an unplanned spike first.
- [x] **No — keep this repo library-only and build the service outside it.**
- [ ] Other.

Owner note: "The plan.md is finished. We will likely create a new plan to implement this."

### 2. Where should the service code live?
- [ ] Same repo, new top-level `service/` directory.
- [ ] Separate repo under `<workspace>/`.
- [ ] Same repo, separate workspace member.
- [ ] Other.

Owner note: "I don't know. This is what we are trying to figure out, what is the best method."

### 3. If it stays in this repo, how should it be packaged?
- [ ] Add an `api` extra.
- [ ] Convert to a uv workspace.
- [ ] Keep it out of the distribution.
- [ ] Other.

Owner note: "See previous answer."

### 4. Which detection policy should the service expose by default?
- [x] **D1 cascade (`CascadeDetector`: FP-1b + STA-1b).**
- [ ] Single-model fakeprint.
- [ ] Configurable per deployment.
- [ ] Other.

Owner note: "Yes, there will be no GPU on it's server so we will not install the --cascade
requirements."

### 5. What runtime should the cascade use?
- [ ] GPU image.
- [ ] CPU-only image.
- [ ] Two image targets.
- [x] **Other.**

Owner note: "See previous answer." (No GPU.)

### 6. What inputs should v1 accept?
- [x] **URL only: `POST /v1/score` with `{"url": "..."}`.**
- [ ] URL and multipart upload.
- [ ] Upload only.
- [ ] Other.

### 7. Synchronous or asynchronous API?
- [x] **Synchronous request/response with a request timeout.**
- [ ] Async submit/poll job API.
- [ ] Other.

### 8. How should server-side URL fetching be constrained?
- [x] **Reuse `aidetect.url` plus an explicit host allowlist and/or network egress rules.**
- [ ] Reuse `aidetect.url` only.
- [ ] Upload only.
- [ ] Other.

### 9. Which HTTP stack?
- [x] **FastAPI + uvicorn.**
- [ ] Flask + gunicorn.
- [ ] Standard library only.
- [ ] Other.

### 10. How should callers authenticate?
- [x] **API key header; rate limiting at the reverse proxy.**
- [ ] API key + in-process rate limit.
- [ ] No auth; trusted network only.
- [ ] Other.

### 11. What should the response body look like?
- [x] **Library JSON as-is, plus an `X-Request-ID` header.**
- [ ] Small envelope with request/service metadata.
- [ ] Minimal fields only.
- [ ] Other.

## 7. Follow-up questions

1. CPU policy: fakeprint-only raw-score band, cascade on CPU, or fakeprint-only with existing
   thresholds?
2. Repo/dependency method: sibling path dependency, pinned git dependency, or published wheel?
3. Database: SQLite or PostgreSQL?
4. Cache invalidation: TTL, indefinite, or content-hash revalidation?
5. Cache failures? Short TTL, never, or indefinitely?
6. Bulk input: JSON array, CSV/text file upload, or both?
7. Bulk maximum: 20, 50, 100, or async?
8. Bulk response: per-item envelope, omit failures, or fail the whole batch?
9. API key details: `X-API-Key`, `Authorization: Bearer`, or DB-managed keys?
10. New plan/contract: create a service repo with its own plan and contract, or reuse the
    library's contract by reference?

## 8. Final answers (2026-11-21)

The owner answered the §7 follow-up and the later bulk/deploy questionnaire:

| # | Question | Answer |
| --- | --- | --- |
| 1 | CPU policy | **Fakeprint-only `FP-1b` with a new raw-score tri-state band.** No cascade, no torch. |
| 2 | Repo/dependency | **Sibling `ai-music-detector-api`, path dependency on `../ai-music-detector`.** |
| 3 | Database | **SQLite in a mounted volume, WAL, one uvicorn worker.** |
| 4 | Cache invalidation | **URL indefinitely, plus `policy_version` in the key.** |
| 5 | Cache failures | **Cache failures for 1 hour.** |
| 6 | Bulk input | **JSON array `{"urls": [...]}`.** |
| 7 | Bulk limit | **100 URLs.** |
| 8 | Bulk response | **Per-item envelope `{"url","status","result"|"error"}`.** |
| 9 | API key | **`X-API-Key` from `AIDETECT_API_KEYS`, constant-time compare.** |
| 10 | Plan/contract | **Service repo with `ai/plans` (filename must include “plan”), an adapted `ai/contracts`, and recreated prompts.** |
| 11 | Bulk processing | **Async job: `202` + `job_id`; `GET /v1/jobs/{id}` polls.** |
| 12 | Job retention | **7 days; cached URL results indefinitely.** |
| 13 | Deployment target | **Same host/user (`<deploy-user>@<deploy-host>`), new remote path, same `proxy-network`.** |
| 14 | Exposure | **External `proxy-network`, expose 8000, NPM handles TLS and rate limiting.** |
| 15 | Backup/migrate | **Conditional `python -m aidetect_api.db backup`, scp to local backups; idempotent `migrate` after `compose up`.** |
| 16 | Image tags | **Timestamped tag plus `latest`; keep the previous image for rollback.** |
| 17 | Service contract | **Own `ai/contracts/coding-contract.md` adapted from the library contract.** |
| 18 | Policy changes | **URL results indefinite but include `policy_version` in the cache key and support `force=true`.** |
| 19 | Weights | **Mount `AIDETECT_CACHE_DIR` and add a post-deploy warmup step.** |

Additional owner notes:

- “If we accept 100 URLs, they will take some time to process (could be several minutes). How
  should we handle this?” → answered by the async job design.
- Deployment should follow `libre-splitkit/scripts/deploy/deploy.sh`.
- NPM is non-standard; read the remote documentation before touching proxy hosts.
- “If we're going to create an `ai` folder in the service repo then we might as well put the plan
  inside it also and recreate the prompts folder modified for the service-repo's requirements.”

## 9. Raw-score band selected

Dev-only analysis of `benches/results/tuning/FP-1b/raw/fakeprint.jsonl` (seed 42, fixed dev/test
split) produced:

```text
raw_score >= 4.139542579650879e-05   -> ai_generated
raw_score <  1.206994056701655e-07   -> human
otherwise                            -> uncertain
```

| Split | True label | `human` | `uncertain` | `ai_generated` |
| --- | --- | --- | --- | --- |
| dev | human | 152 | 8 | 0 |
| dev | ai_generated | 8 | 16 | 136 |
| test | human | 34 | 6 | 0 |
| test | ai_generated | 2 | 3 | 35 |

The high cutoff is the packaged `FP-1b` `flag_threshold` (dev 85% AI recall); the low cutoff is
the dev human 95th percentile. Held-out test has **0/40 human auto-AI labels**. A lower high
cutoff (dev AI q0.075) would have raised dev recall to 92.5% but produced 2/40 held-out human
false positives, so it was rejected.

## 10. Next steps

- Service repo created at `<workspace>/ai-music-detector-api`.
- Plan, contract, prompts and reference log live under its `ai/` folder.
- Confirm the remote path, public domain and production `AIDETECT_API_KEYS` before the first
  deploy.
- Create and verify the NPM proxy host using the remote runbook.
