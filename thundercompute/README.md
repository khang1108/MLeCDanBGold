# ThunderCompute inference module

`thundercompute` is the boundary between the local HCMAI search
application and the models hosted on a temporary GPU VM. It serves only model
inference. The
keyframes, metadata, embeddings, FAISS indexes, FastAPI
search backend, and React UI remain on the local machine.

For VM provisioning, Supervisor, and Cloudflare Tunnel setup, see the
[lifecycle controller section](#lifecycle-controller). This document describes the Python
module and its contracts.

## Architecture

```text
React UI
   │ localhost:8000
   ▼
Local FastAPI ── retrieval / VQA / frame materialization
   │
   │ Docker: LLMService + HTTP adapter
   ▼
LiteLLM private pass-through ── Cloudflare Access ── api.iamphuckhang.dev
                                      │
                                      ▼
                         Cloudflare Tunnel ── localhost:8100 on GPU VM
                                                   │
                                                   ▼
                                  FastAPI API → LLMService
                                                ├─ SigLIP2/BGE encoders
                                                ├─ Florence captioner
                                                ├─ Qwen VL reranker
                                                └─ grounded VQA model
```

The browser never calls the GPU service directly. This keeps Cloudflare
credentials out of the frontend and avoids moving the roughly 100 GB retrieval
corpus to a disposable VM.

## Files

| File | Responsibility |
| --- | --- |
| `pipeline.py` | Public `LLMService` lifecycle and inference facade |
| `server/api.py` | Private FastAPI endpoints, request limits, error translation |
| `adapters/local.py` | Single-process ownership and lifecycle of hosted models |
| `adapters/http.py` | Bounded synchronous client for remote inference |
| `adapters/vqa.py` | Grounded single/multi-frame VQA model |
| `config.py` | Typed model and service configuration loaded from YAML |
| `common/schemas/inference.py` | Shared inference request/response contracts |

The authoritative request and response models are in
[`src/hcmai/common/schemas/inference.py`](../src/hcmai/common/schemas/inference.py).

## Model lifecycle

Importing this package does not load model weights. In production:

1. `LLMService.from_environment()` reads `thundercompute/config.yaml` through its local
   adapter.
2. The FastAPI lifespan calls `service.load()` once.
3. Only enabled model groups stay in memory.
4. Every request reuses those instances.

Run exactly one Uvicorn worker. Additional workers duplicate all model weights
in GPU memory.

`GET /health` only proves that the HTTP process responds. `GET /ready` verifies
that every enabled model is loaded and returns checkpoint provenance.
VQA inference is optional when `HCMAI_ENABLE_VQA=false`.

## Configuration

The checked-in [`config.yaml`](./config.yaml) configures:

- `google/siglip2-base-patch16-224` for visual-query embeddings;
- `BAAI/bge-m3` for multilingual caption/query dense embeddings;
- Florence for caption generation;
- `Qwen/Qwen3-VL-Reranker-2B` for image-query reranking;
- `zai-org/GLM-4.1V-9B-Thinking` only for grounded VQA.

Relevant environment variables are:

| Variable | Used by | Meaning |
| --- | --- | --- |
| `HCMAI_LLM_CONFIG` | GPU service | YAML path; defaults to `thundercompute/config.yaml` |
| `HCMAI_VQA_MODEL` | GPU service | Non-empty VQA checkpoint override |
| `HCMAI_ENABLE_CAPTION` | GPU service | Load caption generation model |
| `HCMAI_ENABLE_VISUAL_EMBEDDING` | GPU service | Load SigLIP2 visual/query encoder |
| `HCMAI_ENABLE_CAPTION_EMBEDDING` | GPU service | Load BGE-M3 caption/query encoder |
| `HCMAI_ENABLE_RERANKER` | GPU service | Load image-query reranker |
| `HCMAI_ENABLE_VQA` | GPU service | Load grounded VQA model |
| `HCMAI_INFERENCE_BASE_URL` | Local backend | Hosted API base URL; Docker Compose overrides it with `http://litellm:4000` |
| `HCMAI_CF_ACCESS_CLIENT_ID` | Local backend | Cloudflare service credential |
| `HCMAI_CF_ACCESS_CLIENT_SECRET` | Local backend | Cloudflare service credential |

An empty `HCMAI_VQA_MODEL` leaves the YAML checkpoint unchanged. Set
`HCMAI_ENABLE_VQA=false` to disable VQA model loading.

Each remote embedding checkpoint, vector dimension, normalization, and dtype
must remain compatible with its visual or caption FAISS artifact. A different text
encoder cannot safely query an index created in another embedding space.

## API

| Method and path | Input | Output |
| --- | --- | --- |
| `GET /health` | None | Process liveness |
| `GET /ready` | None | Per-model readiness and provenance |
| `POST /v1/captions` | Multipart IDs and images | Caption for each input ID |
| `POST /v1/embeddings/text` | JSON with 1–64 texts | Normalized text vectors |
| `POST /v1/rerank` | Multipart query, IDs, and images | Score for each input ID |
| `POST /v1/vqa` | Question, canonical frame, and evidence | Grounded answer bound to the frame |

Check a running service:

```bash
curl -sS http://127.0.0.1:8100/health
curl -sS http://127.0.0.1:8100/ready
```

Request text embeddings:

```bash
curl -sS http://127.0.0.1:8100/v1/embeddings/text \
  -H 'Content-Type: application/json' \
  -d '{"texts":["a 60-second timer","a red bus on a city street"]}'
```

Reranking accepts 1–100 unique item IDs and the same number of image parts.
Each uploaded image is limited to 5 MB. The client thumbnails candidates to at
most 768 × 768 and encodes them as JPEG:

```bash
curl -sS http://127.0.0.1:8100/v1/rerank \
  -F 'query=a person holding a red umbrella' \
  -F 'item_ids=["frame-a","frame-b"]' \
  -F 'images=@frame-a.jpg' \
  -F 'images=@frame-b.jpg'
```

## Run locally

Install the supported dependencies through the repository environment, then
start the service from the repository root:

```bash
aic/bin/python -m pip install -e ".[embedding,dev]"

HCMAI_LLM_CONFIG=thundercompute/config.yaml \
PYTHONPATH=.:src aic/bin/python -m uvicorn thundercompute.server.api:app \
  --host 127.0.0.1 --port 8100 --workers 1
```

The first production start downloads the configured checkpoints. Unit tests
inject fake runtimes and model backends, so they never download checkpoints or
load the real corpus.

Code outside `thundercompute` imports only `LLMService` and configuration
contracts from `thundercompute.pipeline`. It must not import `server/` or
`adapters/` directly. The private server is the intentional exception because it
is the transport entry point owned by this component.

## Pipeline behavior

- The remote SigLIP2 encoder converts each search query into the same vector space as
  the local image index.
- Dense retrieval selects candidates from local FAISS artifacts.
- The remote Qwen model only reorders those candidates. It cannot recover a
  relevant frame that dense retrieval did not include.
- Frame loading, exact `frame_id`/`video_id`/`frame_idx` mapping, and final
  response materialization stay local.
- `/v1/vqa` answers one question about one supplied canonical frame and accepts
  optional caption, OCR, ASR, and object evidence. It reuses the configured
  GLM vision model and preserves request/frame identity.

The configured competition path categorizes embedding, image loading, invalid
reranker scores, and remote inference failures and applies the task's bounded
fallback policy.

## Troubleshooting

| Symptom | Likely cause and action |
| --- | --- |
| `conversation checkpoint is not configured` | The deployed YAML has a null checkpoint or the VM is on an old commit. Inspect `/ready` and redeploy the current `main`. |
| `string indices must be integers, not 'str'` | An old GLM request format sent plain strings instead of content blocks. Deploy the current conversation implementation. |
| Cloudflare returns `502 Bad Gateway` | The tunnel origin is unavailable or uses the wrong scheme. Route it to `http://localhost:8100`, then check Supervisor logs. |
| `/health` succeeds but `/ready` returns 503 | HTTP is alive but at least one required model failed to load. Inspect GPU memory and the model-service log. |
| Embedding checkpoint/dimension mismatch | Hosted encoder and local index provenance differ. Restore the matching config; do not ignore the validation. |
| Queries return unrelated frames | Verify the standalone query and embedding compatibility before tuning reranking. Reranking cannot repair a poor candidate set. |
| Conversation request approaches the client timeout | Thinking models may generate long reasoning. Reduce `max_new_tokens`, increase the configured timeout, or later adopt a smaller resolver model. |

Inference errors are returned as bounded HTTP 503 details. Do not put
Cloudflare tokens or service credentials in YAML, source control, browser code,
or issue reports.

## Verification

Run the focused tests without loading real models:

```bash
PYTHONPATH=.:src aic/bin/pytest -q \
  tests/test_llm_api.py \
  tests/unit/llm/test_inference_gateway.py \
  tests/unit/llm/test_multiframe_vqa.py

python -m compileall -q thundercompute
```

The local backend startup and UI workflow are documented in the
[root README](../README.md).

## Lifecycle controller

The same root directory also owns the laptop-side controller for the
disposable Thunder GPU VM. It does not contain the Cloudflare token or the
private bootstrap implementation in version control.

```text
Docker Compose profile
  -> tnr create
  -> wait for RUNNING
  -> tnr scp deploy_cloudflared_private.sh 0:/home/ubuntu/
  -> SSH command starts the private bootstrap in tmux
  -> graceful controller stop
  -> tnr delete
```

Thunder's documented CLI exposes `create`, `scp`, `connect`, and `delete`, but
not a non-interactive `tnr exec` command. `launch.sh` therefore uses the SSH
host, port, UUID, and key material made available by `tnr status --json` after
the upload. The remote command is still executed without an interactive
terminal.

### Private bootstrap

The private `deploy_cloudflared_private.sh` is ignored by Git. To create a
safe local copy from the checked-in template:

```bash
cp thundercompute/deploy_cloudflared_private.sh.example \
  thundercompute/deploy_cloudflared_private.sh
chmod 700 thundercompute/deploy_cloudflared_private.sh
```

The bootstrap is uploaded to `/home/ubuntu/deploy_cloudflared_private.sh` and
must configure the current `thundercompute.server.api` service, start the
Cloudflare tunnel for `api.iamphuckhang.dev`, and verify `/health` and `/ready`.
The launcher never receives the Cloudflare token as a command-line argument.

### Docker lifecycle

The default Compose stack starts backend, LiteLLM, and frontend. The
ThunderCompute controller is an opt-in `thundercompute` profile:

```bash
mkdir -p .secrets .thundercompute
printf '%s\n' 'your-thunder-api-token' > .secrets/tnr_api_token

export TNR_API_TOKEN_FILE=.secrets/tnr_api_token
export HCMAI_THUNDER_DEPLOY_SCRIPT=./thundercompute/deploy_cloudflared_private.sh

docker compose --profile thundercompute build thundercompute
docker compose --profile thundercompute up thundercompute
```

Run the normal application stack in another terminal when needed:

```bash
docker compose up -d
```

Use `docker compose stop thundercompute` or `docker compose down` to send a
graceful signal. The controller trap then calls `tnr delete` and clears
`.thundercompute/instance-id`. `docker kill` defaults to `SIGKILL`, which no
process can catch; after a forced kill run:

```bash
TNR_API_TOKEN_FILE=.secrets/tnr_api_token \
  bash thundercompute/delete.sh
```

Do not set `restart: always` for this controller: restarting it after a clean
exit would create another billable VM.

### Host-side operation

The same controller can run without Docker when `tnr`, Python 3, and OpenSSH
are installed on the laptop:

```bash
TNR_API_TOKEN_FILE=.secrets/tnr_api_token \
  HCMAI_THUNDER_DEPLOY_SCRIPT=./thundercompute/deploy_cloudflared_private.sh \
  bash thundercompute/launch.sh --gpu l40 --
```

Press Ctrl-C for cleanup. Use `--keep` only when the instance is intentionally
managed separately, and use `--instance ID --delete-reused` only when the
controller owns that existing instance.
