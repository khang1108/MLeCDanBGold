# LLM inference module

`hcmai.llm` is the boundary between the local HCMAI search application and the
models hosted on a temporary GPU VM. It serves only model inference. The
keyframes, metadata, embeddings, FAISS indexes, FastAPI
search backend, and React UI remain on the local machine.

For VM provisioning, Supervisor, and Cloudflare Tunnel setup, see the
[deployment guide](../../../llm/README.md). This document describes the Python
module and its contracts.

## Architecture

```text
React UI
   │ localhost:8000
   ▼
Local FastAPI ── retrieval / VQA / frame materialization
   │
   │ LLMService + HTTP adapter + optional Cloudflare Access headers
   ▼
api.iamphuckhang.dev ── Cloudflare Tunnel ── localhost:8100 on GPU VM
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
| `models/contracts.py` | Adapter-facing data contracts only |

The authoritative request and response models are in
[`common/schemas/inference.py`](../common/schemas/inference.py).

## Model lifecycle

Importing this package does not load model weights. In production:

1. `LLMService.from_environment()` reads `llm/config.yaml` through its local
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

The checked-in [`llm/config.yaml`](../../../llm/config.yaml) configures:

- `google/siglip2-base-patch16-224` for visual-query embeddings;
- `BAAI/bge-m3` for multilingual caption/query dense embeddings;
- Florence for caption generation;
- `Qwen/Qwen3-VL-Reranker-2B` for image-query reranking;
- `zai-org/GLM-4.1V-9B-Thinking` only for grounded VQA.

Relevant environment variables are:

| Variable | Used by | Meaning |
| --- | --- | --- |
| `HCMAI_LLM_CONFIG` | GPU service | YAML path; defaults to `llm/config.yaml` |
| `HCMAI_VQA_MODEL` | GPU service | Non-empty VQA checkpoint override |
| `HCMAI_ENABLE_CAPTION` | GPU service | Load caption generation model |
| `HCMAI_ENABLE_VISUAL_EMBEDDING` | GPU service | Load SigLIP2 visual/query encoder |
| `HCMAI_ENABLE_CAPTION_EMBEDDING` | GPU service | Load BGE-M3 caption/query encoder |
| `HCMAI_ENABLE_RERANKER` | GPU service | Load image-query reranker |
| `HCMAI_ENABLE_VQA` | GPU service | Load grounded VQA model |
| `HCMAI_INFERENCE_BASE_URL` | Local backend | Hosted API base URL |
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

HCMAI_LLM_CONFIG=llm/config.yaml \
PYTHONPATH=src aic/bin/python -m uvicorn hcmai.llm.server.api:app \
  --host 127.0.0.1 --port 8100 --workers 1
```

The first production start downloads the configured checkpoints. Unit tests
inject fake runtimes and model backends, so they never download checkpoints or
load the real corpus.

Code outside `hcmai.llm` imports only `LLMService` and configuration contracts
from `hcmai.llm.pipeline`. It must not import `server/`, `adapters/`, or
`models/` directly. The private server is the intentional exception because it
is the transport entry point owned by this component.

## Pipeline behavior

- KISC sends the current message, bounded history, feedback, prior state, and
  output schema to the conversation endpoint.
- Its `standalone_query` is an English, self-contained visual description used
  by retrieval; it is not a conversational answer to the user.
- The remote SigLIP2 encoder converts that query into the same vector space as
  the local image index.
- Dense retrieval selects candidates from local FAISS artifacts.
- The remote Qwen model only reorders those candidates. It cannot recover a
  relevant frame that dense retrieval did not include.
- Frame loading, exact `frame_id`/`video_id`/`frame_idx` mapping, and final
  response materialization stay local.
- `/v1/vqa` answers one question about one supplied canonical frame and accepts
  optional caption, OCR, ASR, and object evidence. It reuses the configured
  GLM vision model and preserves request/frame identity.

The configured competition path is fail-fast: embedding, conversation, image
loading, invalid reranker scores, and remote inference failures abort the
request instead of silently switching models or preserving an older ranking.

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
PYTHONPATH=src aic/bin/pytest -q \
  tests/test_llm_api.py \
  tests/test_llm_conversation.py \
  tests/test_encoder.py \
  tests/test_qwen_reranker.py

pyright src/hcmai/llm
```

The local backend startup and UI workflow are documented in the
[root README](../../../README.md).
