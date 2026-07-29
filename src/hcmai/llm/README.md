# LLM inference module

`hcmai.llm` is the boundary between the local HCMAI search application and the
models hosted on a temporary GPU VM. It serves only model inference. The
keyframes, metadata, embeddings, FAISS indexes, KISC orchestration, FastAPI
search backend, and React UI remain on the local machine.

For VM provisioning, Supervisor, and Cloudflare Tunnel setup, see the
[deployment guide](../../../llm/README.md). This document describes the Python
module and its contracts.

## Architecture

```text
React UI
   │ localhost:8000
   ▼
Local FastAPI ── KISC / retrieval / frame materialization
   │
   │ InferenceClient + optional Cloudflare Access headers
   ▼
api.iamphuckhang.dev ── Cloudflare Tunnel ── localhost:8100 on GPU VM
                                                   │
                                                   ▼
                                  FastAPI API → LLMRuntime
                                                ├─ SigLIP2 text encoder
                                                ├─ Qwen VL reranker
                                                └─ GLM conversation resolver
```

The browser never calls the GPU service directly. This keeps Cloudflare
credentials out of the frontend and avoids moving the roughly 100 GB retrieval
corpus to a disposable VM.

## Files

| File | Responsibility |
| --- | --- |
| `api.py` | Private FastAPI endpoints, request limits, error translation |
| `client.py` | Local synchronous client and remote dense-encoder adapter |
| `config.py` | Typed model and service configuration loaded from YAML |
| `conversation.py` | Structured conversation inference and JSON extraction |
| `runtime.py` | Single-process ownership and lifecycle of all hosted models |
| `__init__.py` | Public client exports used by the local backend |

The authoritative request and response models are in
[`common/schemas/inference.py`](../common/schemas/inference.py), while KISC
conversation state is defined in
[`common/schemas/conversation.py`](../common/schemas/conversation.py).

## Model lifecycle

Importing this package does not load model weights. In production:

1. `LLMRuntime.from_environment()` reads `llm/config.yaml`.
2. The FastAPI lifespan calls `runtime.load()` once.
3. Only enabled model groups stay in memory.
4. Every request reuses those instances.

Run exactly one Uvicorn worker. Additional workers duplicate all model weights
in GPU memory.

`GET /health` only proves that the HTTP process responds. `GET /ready` verifies
that every required model is loaded and returns checkpoint provenance.
Conversation inference is optional when `conversation.checkpoint` is `null`.

## Configuration

The checked-in [`llm/config.yaml`](../../../llm/config.yaml) configures:

- `google/siglip2-base-patch16-224` for visual-query embeddings;
- `BAAI/bge-m3` for multilingual caption/query dense embeddings;
- `Qwen/Qwen3-VL-Reranker-2B` for image-query reranking;
- `zai-org/GLM-4.1V-9B-Thinking` for KISC state resolution.

Relevant environment variables are:

| Variable | Used by | Meaning |
| --- | --- | --- |
| `HCMAI_LLM_CONFIG` | GPU service | YAML path; defaults to `llm/config.yaml` |
| `HCMAI_CONVERSATION_MODEL` | GPU service | Non-empty checkpoint override |
| `HCMAI_ENABLE_VISUAL_EMBEDDING` | GPU service | Load SigLIP2 visual/query encoder |
| `HCMAI_ENABLE_CAPTION_EMBEDDING` | GPU service | Load BGE-M3 caption/query encoder |
| `HCMAI_INFERENCE_BASE_URL` | Local backend | Hosted API base URL |
| `HCMAI_CF_ACCESS_CLIENT_ID` | Local backend | Cloudflare service credential |
| `HCMAI_CF_ACCESS_CLIENT_SECRET` | Local backend | Cloudflare service credential |

An empty `HCMAI_CONVERSATION_MODEL` does not disable conversation inference; it
leaves the YAML checkpoint unchanged. Set `conversation.checkpoint: null` in
the YAML to disable it explicitly.

Each remote embedding checkpoint, vector dimension, normalization, and dtype
must remain compatible with its visual or caption FAISS artifact. A different text
encoder cannot safely query an index created in another embedding space.

## API

| Method and path | Input | Output |
| --- | --- | --- |
| `GET /health` | None | Process liveness |
| `GET /ready` | None | Per-model readiness and provenance |
| `POST /v1/embeddings/text` | JSON with 1–64 texts | Normalized text vectors |
| `POST /v1/rerank` | Multipart query, IDs, and images | Score for each input ID |
| `POST /v1/conversation/resolve` | Complete bounded KISC context | `ConversationState` |

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

Resolve one KISC turn:

```bash
curl -sS http://127.0.0.1:8100/v1/conversation/resolve \
  -H 'Content-Type: application/json' \
  -d '{
    "instruction":"Convert the conversation into structured search state.",
    "history":[],
    "current_message":"Find a 60-second timer visual",
    "feedback":null,
    "previous_state":null,
    "response_schema":{}
  }'
```

The response must contain all six state fields: `standalone_query`,
`positive_constraints`, `negative_constraints`, `uncertain_constraints`,
`accepted_frame_ids`, and `rejected_frame_ids`. The conversation loader uses
GLM's multimodal processor and text content blocks, then extracts the first
complete JSON object even when a thinking model emits reasoning before it.

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
PYTHONPATH=src aic/bin/python -m uvicorn hcmai.llm.api:app \
  --host 127.0.0.1 --port 8100 --workers 1
```

The first production start downloads the configured checkpoints. Unit tests
inject fake runtimes and model backends, so they never download checkpoints or
load the real corpus.

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

When configured, `RemoteDenseEncoder` can fall back to the local encoder after
a transport or validation failure. Conversation failures use the deterministic
KISC fallback, while reranker failures preserve dense-retrieval order.

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
