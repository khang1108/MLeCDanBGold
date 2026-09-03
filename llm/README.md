# LLM shared inference module

`llm` is the boundary between the local HCMAI application and models
hosted on a temporary GPU VM. It serves shared model inference only. Canonical
keyframes, metadata, enrichment artifacts, FAISS indexes, search backend and
React UI remain local.

The public/runtime task surface is KIS and TRAKE. This module does not own task
pipelines or submission materialization; it supplies bounded model operations
that those workflows can share.

## Architecture

```text
React UI
   │ localhost:8000
   ▼
Local FastAPI ── KIS/TRAKE retrieval, temporal alignment, materialization
   │ HTTP adapter
   ▼
Hosted inference URL ── Cloudflare Tunnel ── localhost:8100 on GPU VM
                                                   │
                                                   ▼
                                  FastAPI API → LLMService
                                                ├─ SigLIP2/BGE encoders
                                                ├─ Qwen VL captioner
                                                ├─ OCR/ASR/diarization services
                                                └─ Qwen VL reranker
```

The browser never calls the GPU service directly. This keeps Cloudflare
credentials out of the frontend and avoids moving the retrieval corpus to a
disposable VM.

## Files and lifecycle

| File | Responsibility |
| --- | --- |
| `pipeline.py` | Public `LLMService` lifecycle and inference facade |
| `server/api.py` | Stable lazy ASGI entry point |
| `server/routers/` | Capability-focused FastAPI endpoints |
| `server/parsing.py` | Bounded multipart image/tensor validation |
| `local/` | Single-process model lifecycle, audio input, and readiness |
| `remote/` | HTTP client, endpoint pool, retries, circuit breaker, and bulkhead |
| `query_preparation/` | Hosted structured-generation adapter |
| `config.py` | Typed model and service configuration loaded from YAML |
| `config.yaml` | Pinned enabled-model configuration |

The authoritative request/response contracts live in
[`src/hcmai/common/schemas/inference.py`](../src/hcmai/common/schemas/inference.py).

Importing `llm` does not load weights. In production,
`LLMService.from_environment()` reads `llm/config.yaml`, the FastAPI
lifespan loads enabled model groups once, and every request reuses them. Run one
Uvicorn worker: additional workers duplicate model weights in GPU memory.

`GET /health` proves that the process responds. `GET /ready` verifies that all
enabled model groups are loaded and returns checkpoint provenance.

## Configuration and compatibility

The checked-in configuration pins the visual encoder, multilingual text
encoder, caption model and image-query reranker. Environment variables control
which shared capabilities load:

| Variable | Used by | Meaning |
| --- | --- | --- |
| `HCMAI_LLM_CONFIG` | GPU service | YAML path; defaults to `llm/config.yaml` |
| `HCMAI_ENABLE_CAPTION` | GPU service | Load caption generation |
| `HCMAI_ENABLE_VISUAL_EMBEDDING` | GPU service | Load visual/query encoder |
| `HCMAI_ENABLE_CAPTION_EMBEDDING` | GPU service | Load text encoder |
| `HCMAI_ENABLE_RERANKER` | GPU service | Load image-query reranker |
| `HCMAI_ENABLE_OCR` | GPU service | Load OCR capability |
| `HCMAI_ENABLE_ASR` | GPU service | Load ASR capability |
| `HCMAI_ENABLE_DIARIZATION` | GPU service | Load diarization capability |
| `HCMAI_INFERENCE_BASE_URL` | Local backend | Hosted inference API base URL |

Every remote encoder checkpoint, vector dimension, normalization and dtype must
remain compatible with its local FAISS artifact. A different embedding contract
must not query an index built in another vector space.

## Private inference API

| Method and path | Input | Output |
| --- | --- | --- |
| `GET /health` | None | Process liveness |
| `GET /ready` | None | Enabled-model readiness and provenance |
| `POST /v1/captions` | Multipart IDs and images | Caption for each input ID |
| `POST /v1/enrichment/ocr` | Multipart IDs and images | OCR evidence for each input ID |
| `POST /v1/embeddings/text` | JSON text batch | Normalized text vectors |
| `POST /v1/embeddings/images` | Multipart IDs and images | Visual vectors |
| `POST /v1/embeddings/dino` | Multipart IDs and images | DINO visual vectors |
| `POST /v1/preprocessing/shot-scores` | Ordered images | Shot-boundary scores |
| `POST /v1/preprocessing/event-scores` | Ordered images | Event-boundary scores |
| `POST /v1/preprocessing/event-window-scores` | Ordered images | Event-window scores |
| `POST /v1/transcripts/asr` | Audio reference | Timestamped transcript segments |
| `POST /v1/transcripts/diarization` | Audio reference | Timestamped diarized segments |
| `POST /v1/rerank` | Query, IDs and images | Score for each supplied ID |

For example:

```bash
curl -sS http://127.0.0.1:8100/health
curl -sS http://127.0.0.1:8100/ready

curl -sS http://127.0.0.1:8100/v1/embeddings/text \
  -H 'Content-Type: application/json' \
  -d '{"texts":["a 60-second timer","a red bus on a city street"]}'
```

Reranking accepts the bounded candidate set supplied by local retrieval. It may
reorder candidates but cannot recover a frame that retrieval did not include;
canonical frame/video/submission identity remains local.

## Run locally

Install the supported dependencies and start the private service from the
repository root:

```bash
aic/bin/python -m pip install -e ".[embedding,dev]"

HCMAI_LLM_CONFIG=llm/config.yaml \
PYTHONPATH=.:src aic/bin/python -m uvicorn llm.server.api:app \
  --host 127.0.0.1 --port 8100 --workers 1
```

To run only the two CPU embedding models from `config.yaml` and connect the
existing named Cloudflare Tunnel using the token kept in the ignored private
deployment script:

```bash
chmod +x llm/scripts/run_local_cpu_cloudflared.sh
llm/scripts/run_local_cpu_cloudflared.sh
```

The launcher disables GPU-only capabilities explicitly, keeps the token out of
the process command line, and stops Uvicorn and `cloudflared` together on
Ctrl+C. It expects `cloudflared`, `curl`, and the existing `aic` environment;
run it with `--help` to override the Python executable, config, port, or private
credentials-file path.

The first production start downloads configured checkpoints. Unit tests inject
fake runtimes/model backends and do not load checkpoints or corpus artifacts.

Code outside this package imports only `LLMService` from `llm.pipeline` and
configuration/contracts from their authoritative root modules. It must not
import `server/`, `local/`, or `remote/` directly. The private server process is
the intentional transport-entry exception.

## Manual GPU VM deployment

VM lifecycle is intentionally operator-run. There is no tracked launcher,
delete script, deployment-script template or container lifecycle controller. Use
your authenticated `tnr` CLI profile, upload the source/config bundle selected
for the deployment, connect to the VM, and run the service there.

```bash
# Create the disposable VM and wait until its status is RUNNING.
tnr create --gpu l40 --num-gpus 1 --vcpus 8 --template base --disk 200 --yes
INSTANCE_ID=<instance-id>
tnr status --no-wait --json

# Copy the reviewed inference source/config package for this deployment.
# Use the CLI's recursive-copy option when the package is a directory.
tnr scp <local-inference-package> "${INSTANCE_ID}:/home/ubuntu/hcmai/" --yes
tnr connect "${INSTANCE_ID}"

# On the VM, install the package's pinned dependencies and start one worker.
cd /home/ubuntu/hcmai
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[embedding]"
HCMAI_LLM_CONFIG=llm/config.yaml \
PYTHONPATH=.:src python -m uvicorn llm.server.api:app \
  --host 127.0.0.1 --port 8100 --workers 1

# From the local machine after the service is no longer needed.
tnr delete --yes "${INSTANCE_ID}"
```

The operator configures the Cloudflare tunnel and its credential through the
VM's private environment or secret store, then verifies `/health` and `/ready`
through the protected hosted URL. Do not put Thunder tokens, Cloudflare tokens
or private deployment code in tracked source or command history. `.secrets/`
and private operator-script paths remain Git ignored.

## Troubleshooting and verification

| Symptom | Likely cause and action |
| --- | --- |
| Cloudflare returns `502 Bad Gateway` | Tunnel origin is unavailable or uses the wrong scheme; route it to `http://localhost:8100` and inspect operator-managed service logs. |
| `/health` succeeds but `/ready` returns `503` | HTTP is alive but an enabled model failed to load; inspect GPU memory and service logs. |
| Embedding checkpoint/dimension mismatch | Hosted encoder and local index provenance differ; deploy the matching configuration. |
| Queries return unrelated frames | Verify query/index compatibility before tuning reranking; reranking cannot repair a poor candidate set. |

Run focused checks without loading real models:

```bash
PYTHONPATH=.:src aic/bin/pytest -q \
  tests/test_llm_api.py \
  tests/unit/llm/test_inference_gateway.py

python -m compileall -q llm
```

The local backend and UI workflow are documented in the
[root README](../README.md).
