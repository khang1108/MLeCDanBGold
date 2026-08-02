# HCMAI 2026 Frame Retrieval

HCMAI is a research-oriented video-frame retrieval project for the Ho Chi
Minh City AI Challenge 2026. The target interaction is a Vietnamese or
English natural-language query that returns the exact matching frame's
official `video_id` and `frame_idx`.

The project is a small, measurable hackathon baseline. The current code
includes canonical frame preparation, shared contracts, visual retrieval,
search orchestration, and an API boundary.

## Current status

Implemented foundations:

- Pydantic 2 schemas for frames, enrichment, retrieval, search, evaluation,
  enums, and conversational feedback.
- A deterministic `official mapping + keyframes → frames.parquet` builder and
  an in-memory `FrameStore`.
- `SearchEngine` orchestration with one selected competition configuration,
  optional reranking, response materialization, and latency fields.
- Visual embedding, FAISS index, and dense-retrieval foundations.
- Caption and OCR enrichment pipelines, Caption/OCR/ASR evidence stores, plus
  caption indexing and optional visual-caption RRF retrieval.
- Resumable multilingual transcription with speaker-labelled, per-video
  Parquet artifacts.
- A FastAPI application and the existing Node.js frontend.
- Utility helpers for YAML/JSON/Parquet I/O, image loading, timing, and
  logging.
- Lightweight schema tests and smoke-testable modules.

Still to implement:

- Add OCR/ASR retrievers and benchmark multimodal fusion weights.
- Benchmark and select the highest-scoring caption text encoder that satisfies
  the competition latency budget.
- Reproducible offline evaluation runners and measured retrieval experiments.

## Data and retrieval flow

The diagram distinguishes the active visual path, reusable components that
still need online wiring, and planned retrieval stages. Every artifact is
joined through the canonical `frame_id`; final submission identifiers always
come from `frames.parquet`.

```mermaid
flowchart TB
    subgraph OFFLINE["Offline ingestion and indexing"]
        direction LR
        MAP["Official frame mappings"] --> META["data/metadata/<br/>frames.parquet"]
        KEY["Keyframe images"] --> META

        META --> VDATA["Frame images"]
        VDATA --> VENC["SigLIP2<br/>image encoder"]
        VENC --> VEMB["artifacts/embeddings/<br/>visual_embeddings.npy"]
        VEMB --> VIDX["artifacts/indexes/visual/<br/>dense.index + mapping + metadata"]

        VDATA --> CAPGEN["Caption generation<br/>florence-community/Florence-2-base-ft"]
        CAPGEN --> CAPSTORE["artifacts/enrichment/caption/<br/>frame_enrichment.parquet"]
        CAPSTORE --> CAPENC["Caption TextEncoder<br/>benchmark: SigLIP2 / E5 / BGE-M3"]
        CAPENC --> CAPEMB["artifacts/indexes/caption/<br/>caption_embeddings.npy"]
        CAPEMB --> CAPIDX["artifacts/indexes/caption/<br/>dense.index + mapping + metadata"]

        VDATA --> OCRGEN["OCR generation<br/>florence-community/Florence-2-base-ft"]
        OCRGEN --> OCRSTORE["artifacts/enrichment/ocr/<br/>frame_enrichment.parquet"]

        VIDEO["Video audio"] --> ASRGEN["ASR pipeline"]
        ASRGEN --> ASRSTORE["artifacts/enrichment/asr/<br/>frame_enrichment.parquet"]
    end

    subgraph ONLINE["Online query path"]
        direction LR
        UI["React UI"] --> API["FastAPI"]
        API --> RESOLVE["KISC resolve<br/>standalone query"]

        RESOLVE --> VQENC["SigLIP2<br/>text query encoder"]
        VQENC --> VRET["DenseRetriever<br/>source: visual"]
        VIDX --> VRET

        RESOLVE --> CQENC["Caption query encoder<br/>same checkpoint as caption index"]
        CQENC --> CRET["CaptionRetriever<br/>enabled for a compatible index"]
        CAPIDX --> CRET

        RESOLVE -.-> ORET["OCR text retriever<br/>planned"]
        OCRSTORE -.-> ORET
        RESOLVE -.-> ARET["ASR text retriever<br/>planned"]
        ASRSTORE -.-> ARET

        VRET --> FUSION["RRFFusionRetriever<br/>union by frame_id"]
        CRET --> FUSION
        ORET -.-> FUSION
        ARET -.-> FUSION
        FUSION --> RERANK["Optional multimodal<br/>reranking"]
        RERANK --> MATERIALIZE["SearchEngine materialization<br/>exact video_id + frame_idx"]
        META --> MATERIALIZE

        CAPSTORE --> EVIDENCE["Caption / OCR / ASR<br/>evidence lookup"]
        OCRSTORE --> EVIDENCE
        ASRSTORE --> EVIDENCE
        EVIDENCE --> MATERIALIZE

        MATERIALIZE --> RESPONSE["Search response"]
        RESPONSE --> UI
    end

    classDef active fill:#dcfce7,stroke:#15803d,color:#14532d;
    classDef component fill:#fef3c7,stroke:#d97706,color:#78350f;
    classDef planned fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:5 5;

    style OFFLINE fill:#f8fafc,stroke:#64748b,stroke-width:2px,color:#0f172a;
    style ONLINE fill:#f8fafc,stroke:#64748b,stroke-width:2px,color:#0f172a;
    linkStyle default stroke:#64748b,stroke-width:1.5px;

    class MAP,KEY,META,VDATA,VENC,VEMB,VIDX,CAPGEN,CAPSTORE,OCRGEN,OCRSTORE,ASRSTORE,UI,API,RESOLVE,VQENC,VRET,CQENC,CRET,FUSION,EVIDENCE,RERANK,MATERIALIZE,RESPONSE active;
    class CAPENC,CAPEMB,CAPIDX component;
    class VIDEO,ASRGEN,ORET,ARET planned;
```

Green nodes are active when their compatible artifacts are available. Amber
nodes have artifact contracts or reusable code but still require a benchmark
decision. Dashed gray nodes are planned. The caption encoder is deliberately
marked as a benchmark choice: `SigLIP2`, `multilingual-e5-small`, and `BGE-M3`
must be measured before selecting the competition checkpoint. Until a
separate text encoder is configured, caption fusion only activates when its
index matches the current query-encoder checkpoint and dataset version.

The design keeps expensive offline work separate from online search. Model
checkpoints and candidate counts belong in
configuration, while frame identifiers and API shapes belong in the shared
schemas.

## Repository structure

```text
frontend/                         Existing Node.js UI
scripts/                          Root-level data and experiment CLIs
src/hcmai/
├── app.py                        FastAPI boundary
├── search.py                     Search orchestration
├── kisc.py                       Conversational state
├── data/                         Canonical builder and FrameStore
├── transcripts/                  ASR pipeline and TranscriptStore
├── embedding/                    Visual embedding pipeline
├── retriever/                    Dense, caption, fusion, and evaluation packages
├── reranking/                    Multimodal pipeline and model backends
└── common/
    ├── config.py                 Shared settings scaffolding
    ├── schemas/                  Pydantic contracts
    │   └── README.md              Schema documentation
    └── utils/                    Generic helpers
        └── README.md              Utility documentation
configs/                          Experiment and search configuration
data/                             Local corpus and metadata
artifacts/                        Generated embeddings and indexes
runs/                             Evaluation outputs
tests/                            Contract tests and smoke tests
```

Install the project and its declared dependencies:

```bash
aic/bin/python -m pip install -e ".[embedding,reranking,dev]"
```

## Host the AI models on a temporary GPU VM

The frontend, FastAPI search backend, dataset, FAISS index, and keyframes stay
on the local machine. A temporary Thunder Compute VM hosts only the embedding,
reranking, and conversation models:

```text
React UI (localhost:3000)
        |
        v
Local FastAPI + artifacts (localhost:8000)
        |
        v
Cloudflare Access + Tunnel (api.iamphuckhang.dev)
        |
        v
Temporary GPU model API (localhost:8100 on the VM)
```

`configs/baseline.yaml` owns dataset, artifact, search, fusion, API, and
inference-connection settings. `llm/config.yaml` is the single authority for
visual embedding, caption embedding, reranking, and conversation checkpoints.

This keeps the roughly 100 GB retrieval artifacts off the VM. Deleting the VM
therefore loses only the installed environment and downloaded model cache, not
the local search data.

### 1. One-time local and Cloudflare setup

Install and authenticate the
[Thunder Compute CLI](https://www.thundercompute.com/docs/cli/quickstart):

```bash
curl -fsSL https://raw.githubusercontent.com/Thunder-Compute/thunder-cli/main/scripts/install.sh | bash
tnr login
```

In Cloudflare, create one remotely managed Tunnel and configure:

- Published hostname: `api.iamphuckhang.dev`
- Service URL: `http://localhost:8100`
- Cloudflare Access policy: Service Auth

The Tunnel, DNS record, and Access policy are persistent; do not recreate them
for every VM. Keep the Tunnel token and Cloudflare Access credentials private.
The file `llm/deploy_cloudflared_private.sh` is intentionally ignored by Git.
Configure its repository, branch, Tunnel token, and any model overrides once
on the local machine. Never commit or paste this private script into an issue
or log.

Before creating a VM, commit and push the code and configuration that the
bootstrap must clone. An uncommitted local change is not available to the VM.

### 2. Create a throwaway VM

Run the interactive creator:

```bash
tnr create
```

For the current BF16 model configuration, select:

- Prototyping mode for short development sessions
- One L40 or A6000 GPU with 48 GB VRAM
- The base Ubuntu/PyTorch template
- At least 80 GB of primary disk; 100 GB gives safer room for packages and
  Hugging Face model caches

Wait until the instance is `RUNNING`, then copy its numeric ID from:

```bash
tnr status --no-wait
```

Use that ID in place of `<instance-id>` below. It is often `0`, but must not be
assumed.

### 3. Upload and run the all-in-one bootstrap

From the repository root on the local machine:

```bash
tnr scp llm/deploy_cloudflared_private.sh <instance-id>:/home/ubuntu/
tnr connect <instance-id>
```

Then run the model set needed for the current session. For caption generation
and CaptionStore indexing:

```bash
sudo bash /home/ubuntu/deploy_cloudflared_private.sh \
  --caption true \
  --caption-embedding true
```

The private bootstrap also accepts `--visual-embedding true`,
`--reranker true`, and `--conversation true`. All five model flags default to
`false`, and disabled models are not loaded into VRAM. Visual embedding uses
SigLIP2; caption embedding uses BGE-M3.

The script clones the configured repository into `/opt/hcmai/repo`, installs
the Python environment and a Python 3.12-compatible Supervisor, downloads the
configured checkpoints, starts the model API, starts `cloudflared`, and checks
both processes. Git runs only as the `hcmai` service user, so no Git login,
global ownership exception, or author configuration is required for the public
repository. Each bootstrap run fetches the configured branch and resets this
deployment checkout to its newest commit. It does not download the HCMAI
keyframes, embeddings, mappings, or FAISS index.

The first launch can take several minutes because it downloads the model
checkpoints. The default conversation checkpoint comes from `llm/config.yaml`;
leave `HCMAI_CONVERSATION_MODEL` empty in the private script unless an explicit
override is required.

### 5. Finish the session and stop GPU billing

Thunder Compute has no native stopped-instance state. If the VM is disposable,
exit its shell and delete it after finishing:

```bash
exit
tnr delete <instance-id>
```

Deletion is permanent and stops GPU billing. A later VM can be created and
bootstrapped with the same private script. To avoid downloading the model cache
again, optionally create a snapshot first and wait until it is `READY`:

```bash
tnr snapshot create --instance-id <instance-id> --name hcmai-model-host
tnr snapshot list
```

## Data preparation

If the dataset arrived as zip archives under `data/`, extract them first:

```bash
aic/bin/python scripts/extract_zips.py --data-dir data
```

Build the single canonical metadata artifact from the official mapping and
provided keyframe images:

```bash
PYTHONPATH=src aic/bin/python scripts/prepare_data.py \
  --dataset-root data \
  --output data/metadata/frames.parquet
```

## Transcript preparation

Install the optional ASR dependencies, then smoke-test two videos:

```bash
pip install -e '.[transcripts]'
export HF_TOKEN="hf_..."
PYTHONPATH=src python scripts/prepare_transcripts.py \
  --videos-root /path/to/videos \
  --output artifacts/transcripts \
  --limit 2
```

The command writes one speaker-labelled transcript Parquet per video. Pass
`--no-resume` to reprocess existing outputs. See the
[transcript pipeline guide](src/hcmai/transcripts/README.md) for artifact
schemas, diarization behavior, and configuration.

## Initialize the local FastAPI backend

The FastAPI backend runs on the local data machine, not on the temporary GPU
VM. It loads `frames.parquet` and the FAISS index from local storage, serves
frame images to the React UI, manages KISC sessions, and sends only model
inference requests to `api.iamphuckhang.dev`.

### 1. Create the Python environment

Run these commands from the repository root. Python 3.11 or newer is required:

```bash
python3 --version
python3 -m venv aic
aic/bin/python -m pip install --upgrade pip
aic/bin/python -m pip install -e ".[embedding,reranking,dev]"
```

The virtual environment is created only once. Rerun the final install command
after pulling a commit that changes `pyproject.toml`.

### 2. Configure the backend environment

Create a private local environment file:

```bash
cp .env.example .env
nano .env
```

The application reads process environment variables; it does not automatically
load the root `.env` file. Export it in every new backend terminal:

```bash
set -a
source .env
set +a
```

The default artifact paths come from `configs/baseline.yaml`:

```text
data/metadata/frames.parquet
artifacts/indexes/visual/
├── dense.index
├── frame_mapping.parquet
└── metadata.json
```
### 3. Start FastAPI

Keep the GPU VM and Cloudflare Tunnel running, then launch the local backend:

```bash
PYTHONPATH=src aic/bin/python -m uvicorn hcmai.app:app \
  --host 127.0.0.1 --port 8000 --reload
```

### 4. Verify backend readiness

From another local terminal:

```bash
curl -sS http://127.0.0.1:8000/health | jq
```

A search-ready backend reports:

```json
{
  "status": "ok",
  "ready": true,
  "frame_store_loaded": true,
  "retriever_loaded": true
}
```

The React app creates a server-side KISC session on launch; it no longer
contains mock frames or client-generated conversation IDs. Configure a
different backend in `frontend/.env` when needed:

```bash
cp frontend/.env.example frontend/.env
cd frontend
npm start
```
