# MLeCDanBGold — Multimodal Video Retrieval for VBS

MLeCDanBGold is an interactive multimodal video retrieval and reasoning system built for the **Video Browser Showdown (VBS)**. It enables search operators to rapidly retrieve target moments from large-scale video archives using natural language queries, image-similarity search, and metadata filters, then submit one validated answer directly to the competition evaluation server.

---

## 1. What This Is

At VBS, competing teams search across massive video archives under strict time limits. MLeCDanBGold addresses this challenge through a multi-stage retrieval engine and a direct, participant-private DRES submission flow:

- **Multimodal Retrieval**: Indexes visual keyframes and multimodal evidence using dense visual embeddings (SigLIP2), multilingual text embeddings (BGE-M3), and reciprocal rank fusion (RRF) across specialist modalities (Captions, OCR, detected Objects, and timestamped ASR transcripts).
- **VBS Competition Tasks**:
  - **KIS (Known-Item Search)**: Locate a single target scene and submit one temporal answer using its canonical `video_id` and `timestamp_ms`.
  - **AVS (Ad-hoc Video Search)**: Retrieve moments matching a semantic description and submit each selected temporal answer separately.
  - **VQA (Video Question Answering)**: Answer free-form or multi-choice natural-language questions about video content.
- **DRES v2 Integration**: Connects mapped participants to the official Distributed Retrieval Evaluation Server (DRES), reads the active task scope, submits one answer, and streams operator interaction logs.
- **Private DRES Submission**: Each connected participant edits one temporary KIS/AVS temporal answer or VQA text answer and sends it through that participant’s backend-held DRES session.

---

## 2. System Architecture

```text
       ┌────────────────────────────────────────────────────────┐
       │                React Frontend (:3000)                  │
       │  - Search bar (text & drag-and-drop image query)       │
       │  - Integrated metadata filters (Folder, Video, OCR...) │
       │  - Ephemeral one-answer submission dialog              │
       │  - Private VBS participant session controls            │
       └──────────────────────────┬─────────────────────────────┘
                                  │ HTTP
                                  ▼
       ┌────────────────────────────────────────────────────────┐
       │               FastAPI Backend (:8000)                  │
       │  - Canonical FrameStore (Parquet metadata)             │
       │  - Hybrid Retrieval Engine (SigLIP2, BGE-M3, BM25)     │
       │  - SQLite query history                                │
       │  - DRES Client (sessions, logging, submissions)        │
       └──────────────┬───────────────────────────┬─────────────┘
                      │                           │
                      ▼                           ▼
       ┌───────────────────────────┐ ┌──────────────────────────┐
       │     Inference Service     │ │  DRES Evaluation Server  │
       │  (Local or GPU remote)    │ │  (Official or Mock :8080)│
       └───────────────────────────┘ └──────────────────────────┘
```

### Core Invariants

Canonical frame identity must remain intact across all retrieval and ranking stages:

- `video_id`: Source video identifier matching official DRES media names.
- `frame_id`: Internal unique frame key (`<video_id>_<order>`) used for joins and asset lookups.
- `frame_idx`: Official competition frame coordinate (integer from official mapping); temporal DRES answers use canonical `video_id` and `timestamp_ms`.
- `timestamp_ms`: Canonical millisecond timestamp.

---

## 3. Getting Started

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** and **npm**
- (Optional) Docker for running the local DRES mock server

### 1. Backend Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install backend dependencies
pip install --upgrade pip
pip install -e ".[embedding,reranking,dev]"

# Configure environment
cp .env.example .env
# Configure inference endpoints (LLM & Embeddings) in .env:
#   Self-hosted example:
#     HCMAI_LLM_BASE_URL=https://api.iamphuckhang.dev/v1
#     HCMAI_LLM_MODEL=Qwen/Qwen3-4B
#     HCMAI_EMBEDDING_BASE_URL=https://api.iamphuckhang.dev/v1
#     HCMAI_EMBEDDING_MODEL=google/siglip2-base-patch16-224
#   Third-party example:
#     HCMAI_LLM_BASE_URL=https://api.openai.com/v1
#     HCMAI_LLM_API_KEY=sk-...
#     HCMAI_LLM_MODEL=gpt-4o-mini
#     HCMAI_EMBEDDING_BASE_URL=https://api.openai.com/v1
#     HCMAI_EMBEDDING_API_KEY=sk-...
#     HCMAI_EMBEDDING_MODEL=text-embedding-3-small

# Start the FastAPI server
uvicorn hcmai.app:app --host 127.0.0.1 --port 8000 --reload
```

Verify backend health:

```bash
curl -s http://127.0.0.1:8000/health
```

### 2. Frontend Setup

In a separate terminal:

```bash
cd frontend

# Install dependencies
npm ci

# Configure environment (points to http://127.0.0.1:8000 by default)
cp .env.example .env

# Start development server
npm start
```

Open `http://localhost:3000` in your browser.

### VBS participant setup and rehearsal

Set `HCMAI_DRES_USERS_JSON` on the backend to map each VBS participant ID to that participant’s DRES credentials. In each browser, connect its own participant ID; the backend keeps independent private sessions and never sends a DRES session ID to the browser. Set `HCMAI_DRES_MEDIA_ID_PREFIX_TO_STRIP` only when DRES media names require an exact leading-prefix removal; the backend maps canonical `video_id` to `mediaItemName`.

Before a live round, test DRES submissions against the mock server. KIS and AVS submit one temporal answer per action using the selected `timestamp_ms`; VQA submits one text answer. Confirm the active task scope before sending, and freeze the deployed task configuration after the rehearsal.

Before upgrading an existing deployment, create a backup of `runtime/workspace.sqlite3` if its retired answer data may still be needed. Schema v3 removes the old answer workspace, candidate, submission-attempt, and submission-file storage while preserving `query_history` and its viewed-frame values. Submission-file storage is retired; the database remains for query history.

### 3. (Optional) DRES Mock Server for Rehearsals

For local end-to-end rehearsal without the live competition server:

```bash
cd dres-mock-server
pip install -e .
python -m dres_mock_server.app
```

Or via Docker:

```bash
docker build -t dres-mock dres-mock-server/
docker run -p 8080:8080 dres-mock
```
---

## 4. Repository Structure

```text
MLeCDanBGold_2026/
├── src/MLeCDanBGold/               # Core Python package & FastAPI application
│   ├── api/                 # HTTP routers (search, keyframes, VBS, query history)
│   ├── common/              # Shared data schemas, configuration, and contracts
│   ├── data/                # Ingestion, FrameStore parquet reader, artifact schemas
│   ├── retrieval/           # Visual and context embeddings, FAISS indexes, RRF fusion
│   ├── temporal/            # Temporal alignment & sequential event planning (TRAKE)
│   ├── vbs/                 # DRES API client, session management, and submissions
│   └── app.py               # FastAPI application factory
├── frontend/                # React application
│   ├── src/features/search/ # Search workspace (text and image queries, filters)
│   ├── src/features/submission/ # Ephemeral direct DRES answer dialog
│   ├── src/features/vbs/    # VBS participant session controls
│   └── src/api/             # REST API transport clients
├── dres-mock-server/        # Standalone mock server implementing DRES v2 API
├── configs/                 # Search, ingestion, and model configuration YAMLs
├── scripts/                 # Index building, evaluation, and data preparation CLI tools
├── tests/                   # Backend unit and integration tests
└── README.md
```

---

## 5. Key Design Decisions

1. **Decoupled Serving and Inference**: Model inference (SigLIP2 visual encoding, BGE-M3 text encoding) is separated via an HTTP inference gateway. The local FastAPI backend and React frontend can operate on standard laptops while GPU inference scales independently.
2. **Deterministic FrameContext**: Specialist evidence (OCR, Captions, Objects) is stored with full provenance. Rather than destructively flattening them into a generic blob, they are indexed separately and joined into a deterministic `FrameContext` view for hybrid retrieval.
3. **Traceable Submissions & Security**: Official DRES credentials remain on the backend (`HCMAI_DRES_USERS_JSON`). Operators connect using their participant ID, and the backend forwards submissions through that participant’s private DRES session while reporting retrieval-log status (`X-DRES-Log-Status`).
4. **Private Submission**: Answer drafts and DRES outcomes stay temporary in the browser; each one-answer request is scoped to the active task and uses only the submitting participant’s private backend session.

---

## 6. VBS 2027 Competition Runbook

### Migration archive notice

Before migration, stop the backend and create a backup of `runtime/workspace.sqlite3` in a timestamped archive in the operator’s protected backup location. The legacy submission-file workflow and its stored data are retired by this migration; keep the archive in case an older deployment needs to be restored. Do not copy the database into the frontend or publish it with a public build.

### Prepare and rehearse

1. On the backend only, set `HCMAI_DRES_BASE_URL` to the organizer’s test DRES endpoint and configure `HCMAI_DRES_USERS_JSON` as the mapping from each VBS `user_id` to its DRES username/password. Keep this JSON in a protected backend environment file or secret store. A browser submits only its VBS user ID; never put credentials or a DRES session in frontend configuration.
2. Verify every HCMAI `video_id` against the organizer’s exact DRES `mediaItemName`. Confirm `HCMAI_DRES_MEDIA_ID_PREFIX_TO_STRIP` against known media before setting it; do not infer the mapping from display labels or `frame_idx`. Save and review the verified mapping before rehearsal.
3. Start the backend against test DRES. In each browser, enter the assigned VBS user ID and wait for the connected state before opening the shared answer workspace. Confirm separate browsers converge on the same candidates. The official current-task response has no task ID: HCMAI derives a deterministic `task_scope_key` from its evaluation and task-template fields for internal workspace isolation only. DRES submissions send the freshly resolved `taskName`; the internal key is never sent upstream. Existing workspace DB v1 rows migrate as `legacy-unverified` and require an explicit clear-and-switch before use with a current official task.
4. Against test DRES, submit one KIS FRAME captured from live playback and one added from a FrameCard. Confirm the reviewed `timestamp_ms` is exact and DRES receives a point answer with `start == end`. Submit one VQA TEXT answer and confirm the answer contains text only. Add several AVS frames and confirm one ordered batch reaches DRES in a single request. DRES HTTP 200/202 success must include a verdict (`CORRECT`, `WRONG`, `INDETERMINATE`, or `UNDECIDABLE`); `accepted` means DRES processed the submission, not that the answer was correct.
5. With each assigned user connected to test DRES, run text search, image search, and Filter. Confirm the header and indicator report `Log sent`; make one controlled logging-failure check and confirm retrieval still displays results with `Last log failed`.
6. After the rehearsal passes, freeze the verified media mapping, evaluation settings, indexes, backend/frontend revisions, and non-secret environment configuration. Keep credentials in the backend secret store; each successful search is logged with the connected participant's DRES session.

The DRES client current-task endpoint cannot distinguish consecutive task instances whose `name`, `taskGroup`, `taskType`, and `duration` are identical. The internal key therefore isolates observable task templates, not hidden task instance identity; confirm that staging exposes a task-name change or another supported client-visible boundary before relying on automatic scope detection.

---

## Contributors

<a href="https://github.com/khang1108/MLeCDanBGold/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=khang1108/MLeCDanBGold" alt="Contributors" />
</a>
