# MLeCDanBGold — Multimodal Video Retrieval for VBS

MLeCDanBGold is an interactive multimodal video retrieval and reasoning system built for the **Video Browser Showdown (VBS)**. It enables search operators to rapidly retrieve target moments from large-scale video archives using natural language queries, image-similarity search, and metadata filters, then submit validated answers directly to the competition evaluation server.

---

## 1. What This Is

At VBS, competing teams search across massive video archives under strict time limits. MLeCDanBGold addresses this challenge through a multi-stage retrieval engine paired with a real-time collaborative workspace:

- **Multimodal Retrieval**: Indexes visual keyframes and multimodal evidence using dense visual embeddings (SigLIP2), multilingual text embeddings (BGE-M3), and reciprocal rank fusion (RRF) across specialist modalities (Captions, OCR, detected Objects, and timestamped ASR transcripts).
- **VBS Competition Tasks**:
  - **KIS (Known-Item Search)**: Locate a single target scene and submit the exact `(video_id, frame_idx)` or timestamp.
  - **AVS (Ad-hoc Video Search)**: Collect and submit a ranked batch of diverse keyframes matching a semantic description.
  - **VQA (Video Question Answering)**: Answer free-form or multi-choice natural-language questions about video content.
- **DRES v2 Integration**: Connects to the official Distributed Retrieval Evaluation Server (DRES) to authenticate participants, receive task updates, submit answers, and stream operator interaction logs.
- **Collaborative Answer Workspace**: Multi-user interface synchronized in real time via WebSockets, allowing team members to coordinate, triage candidate frames, and submit answers without duplicate collisions.

---

## 2. System Architecture

```text
       ┌────────────────────────────────────────────────────────┐
       │                React Frontend (:3000)                  │
       │  - Search bar (text & drag-and-drop image query)       │
       │  - Integrated metadata filters (Folder, Video, OCR...) │
       │  - Real-time collaborative Answer Workspace            │
       │  - VBS session management & DRES submit triggers       │
       └──────────────────────────┬─────────────────────────────┘
                                  │ HTTP / WebSocket
                                  ▼
       ┌────────────────────────────────────────────────────────┐
       │               FastAPI Backend (:8000)                  │
       │  - Canonical FrameStore (Parquet metadata)             │
       │  - Hybrid Retrieval Engine (SigLIP2, BGE-M3, BM25)     │
       │  - SQLite Workspace & Reservation Store                │
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
- `frame_idx`: Official competition frame coordinate (integer from official mapping) required for submissions.
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

# Start the FastAPI server
uvicorn MLeCDanBGold.app:app --host 127.0.0.1 --port 8000 --reload
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
│   ├── api/                 # HTTP routers (search, keyframes, vbs, workspace)
│   ├── common/              # Shared data schemas, configuration, and contracts
│   ├── data/                # Ingestion, FrameStore parquet reader, artifact schemas
│   ├── retrieval/           # Visual and context embeddings, FAISS indexes, RRF fusion
│   ├── temporal/            # Temporal alignment & sequential event planning (TRAKE)
│   ├── vbs/                 # DRES API client, session management, and submissions
│   └── app.py               # FastAPI application factory
├── frontend/                # React application
│   ├── src/features/search/ # Search workspace (text and image queries, filters)
│   ├── src/features/answer-workspace/ # Live answer triage and submission panel
│   ├── src/features/vbs/    # VBS participant session controls
│   └── src/api/             # API transport clients (REST and WebSockets)
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
3. **Traceable Submissions & Security**: Official DRES credentials remain securely on the backend (`HCMAI_DRES_USERS_JSON`). Operators connect using their team user ID, and the backend signs submissions and streams required audit logs (`X-DRES-Log-Status`).
4. **Collaborative Synchronization**: Operators share an active answer board backed by SQLite and WebSockets with optimistic locking to eliminate redundant duplicate submissions during high-speed competition rounds.

---

## Contributors

<a href="https://github.com/khang1108/MLeCDanBGold/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=khang1108/MLeCDanBGold" alt="Contributors" />
</a>
