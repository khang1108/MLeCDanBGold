# `hcmai` Package Reference

The `hcmai` package provides the core video-frame retrieval pipeline for the
Ho Chi Minh City AI Challenge 2026. It includes canonical frame preparation,
candidate retrieval, search orchestration, conversational state, and HTTP API
endpoints behind shared Pydantic contracts.

---

## Package Layout

```text
src/hcmai/
├── app.py          # FastAPI application server and REST endpoints
├── kisc.py         # KISC conversational session & feedback state manager
├── search.py       # High-level SearchEngine orchestration
├── common/         # Shared schemas, configuration, and generic utilities
│   ├── config.py   # Global configuration settings & Pydantic settings
│   ├── schemas/    # Pydantic 2 data contracts (SearchRequest, FrameRecord, etc.)
│   └── utils/      # Generic I/O, image loading, timing, and logging helpers
├── data/           # Canonical Parquet builder and FrameStore
├── embedding/      # Image embedding generation pipeline
├── retriever/      # FAISS visual index, DenseEncoder, DenseRetriever, benchmarking
└── scripts/        # Internal pipeline build scripts
```

---

## Core Modules & Primary Contracts

### 1. HTTP API Server (`hcmai.app`)
- **FastAPI Application**: Exposes REST endpoints for the Node.js frontend.
- **Key Endpoints**:
  - `GET /health`: Health status & loaded dataset frame count.
  - `POST /api/v1/search`: Frame search (supports single queries and KISC turns).
  - `POST /api/v1/session`: Create a new KISC conversation session.
  - `POST /api/v1/feedback`: Update accepted/rejected human frame feedback.
  - `GET /api/v1/frames/{frame_id}`: Fetch single frame metadata.
  - `GET /api/v1/frames/{frame_id}/neighbors`: Temporal $\pm N$ neighbor frame expansion.
  - `POST /api/v1/submit`: Format frame ID into official BTC submission code (`video_id,frame_idx`).

### 2. KISC State Manager (`hcmai.kisc`)
- **`KiscSessionManager`**: Manages stateful `ConversationSession` instances in memory, accumulates human feedback (`accepted_frame_ids` and `rejected_frame_ids`), filters out rejected candidates from search results, and formats competition submission outputs.

### 3. Search Orchestrator (`hcmai.search`)
- **`SearchEngine`**: Orchestrates `candidate_retrieval`, optional `reranking`, and response `materialization` into `SearchResponse` objects with latency tracking.
- Supports `accurate` and `fast` search profiles.

### 4. Data Pipeline & Store (`hcmai.data`)
- **`FrameStore`**: In-memory metadata store backed by `frames.parquet`.
  Supports lookup, deterministic iteration, submission-pair membership, and
  temporal neighbors.
- **Data Preparation**: `prepare_frames()` joins official mappings to
  keyframe images and writes one validated canonical `frames.parquet`.
- `frame_idx` always comes from the official mapping; consumers must not parse
  `frame_id` or infer it from time/FPS.

### 5. Dense Retriever & Index (`hcmai.retriever`)
- **`DenseRetriever`**: Pairs a text query encoder (`DenseEncoder`) with a FAISS index (`VisualIndex`) to execute vector similarity searches over keyframe embeddings.

---

## Quick Usage Examples

### Running the API Server
```bash
uv run uvicorn hcmai.app:app --host 127.0.0.1 --port 8000 --reload
```

### Python Search Engine Orchestration
```python
from hcmai.data import FrameStore
from hcmai.search import SearchEngine
from hcmai.common.schemas import SearchRequest

# Load metadata store and initialize engine
store = FrameStore.load("data/metadata/frames.parquet")
engine = SearchEngine(frame_store=store, retriever=dense_retriever)

# Execute search
request = SearchRequest(query="một người đang đi bộ", top_k=10)
response = engine.search(request)

for result in response.results:
    print(f"Rank {result.rank}: {result.frame_id} (score: {result.scores.final:.4f})")
```
