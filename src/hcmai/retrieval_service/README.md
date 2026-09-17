# Retrieval Service

Standalone long-lived gRPC service (`hcmai.retrieval_service`) hosting heavy retrieval models and vector indexes for the HCMAI multimodal video retrieval system.

## Overview

The retrieval service decouples heavy offline-generated artifacts from the FastAPI web application, allowing FastAPI to reload in milliseconds during development while keeping indexes and models resident in memory.

### Owned Capabilities

- **SigLIP2 Visual Embeddings & FAISS Index**: Dense vector search across canonical video keyframes.
- **BM25 Lexical Index**: Keyword retrieval over transcripts, OCR, and captions.
- **Multimodal Plan Scoring**: Full temporal matrix scoring for KIS retrieval plans.
- **Dynamic Programming Alignment**: Exact sequence decoding for KIS and TRAKE events.
- **Selected-Video Scoring**: Targeted single-video score retrieval for event-trail and exploration branches.
- **Direct Image Search**: SigLIP2 visual encoding and similarity retrieval for image queries.

## Architecture

```text
FastAPI Backend (:8000)
    │
    │ gRPC (retrieval.proto)
    ▼
Retrieval Service (:8002)
    ├── RetrievalServicer (gRPC RPC handler & error mapping)
    ├── RetrievalRuntime (artifact container & thread-safe dispatch)
    │     ├── TemporalSearchService (FAISS, BM25, SigLIP2)
    │     ├── ImageQueryTemporalScorer (image query scoring)
    │     ├── ImageSearchService (direct image retrieval)
    │     └── Corpus (canonical frame metadata)
    └── Standard gRPC Health Service (SERVING / NOT_SERVING)
```

## Running the Server

```bash
# Start standalone retrieval service (no Uvicorn, no --reload)
aic/bin/python -m hcmai.retrieval_service.server --host 127.0.0.1 --port 8002
```

### Operational Constraints

- **Single Process Only**: Must be run in exactly one process. Do not run under multiple Uvicorn workers or process managers that spawn parallel instances, as each instance loads multi-gigabyte FAISS and embedding indexes.
- **No Hot Reloading**: The retrieval service is intended to remain running long-term. Do not launch with `--reload`.
- **Fault-Tolerant Startup**: If index or model loading encounters errors, the server process still starts its gRPC listener in `NOT_SERVING` health state, logging startup diagnostics rather than crashing in a loop.
- **Health Checks**: Standard gRPC health checking protocol is supported on service name `hcmai.retrieval.v1.RetrievalService` and overall `""`.

## Configuration

The service and its clients are configured via environment variables:

| Variable | Default | Description |
|---|---|---|
| `HCMAI_RETRIEVAL_TARGET` | `127.0.0.1:8002` | gRPC server host and port |
| `HCMAI_RETRIEVAL_TIMEOUT_SECONDS` | `120` | RPC deadline for retrieval queries |
| `HCMAI_RETRIEVAL_HEALTH_TIMEOUT_SECONDS` | `1` | Readiness probe timeout |
| `HCMAI_RETRIEVAL_MAX_MESSAGE_MIB` | `64` | Maximum gRPC payload size in MiB |

## Proto Definitions

The interface definition is located in `src/hcmai/retrieval_service/proto/retrieval.proto`. Python bindings are pre-compiled and committed under `src/hcmai/retrieval_service/proto/`.
