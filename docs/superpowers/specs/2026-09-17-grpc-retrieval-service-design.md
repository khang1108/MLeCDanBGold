# Design Specification: Standalone gRPC Retrieval Service

**Date:** 2026-09-17
**Status:** Approved
**Topic:** Isolate immutable retrieval artifacts from the reloadable FastAPI backend

---

## 1. Context and Goal

The current FastAPI application constructs `SearchService` during its lifespan.
That startup path loads the canonical corpus, FAISS indexes, Context and ASR
indexes, BM25 evidence, query encoders, image encoders, and temporal evidence.
Consequently, every Uvicorn reload caused by unrelated backend work reloads the
large retrieval artifacts.

Retrieval logic and artifacts are now stable, while API, KIS, event-trail, VBS,
and frontend code remain under active development. The runtime must therefore
separate these lifecycles without changing retrieval, fusion, or alignment
semantics.

The approved direction is a private gRPC retrieval process on
`127.0.0.1:8002`. The public FastAPI backend remains on port `8000` and never
loads retrieval indexes. Restarting or reloading the FastAPI process must leave
the retrieval process and its resident artifacts untouched.

This is an operational architecture change, not a retrieval-quality change. It
does not introduce a new scoring algorithm, claim an accuracy improvement, or
require scholarly retrieval research.

## 2. Scope

### 2.1 In scope

- A single-process gRPC server that loads retrieval artifacts once.
- A typed protobuf contract for readiness, KIS temporal search, TRAKE temporal
  search, selected-video scoring, and direct image search.
- A backend gRPC client and adapters used by existing orchestration workflows.
- Graceful FastAPI startup when the retrieval process is unavailable.
- Automatic use of the retrieval process after it becomes available, without a
  FastAPI restart.
- Preservation of KIS, TRAKE, image search, event-trail, exploration, filtering,
  DRES logging, and canonical frame identity behavior.
- Runtime documentation and environment configuration for both processes.

### 2.2 Out of scope

- Local retrieval fallback inside the FastAPI process.
- Multiple retrieval workers or distributed retrieval.
- Public exposure, TLS, authentication, or service discovery for the gRPC port.
- Changes to Dense, BM25, adaptive fusion, ASR projection, image scoring, or DP
  alignment algorithms.
- Moving KIS intent resolution, translation, DRES, workspace state, video
  serving, or public HTTP routes into the retrieval process.
- Rebuilding or regenerating offline artifacts during serving.

## 3. Process Architecture and Ownership

```text
Browser / frontend
       |
       | public HTTP
       v
FastAPI backend :8000
  - public API contracts and routers
  - KIS intent resolution and event translation
  - canonical Corpus for lookup, filtering, and materialization
  - DRES, workspace, videos, sessions, event-trail state
  - selected-video constraint decoding over bounded snapshots
       |
       | private gRPC
       v
Retrieval server :8002
  - visual, Context, ASR-segment, and BM25 indexes
  - text and image query encoders
  - multimodal evidence scoring and fusion
  - initial temporal DP alignment
  - direct image nearest-neighbor search
```

The retrieval server runs as exactly one process. Multiple gRPC threads may
share the loaded runtime, but starting multiple server processes is prohibited
because every process would load another complete copy of the indexes.

The FastAPI backend continues to load `Corpus`. Corpus remains necessary for
frame lookup, literal filtering, metadata materialization, evidence display,
video navigation, and canonical identity validation. The heavy retrieval
indexes and encoders are loaded only by the retrieval server.

The retrieval server may also load Corpus where the existing retrieval
constructors require it. This duplication is accepted because the goal is to
isolate expensive retrieval artifacts from FastAPI reloads, not to make the two
processes share Python objects.

## 4. Runtime Composition

### 4.1 Retrieval server

The retrieval process has a dedicated composition root. It reuses the existing
loaders and algorithms rather than copying retrieval logic:

1. Load repository environment and application/model configuration.
2. Load canonical Corpus required by existing retrieval construction.
3. Load visual, Context, ASR-segment, and BM25 indexes.
4. Construct existing query encoders, `TemporalEvidenceScorer`,
   `TemporalSearchService`, `ImageQueryTemporalScorer`, and direct image search
   dependencies.
5. Publish a stable `scoring_revision` generated once per successful server
   runtime.
6. Mark the standard gRPC health service as `SERVING` only after required
   capabilities are ready.

Artifact failures do not terminate the gRPC listener. The process logs the
startup diagnostics and remains `NOT_SERVING`; retrieval RPCs return
`UNAVAILABLE`. This retains the current diagnostic style and allows operators
to inspect readiness without repeatedly respawning a failing process.

The supported command is:

```bash
aic/bin/python -m hcmai.retrieval_service.server \
  --host 127.0.0.1 \
  --port 8002
```

The server does not run under Uvicorn and does not have a reload mode.

### 4.2 FastAPI backend

FastAPI startup constructs one long-lived gRPC channel and retrieval client
using `HCMAI_RETRIEVAL_TARGET`, defaulting to `127.0.0.1:8002`. It performs one
bounded readiness probe and records a clear startup warning if the target is
unavailable. Failure of this probe never prevents FastAPI startup.

The backend does not call local retrieval/index loaders under any configuration.
There is no fallback mode. Each search request uses the same channel; gRPC
reconnect behavior permits a retrieval process that starts later or restarts to
serve subsequent requests without restarting FastAPI.

The backend closes its channel during lifespan shutdown.

## 5. Protobuf Boundary

The service is versioned as `hcmai.retrieval.v1.RetrievalService`. Generated
Python modules are committed to the repository. Runtime environments require
`grpcio` and `protobuf`; code generation is a development step and is not run
at server startup.

All application RPCs are unary. Streaming is intentionally excluded because
each current operation is a bounded request/response and streaming would add
state and failure modes without improving the required workflow.

### 5.1 RPC methods

```protobuf
service RetrievalService {
  rpc GetCapabilities(GetCapabilitiesRequest) returns (GetCapabilitiesResponse);
  rpc SearchPlan(SearchPlanRequest) returns (TemporalSearchArtifact);
  rpc SearchEvents(SearchEventsRequest) returns (TemporalSearchResult);
  rpc ScoreVideo(ScoreVideoRequest) returns (SelectedVideoScore);
  rpc SearchImage(SearchImageRequest) returns (ImageSearchCandidates);
}
```

The process also registers the standard gRPC health service. Health reports
process/runtime serving state; `GetCapabilities` returns retrieval-specific
modalities, diagnostics, limits, and `scoring_revision`.

### 5.2 Canonical identity messages

Every aligned path and image candidate carries explicit canonical identity:

```text
video_id
frame_id / frame_ids
frame_idx / frame_idxs
timestamp_ms / timestamps_ms
```

Array offsets are never treated as frame identity. The backend validates all
returned identities against its Corpus before constructing public responses.
Mismatches are protocol failures and must not be silently repaired.

### 5.3 Retrieval plans and query images

`SearchPlanRequest` preserves one ordered row for every KIS event, including:

- server-owned `event_id` in exact `E1..En` order;
- canonical, Dense, and BM25 text views independently;
- content-addressed image `asset_id` values;
- `use_dense`, `use_bm25`, and `top_k`.

Both processes use the same configured KIS query-image directory. The backend
canonicalizes image references before sending the plan. The retrieval process
opens the immutable asset by ID and performs existing image scoring. Missing or
invalid assets fail the request; they are not interpreted as absent evidence.

`SearchImageRequest` carries the direct-upload image bytes and media type in the
protobuf message. It does not create a shared temporary file.

### 5.4 Score-array encoding

Full-corpus score matrices never cross the process boundary.

`SearchPlan` returns score arrays only for the distinct videos represented in
the returned top-k paths. `ScoreVideo` returns exactly one requested video's
score snapshot. Each selected-video payload contains:

- canonical video ID;
- frame IDs as strings;
- `frame_idx` and timestamps as packed integer bytes;
- the event-by-frame score matrix as little-endian float32 bytes;
- explicit event count and frame count;
- an encoding version.

The receiver validates encoding version, byte lengths, shape, event count,
strict field alignment, and canonical identities before constructing NumPy
arrays. Decoded snapshot arrays are immutable in the backend.

Using binary numeric buffers avoids JSON expansion and protobuf object creation
for every numeric cell. Strings remain explicit because canonical frame IDs
must not be inferred from numeric positions.

### 5.5 Message limits

Both client and server set the same configurable maximum send and receive
message size. The default is 64 MiB and the value must be positive and bounded
by configuration validation. Application-level checks continue to enforce
query image byte/pixel limits and maximum temporal event count before expensive
work.

## 6. Request Flows

### 6.1 KIS

1. FastAPI resolves or updates `KISIntent` and translates Dense text as today.
2. Backend builds the immutable `KISRetrievalPlan` and canonicalizes image refs.
3. Backend calls `SearchPlan`.
4. Retrieval server scores text/image evidence using existing logic, fuses it,
   aligns paths, and returns paths plus top-video score snapshots.
5. Backend validates and materializes paths through Corpus.
6. Backend stores bounded score snapshots for event-trail and emits the existing
   public KIS response and DRES log.

### 6.2 TRAKE

1. Backend validates the public TRAKE request.
2. Backend calls `SearchEvents` with original, Dense, and BM25 event views.
3. Retrieval server runs existing scoring and alignment.
4. Backend validates identity and builds the existing public TRAKE response.

### 6.3 Event-trail and exploration

An initial KIS response already includes snapshots for returned videos, so
event-trail constraint decoding continues locally without another retrieval
call. The backend keeps only the bounded selected-video matrices it already
needs.

Opening an exploration branch calls `ScoreVideo` for the selected video when no
usable snapshot is already available. The remote server may compute the current
full-corpus scores internally, but only the requested video's data crosses the
boundary. Subsequent confirm/reject/window/undo operations apply masks and pure
DP decoding locally against the immutable selected-video snapshot.

`scoring_revision` belongs to the retrieval process. A changed revision makes
old exploration bindings stale. In-memory FastAPI event-trail snapshots remain
process-local and naturally disappear on a backend reload.

### 6.4 Direct image search

The existing FastAPI route retains public validation and DRES behavior. It sends
validated image bytes to `SearchImage`; the retrieval server decodes, embeds,
and searches the visual index. The backend validates returned candidates and
materializes the unchanged public `ImageSearchResponse`.

### 6.5 Literal filter and frame/video routes

Literal metadata filtering remains local because it uses Corpus evidence rather
than the heavy vector indexes. Frame, video, history, VBS, and asset upload/read
routes remain local and do not call gRPC.

## 7. Failure Semantics and Observability

### 7.1 Status mapping

The backend maps gRPC failures consistently:

| gRPC outcome | Public HTTP outcome |
|---|---:|
| `UNAVAILABLE` | 503 |
| `DEADLINE_EXCEEDED` | 503 |
| `INVALID_ARGUMENT` | 422 |
| `NOT_FOUND` | 404 |
| `RESOURCE_EXHAUSTED` for bounded user input | 413 |
| `INTERNAL`, `DATA_LOSS`, invalid response contract | 502 |

The client does not expose raw remote response bodies, targets containing
credentials, binary payloads, or stack traces in public error details.

### 7.2 Deadlines

Every call has an explicit deadline. Readiness uses a short dedicated deadline;
search uses `HCMAI_RETRIEVAL_TIMEOUT_SECONDS`. A deadline is cancellation, not
a local fallback trigger. Retrieval algorithms do not retry automatically
because repeating a large scoring request could double resource consumption.

### 7.3 Health and logs

FastAPI `/health` includes a `remote_retrieval` object with configured target,
reachability, readiness, capabilities, and last observed scoring revision. A
bounded gRPC health probe refreshes this state. Health probes never load models,
discover artifacts, or run retrieval.

The retrieval server logs startup stages, loaded modalities, RPC name, duration,
event count/top-k where applicable, and error category. It never logs image
bytes, numeric score buffers, secrets, or complete user queries by default.

## 8. Configuration

The following backend environment values are introduced:

```text
HCMAI_RETRIEVAL_TARGET=127.0.0.1:8002
HCMAI_RETRIEVAL_TIMEOUT_SECONDS=120
HCMAI_RETRIEVAL_HEALTH_TIMEOUT_SECONDS=1
HCMAI_RETRIEVAL_MAX_MESSAGE_MIB=64
```

The retrieval server continues to use the existing artifact, embedding, model,
dataset, and KIS asset settings. Command-line `--host` and `--port` default to
`127.0.0.1` and `8002`.

The target defaults are appropriate only for same-host deployment. Binding the
gRPC server to a non-loopback address requires a future security design and is
not part of this work.

## 9. Requirements

- **REQ-001:** FastAPI startup and reload must never load FAISS, Context, ASR,
  BM25, retrieval encoders, or temporal evidence locally.
- **REQ-002:** The retrieval process must load existing retrieval algorithms and
  artifacts once and serve them from one process on the configured gRPC port.
- **REQ-003:** FastAPI must start when retrieval is unavailable, log one clear
  startup diagnostic, and keep non-retrieval routes available.
- **REQ-004:** A retrieval process that becomes available later must serve the
  next backend search without requiring a backend restart.
- **REQ-005:** KIS text, image-only, and mixed text/image plans must preserve
  event order, specialist evidence selection, fusion, alignment, and public
  response behavior.
- **REQ-006:** TRAKE must preserve original/Dense/BM25 event views and ordered
  alignment behavior.
- **REQ-007:** Direct image search must transmit validated bytes over gRPC and
  preserve its public result and DRES behavior.
- **REQ-008:** Event-trail and exploration must retain immutable selected-video
  score snapshots and reject stale retrieval revisions.
- **REQ-009:** Every remote result must preserve and validate `video_id`,
  `frame_id`, `frame_idx`, and `timestamp_ms`; array position must never become
  canonical identity.
- **REQ-010:** Full-corpus matrices must not cross gRPC; numeric selected-video
  arrays must use the versioned binary encoding with strict shape/length checks.
- **REQ-011:** Every RPC must have a deadline and consistent gRPC-to-HTTP failure
  mapping, with no local retrieval fallback or automatic scoring retry.
- **REQ-012:** Client and server must enforce matching configurable message-size
  limits and existing query input bounds.
- **REQ-013:** Standard gRPC health and retrieval capabilities must expose
  readiness without performing retrieval work.
- **REQ-014:** Literal filters and non-retrieval public routes must continue to
  work locally when retrieval is unavailable.
- **REQ-015:** The runtime documentation must show separate commands, require one
  retrieval process, omit reload for retrieval, and retain reload for FastAPI.

## 10. Testing Strategy

Implementation follows red/green TDD against the requirements above.

- Protobuf/codec tests cover binary array round-trips, malformed byte lengths,
  shape conflicts, event count, immutable arrays, and canonical identity.
- gRPC servicer tests use a fake runtime so endpoint validation and status codes
  do not require real FAISS artifacts.
- Client tests use an in-process gRPC server on an ephemeral loopback port and
  cover reconnect, deadlines, status mapping, message limits, and invalid
  responses.
- Startup composition tests monkeypatch local retrieval loaders to fail if the
  FastAPI path calls them, and verify an unavailable gRPC target does not abort
  FastAPI lifespan.
- Workflow tests cover KIS, TRAKE, direct image search, event-trail, exploration,
  filters, health, and canonical materialization through remote adapters.
- A small hand-checkable integration fixture compares existing local algorithm
  output with the gRPC adapter result for identical inputs. This validates
  transport equivalence; it does not claim retrieval-quality improvement.
- Focused tests run before the broader repository suite. No test requires the
  production indexes or a live external model provider.

## 11. Compatibility and Migration

Public HTTP request and response contracts remain unchanged. The frontend and
DRES integrations require no endpoint migration.

The operational migration is intentionally explicit:

1. Install the gRPC runtime dependencies.
2. Start retrieval server `:8002` and wait for `SERVING`.
3. Start or reload FastAPI `:8000` with the configured target.
4. Confirm `/health` reports remote retrieval ready.
5. Run KIS, TRAKE, image, filter, event-trail, and exploration smoke tests.

There is no mixed local/remote retrieval mode. Deployment rollback is a code
rollback, not an environment toggle that silently reloads indexes in FastAPI.

## 12. Knowledge and Research Record

No retrieval algorithm, model, scoring hypothesis, or measured quality result
changes in this design. `KNOWLEDGE.md` is therefore not updated. If
implementation reveals a material R&D finding, it must be recorded separately
with the appropriate SOURCE/PAPER/PROPOSED status.
