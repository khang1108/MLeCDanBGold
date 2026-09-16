# Standalone gRPC Retrieval Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all heavy online retrieval artifacts and execution into one long-lived gRPC process on port 8002 so the FastAPI backend can reload without reloading indexes.

**Architecture:** A private `hcmai.retrieval.v1.RetrievalService` owns the existing local retrieval runtime, initial temporal alignment, and image retrieval. The FastAPI backend keeps Corpus, semantic intent, materialization, DRES, event-trail, and exploration state, and talks to the retrieval process through one synchronous gRPC client used from existing FastAPI threadpool boundaries.

**Tech Stack:** Python 3.11+, grpcio, Protocol Buffers, grpcio-health-checking, FastAPI, Pydantic, NumPy, pytest

**Spec:** `docs/superpowers/specs/2026-09-17-grpc-retrieval-service-design.md`

## Global Constraints

- Use the repository virtual environment at `aic/bin/python`.
- The public FastAPI API and frontend contracts remain unchanged.
- FastAPI never loads FAISS, Context, ASR, BM25, retrieval encoders, or temporal evidence and has no local fallback.
- Retrieval runs as one process bound to `127.0.0.1:8002` by default; do not add Uvicorn or reload behavior to it.
- Preserve `video_id`, `frame_id`, `frame_idx`, and `timestamp_ms` explicitly across gRPC; never infer canonical identity from array position.
- Do not send full-corpus score matrices over gRPC. Only top-result or explicitly selected video matrices may cross the boundary.
- Preserve the existing Dense, BM25, fusion, ASR projection, image scoring, and DP algorithms without semantic changes.
- Every RPC has an explicit deadline and no automatic scoring retry.
- Keep the existing dirty worktree intact. In particular, do not stage unrelated frontend/VBS changes or the existing untracked `uv.lock`.
- `src/hcmai/api/routers/kis.py` already contains user changes. Avoid modifying it: its existing generic failure branch already maps unexpected KIS gateway failures to 502.

---

## File Structure

New transport code is isolated from retrieval algorithms:

```text
src/hcmai/retrieval_service/
├── __init__.py                 # package boundary only
├── client.py                   # synchronous gRPC client and domain decoding
├── codec.py                    # protobuf/domain and binary NumPy conversion
├── config.py                   # validated client/server environment settings
├── errors.py                   # transport error taxonomy
├── remote.py                   # backend temporal/image workflow adapters
├── runtime.py                  # heavy local retrieval composition and calls
├── servicer.py                 # protobuf validation and gRPC status mapping
├── server.py                   # one-process CLI and standard health service
└── proto/
    ├── __init__.py
    ├── retrieval.proto
    ├── retrieval_pb2.py        # generated and committed
    └── retrieval_pb2_grpc.py   # generated and committed

tests/retrieval_service/
├── __init__.py
├── fakes.py
├── test_client.py
├── test_codec.py
├── test_proto_contract.py
├── test_remote_adapters.py
├── test_runtime.py
└── test_server.py
```

Existing modules remain responsible for orchestration, public HTTP, and pure
temporal decoding. Do not move algorithm implementations into the transport
package.

---

### Task 1: Add the versioned protobuf contract and generated bindings

**Requirements:** REQ-002, REQ-009, REQ-010, REQ-012, REQ-013

**Files:**
- Modify: `pyproject.toml`
- Create: `src/hcmai/retrieval_service/__init__.py`
- Create: `src/hcmai/retrieval_service/proto/__init__.py`
- Create: `src/hcmai/retrieval_service/proto/retrieval.proto`
- Generate: `src/hcmai/retrieval_service/proto/retrieval_pb2.py`
- Generate: `src/hcmai/retrieval_service/proto/retrieval_pb2_grpc.py`
- Create: `tests/retrieval_service/__init__.py`
- Create: `tests/retrieval_service/test_proto_contract.py`

**Interfaces:**
- Produces: protobuf service `hcmai.retrieval.v1.RetrievalService`
- Produces RPCs: `GetCapabilities`, `SearchPlan`, `SearchEvents`, `ScoreVideo`, `SearchImage`
- Produces messages consumed by every later task.

- [ ] **Step 1: Write the failing descriptor test**

```python
"""Contract tests for the private retrieval protobuf surface."""

from hcmai.retrieval_service.proto import retrieval_pb2


def test_REQ_009_proto_carries_all_canonical_identity_fields() -> None:
    path = retrieval_pb2.AlignedPath.DESCRIPTOR.fields_by_name
    candidate = retrieval_pb2.ImageCandidate.DESCRIPTOR.fields_by_name

    assert {"video_id", "frame_ids", "frame_idxs", "timestamps_ms"} <= set(path)
    assert {"video_id", "frame_id", "frame_idx", "timestamp_ms"} <= set(candidate)


def test_REQ_013_service_exposes_only_the_approved_rpc_surface() -> None:
    service = retrieval_pb2.DESCRIPTOR.services_by_name["RetrievalService"]
    assert [method.name for method in service.methods] == [
        "GetCapabilities",
        "SearchPlan",
        "SearchEvents",
        "ScoreVideo",
        "SearchImage",
    ]
```

- [ ] **Step 2: Run the test and confirm the missing module is the failure**

Run:

```bash
aic/bin/python -m pytest tests/retrieval_service/test_proto_contract.py -q
```

Expected: FAIL because `hcmai.retrieval_service.proto.retrieval_pb2` does not exist.

- [ ] **Step 3: Add gRPC dependencies without touching the untracked lockfile**

Add these runtime dependencies to `[project].dependencies`:

```toml
"grpcio>=1.76,<2.0",
"grpcio-health-checking>=1.76,<2.0",
"protobuf>=6.31,<7.0",
```

Add `"grpcio-tools>=1.76,<2.0"` to both existing dev dependency lists so the
project's duplicated dev declarations stay consistent. Install through the
required environment:

```bash
aic/bin/python -m pip install -e '.[dev]'
```

- [ ] **Step 4: Create the complete protobuf schema**

Use `syntax = "proto3"`, package `hcmai.retrieval.v1`, and define these exact
messages and fields:

```protobuf
syntax = "proto3";

package hcmai.retrieval.v1;

message GetCapabilitiesRequest {}

message GetCapabilitiesResponse {
  bool ready = 1;
  string scoring_revision = 2;
  repeated string active_modalities = 3;
  repeated string startup_messages = 4;
  uint32 max_temporal_event_count = 5;
  uint64 image_max_upload_bytes = 6;
  uint64 image_max_pixels = 7;
}

message RetrievalEvent {
  string event_id = 1;
  optional string canonical_text = 2;
  optional string dense_text = 3;
  optional string bm25_text = 4;
  repeated string image_asset_ids = 5;
}

message SearchPlanRequest {
  repeated RetrievalEvent events = 1;
  bool use_dense = 2;
  bool use_bm25 = 3;
  uint32 top_k = 4;
}

message SearchEventsRequest {
  repeated string original_events = 1;
  repeated string retrieval_events = 2;
  repeated string caption_events = 3;
  bool has_caption_events = 4;
  bool use_dense = 5;
  bool use_bm25 = 6;
  uint32 top_k = 7;
}

message AlignedPath {
  string video_id = 1;
  double score = 2;
  repeated string frame_ids = 3;
  repeated int64 frame_idxs = 4;
  repeated int64 timestamps_ms = 5;
}

message DecoderConfig {
  double lambda_gap = 1;
  double event_power = 2;
  double cluster_delta = 3;
  int64 path_min_separation_ms = 4;
}

message VideoScores {
  uint32 encoding_version = 1;
  string video_id = 2;
  repeated string frame_ids = 3;
  bytes frame_idxs_i64_le = 4;
  bytes timestamps_ms_i64_le = 5;
  bytes scores_f32_le = 6;
  uint32 event_count = 7;
  uint32 frame_count = 8;
}

message TemporalSearchResult {
  repeated AlignedPath paths = 1;
  double retrieval_ms = 2;
  double alignment_ms = 3;
  string scoring_revision = 4;
}

message TemporalSearchArtifact {
  TemporalSearchResult result = 1;
  repeated VideoScores video_scores = 2;
  DecoderConfig decoder_config = 3;
}

message ScoreVideoRequest {
  SearchPlanRequest plan = 1;
  string video_id = 2;
}

message SelectedVideoScore {
  VideoScores video_scores = 1;
  double retrieval_ms = 2;
  DecoderConfig decoder_config = 3;
  string scoring_revision = 4;
}

message SearchImageRequest {
  bytes payload = 1;
  string content_type = 2;
  uint32 top_k = 3;
}

message ImageCandidate {
  string video_id = 1;
  string frame_id = 2;
  int64 frame_idx = 3;
  int64 timestamp_ms = 4;
  double score = 5;
}

message ImageSearchCandidates {
  repeated ImageCandidate candidates = 1;
  double query_ms = 2;
  double retrieval_ms = 3;
}

service RetrievalService {
  rpc GetCapabilities(GetCapabilitiesRequest) returns (GetCapabilitiesResponse);
  rpc SearchPlan(SearchPlanRequest) returns (TemporalSearchArtifact);
  rpc SearchEvents(SearchEventsRequest) returns (TemporalSearchResult);
  rpc ScoreVideo(ScoreVideoRequest) returns (SelectedVideoScore);
  rpc SearchImage(SearchImageRequest) returns (ImageSearchCandidates);
}
```

- [ ] **Step 5: Generate and commit Python bindings**

Run:

```bash
aic/bin/python -m grpc_tools.protoc \
  -I src \
  --python_out=src \
  --grpc_python_out=src \
  src/hcmai/retrieval_service/proto/retrieval.proto
```

Do not manually edit generated files. Verify their imports use the package path
`hcmai.retrieval_service.proto`.

- [ ] **Step 6: Run the contract test**

Run:

```bash
aic/bin/python -m pytest tests/retrieval_service/test_proto_contract.py -q
```

Expected: 2 passed.

- [ ] **Step 7: Commit only Task 1 files**

```bash
git add pyproject.toml \
  src/hcmai/retrieval_service/__init__.py \
  src/hcmai/retrieval_service/proto \
  tests/retrieval_service/__init__.py \
  tests/retrieval_service/test_proto_contract.py
git commit -m "feat: define grpc retrieval contract"
```

---

### Task 2: Implement strict protobuf and NumPy codecs

**Requirements:** REQ-005, REQ-006, REQ-009, REQ-010

**Files:**
- Create: `src/hcmai/retrieval_service/codec.py`
- Create: `tests/retrieval_service/test_codec.py`

**Interfaces:**
- Produces: `encode_plan(plan, *, use_dense, use_bm25, top_k) -> SearchPlanRequest`
- Produces: `decode_plan(message) -> KISRetrievalPlan`
- Produces: `encode_video_scores(video) -> VideoScores`
- Produces: `decode_video_scores(message, *, expected_event_count, corpus) -> VideoEventScores`
- Produces path, decoder-config, result, and artifact conversion helpers used by client and servicer.

- [ ] **Step 1: Write failing round-trip and corruption tests**

Create a two-frame `VideoEventScores` fixture whose IDs are not derivable from
array positions. Assert:

```python
def test_REQ_009_video_score_round_trip_preserves_canonical_identity(corpus) -> None:
    encoded = encode_video_scores(_video_scores())
    decoded = decode_video_scores(encoded, expected_event_count=2, corpus=corpus)

    assert decoded.video_id == "L01_V001"
    assert decoded.frame_ids.tolist() == ["kf_alpha", "kf_omega"]
    assert decoded.frame_idx.tolist() == [17, 931]
    assert decoded.timestamps_ms.tolist() == [680, 37_240]
    assert decoded.scores.dtype == np.float32
    assert decoded.scores.shape == (2, 2)
    assert not decoded.scores.flags.writeable


def test_REQ_010_rejects_truncated_score_bytes(corpus) -> None:
    encoded = encode_video_scores(_video_scores())
    encoded.scores_f32_le = encoded.scores_f32_le[:-1]

    with pytest.raises(ValueError, match="score byte length"):
        decode_video_scores(encoded, expected_event_count=2, corpus=corpus)
```

Also cover wrong encoding version, event count, frame count, metadata byte
lengths, mixed video identity, mismatched `frame_idx`, mismatched timestamp,
optional text presence, path cardinality, and exact `E1..En` event order.

- [ ] **Step 2: Run the codec tests and confirm import failure**

```bash
aic/bin/python -m pytest tests/retrieval_service/test_codec.py -q
```

Expected: FAIL because `codec.py` does not exist.

- [ ] **Step 3: Implement the binary codec with one encoding constant**

Use:

```python
VIDEO_SCORES_ENCODING_VERSION = 1
_I64_LE = np.dtype("<i8")
_F32_LE = np.dtype("<f4")


def encode_video_scores(video: VideoEventScores) -> retrieval_pb2.VideoScores:
    frame_idxs = np.asarray(video.frame_idx, dtype=_I64_LE)
    timestamps = np.asarray(video.timestamps_ms, dtype=_I64_LE)
    scores = np.asarray(video.scores, dtype=_F32_LE, order="C")
    event_count, frame_count = scores.shape
    return retrieval_pb2.VideoScores(
        encoding_version=VIDEO_SCORES_ENCODING_VERSION,
        video_id=video.video_id,
        frame_ids=[str(value) for value in video.frame_ids],
        frame_idxs_i64_le=frame_idxs.tobytes(order="C"),
        timestamps_ms_i64_le=timestamps.tobytes(order="C"),
        scores_f32_le=scores.tobytes(order="C"),
        event_count=event_count,
        frame_count=frame_count,
    )
```

Decode with `np.frombuffer(...).copy()`, validate byte counts before reshape,
validate every frame against `Corpus`, and set all returned arrays read-only.
Use `HasField` for proto optional strings so `None` is not collapsed into an
empty string.

- [ ] **Step 4: Implement domain conversions without duplicating schemas**

Convert directly between protobuf messages and existing domain types:

```python
KISRetrievalPlan
KISRetrievalEvent
AlignedPath
TemporalSearchResult
TemporalSearchArtifact
DecoderConfigSnapshot
VideoEventScores
```

Add `scoring_revision: str | None = None` to `TemporalSearchArtifact` so remote
artifacts retain their serving generation while existing local tests remain
constructible.

- [ ] **Step 5: Run codec and existing temporal artifact tests**

```bash
aic/bin/python -m pytest \
  tests/retrieval_service/test_codec.py \
  tests/orchestration/workflows/test_temporal_search_artifact.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/hcmai/retrieval_service/codec.py \
  src/hcmai/orchestration/workflows/temporal_search.py \
  tests/retrieval_service/test_codec.py
git commit -m "feat: encode retrieval grpc payloads"
```

---

### Task 3: Introduce a selected-video temporal gateway seam

**Requirements:** REQ-005, REQ-006, REQ-008, REQ-010

**Files:**
- Modify: `src/hcmai/orchestration/workflows/temporal_search.py`
- Modify: `src/hcmai/orchestration/workflows/kis.py`
- Modify: `src/hcmai/orchestration/workflows/trake.py`
- Modify: `src/hcmai/orchestration/workflows/temporal_exploration.py`
- Modify: `tests/orchestration/workflows/test_kis_pipeline.py`
- Modify: `tests/orchestration/test_temporal_exploration.py`
- Modify: `tests/orchestration/workflows/test_temporal_exploration.py`

**Interfaces:**
- Produces: `SelectedVideoScoreResult(video, retrieval_ms, decoder_config, scoring_revision)`
- Produces: `TemporalSearchGateway` protocol used by KIS, TRAKE, event-trail, and exploration.
- Produces: `TemporalSearchService.score_video(...)` for local server-side reuse.
- Produces: pure `decode_video_scores(...)` used by both local and remote adapters.

- [ ] **Step 1: Write a failing selected-video test**

```python
def test_REQ_010_score_video_returns_only_requested_video(temporal_service, plan) -> None:
    selected = temporal_service.score_video(
        plan,
        video_id="video-b",
        use_dense=True,
        use_bm25=False,
    )

    assert selected.video.video_id == "video-b"
    assert selected.retrieval_ms >= 0
```

Update exploration fakes to expose `score_video` and assert `score_plan` is not
called when opening a selected-video branch.

- [ ] **Step 2: Run focused tests and confirm the missing method**

```bash
aic/bin/python -m pytest \
  tests/orchestration/workflows/test_temporal_search_artifact.py \
  tests/orchestration/test_temporal_exploration.py \
  tests/orchestration/workflows/test_temporal_exploration.py -q
```

Expected: FAIL because `score_video` is missing.

- [ ] **Step 3: Add the gateway contract and selected result**

Define:

```python
@dataclass(frozen=True, slots=True)
class SelectedVideoScoreResult:
    video: VideoEventScores
    retrieval_ms: float
    decoder_config: DecoderConfigSnapshot
    scoring_revision: str | None = None


class TemporalSearchGateway(Protocol):
    corpus: Corpus

    def search(self, original_events: Sequence[str], *, top_k: int,
               retrieval_events: Sequence[str] | None = None,
               caption_events: Sequence[str] | None = None,
               use_dense: bool = True,
               use_bm25: bool = False) -> TemporalSearchResult: ...

    def search_plan_artifact(self, plan: KISRetrievalPlan, *,
                             image_component: TemporalScoreComponent | None = None,
                             use_dense: bool = True,
                             use_bm25: bool = False,
                             top_k: int = 20) -> TemporalSearchArtifact: ...

    def score_video(self, plan: KISRetrievalPlan, *, video_id: str,
                    image_component: TemporalScoreComponent | None = None,
                    use_dense: bool = True,
                    use_bm25: bool = False) -> SelectedVideoScoreResult: ...

    def decode_video(self, video: VideoEventScores, *, allowed: np.ndarray,
                     decoder_config: DecoderConfigSnapshot | None = None
                     ) -> tuple[AlignedPath, ...]: ...

```

- [ ] **Step 4: Implement local `score_video` and pure decoding reuse**

`TemporalSearchService.score_video` calls `score_plan` exactly once, selects by
explicit `video_id`, raises `KeyError` when absent, and returns the current
decoder snapshot. Extract the existing `decode_video` body to a helper accepting
Corpus and `DecoderConfigSnapshot`; keep canonical validation unchanged.

- [ ] **Step 5: Make workflows consume the gateway**

- Type KIS and TRAKE against `TemporalSearchGateway`.
- Keep KIS's optional local `image_scorer` behavior for server-side/local tests,
  but when it is absent pass the full plan to the gateway instead of rejecting
  image refs. The remote adapter will score those refs.
- Change exploration open from “score all then search for video” to one
  `score_video(plan, video_id=...)` call. If a local `image_scorer` is injected,
  pass its component; otherwise let the gateway own image scoring.
- Store the returned decoder config and scoring revision with the branch.

- [ ] **Step 6: Run focused workflow tests**

```bash
aic/bin/python -m pytest \
  tests/orchestration/workflows/test_kis_pipeline.py \
  tests/orchestration/test_temporal_exploration.py \
  tests/orchestration/workflows/test_temporal_exploration.py \
  tests/event_trail/test_decoder.py -q
```

Expected: all pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add src/hcmai/orchestration/workflows/temporal_search.py \
  src/hcmai/orchestration/workflows/kis.py \
  src/hcmai/orchestration/workflows/trake.py \
  src/hcmai/orchestration/workflows/temporal_exploration.py \
  tests/orchestration/workflows/test_kis_pipeline.py \
  tests/orchestration/test_temporal_exploration.py \
  tests/orchestration/workflows/test_temporal_exploration.py
git commit -m "refactor: isolate temporal retrieval gateway"
```

---

### Task 4: Compose the heavy retrieval runtime once

**Requirements:** REQ-002, REQ-005, REQ-006, REQ-007, REQ-013

**Files:**
- Modify: `src/hcmai/orchestration/corpus_setup.py`
- Modify: `src/hcmai/orchestration/setup.py`
- Create: `src/hcmai/retrieval_service/runtime.py`
- Create: `tests/retrieval_service/fakes.py`
- Create: `tests/retrieval_service/test_runtime.py`

**Interfaces:**
- Produces: `RetrievalRuntime.load(messages) -> RetrievalRuntime | None`
- Produces methods: `capabilities`, `search_plan`, `search_events`, `score_video`, `search_image`
- Reuses existing local index loaders and algorithms without copying them.

- [ ] **Step 1: Write failing runtime delegation tests**

Use fake temporal, image scorer, and image search components. Assert:

```python
def test_REQ_005_runtime_scores_kis_images_before_existing_temporal_search(runtime, plan) -> None:
    artifact = runtime.search_plan(plan, use_dense=True, use_bm25=True, top_k=5)

    runtime.image_scorer.score_events.assert_called_once_with(plan.image_ref_rows)
    runtime.temporal.search_plan_artifact.assert_called_once_with(
        plan,
        image_component=runtime.image_scorer.score_events.return_value,
        use_dense=True,
        use_bm25=True,
        top_k=5,
    )
    assert artifact.scoring_revision == runtime.scoring_revision


def test_REQ_010_runtime_score_video_returns_one_video(runtime, plan) -> None:
    selected = runtime.score_video(plan, "video-2", use_dense=True, use_bm25=False)
    assert selected.video.video_id == "video-2"
    assert selected.scoring_revision == runtime.scoring_revision
```

Also assert capabilities contain only actually loaded modalities and do not run
retrieval work.

- [ ] **Step 2: Run runtime tests and confirm missing module**

```bash
aic/bin/python -m pytest tests/retrieval_service/test_runtime.py -q
```

Expected: FAIL because `runtime.py` does not exist.

- [ ] **Step 3: Promote shared startup helpers instead of duplicating them**

In `orchestration/setup.py`, rename and keep using:

```python
load_app_config()
load_model_config()
load_remote_inference(settings, messages)
load_kis_image_assets(settings, messages)
```

In `corpus_setup.py`, add:

```python
def load_configured_corpus(settings: AppConfig, messages: list[str]) -> Corpus | None:
    """Resolve configured metadata roots and load the canonical Corpus."""
```

Move the existing metadata/dataset-root resolution from `load_search_service`
into this helper so backend and retrieval process resolve identical paths.

- [ ] **Step 4: Implement `RetrievalRuntime` as a thin composition facade**

Use a frozen/slotted dataclass holding:

```python
corpus: Corpus
temporal: TemporalSearchService
image_scorer: ImageQueryTemporalScorer | None
image_search: ImageSearchService | None
active_modalities: tuple[str, ...]
startup_messages: tuple[str, ...]
max_temporal_event_count: int
image_max_upload_bytes: int
image_max_pixels: int
scoring_revision: str
```

`load()` must call the existing `load_retrieval`, `select_visual_retriever`,
`load_image_encoder`, and `load_temporal_evidence` functions exactly once.
Return `None` when required visual/temporal capabilities are absent; retain
diagnostic messages. Generate `scoring_revision` only for a ready runtime.

- [ ] **Step 5: Implement runtime methods by delegation**

- `search_plan`: compute image component only when refs exist, call existing
  `TemporalSearchService.search_plan_artifact`, attach runtime revision, and
  retain score rows only for distinct returned video IDs.
- `search_events`: call existing `TemporalSearchService.search` unchanged.
- `score_video`: compute an optional image component, call local `score_video`,
  and attach runtime revision.
- `search_image`: call existing `ImageSearchService.search`; transport projection
  happens later in the servicer.

- [ ] **Step 6: Run runtime and loader regression tests**

```bash
aic/bin/python -m pytest \
  tests/retrieval_service/test_runtime.py \
  tests/orchestration/test_retrieval_setup.py \
  tests/orchestration/test_setup_modules.py -q
```

Expected: all pass.

- [ ] **Step 7: Commit Task 4**

```bash
git add src/hcmai/orchestration/corpus_setup.py \
  src/hcmai/orchestration/setup.py \
  src/hcmai/retrieval_service/runtime.py \
  tests/retrieval_service/fakes.py \
  tests/retrieval_service/test_runtime.py
git commit -m "feat: compose standalone retrieval runtime"
```

---

### Task 5: Serve the runtime through one gRPC process

**Requirements:** REQ-002, REQ-005, REQ-006, REQ-007, REQ-011, REQ-012, REQ-013

**Files:**
- Create: `src/hcmai/retrieval_service/servicer.py`
- Create: `src/hcmai/retrieval_service/server.py`
- Create: `tests/retrieval_service/test_server.py`

**Interfaces:**
- Produces: `RetrievalServicer(runtime)`
- Produces: `create_server(runtime, *, host, port, max_message_bytes)`
- Produces CLI: `python -m hcmai.retrieval_service.server --host ... --port ...`

- [ ] **Step 1: Write failing in-process server tests**

Start the gRPC server on `127.0.0.1:0` with a fake runtime and generated stub.
Test:

```python
def test_REQ_013_ready_runtime_reports_serving(grpc_server) -> None:
    health = health_pb2_grpc.HealthStub(grpc_server.channel)
    response = health.Check(
        health_pb2.HealthCheckRequest(service="hcmai.retrieval.v1.RetrievalService"),
        timeout=1,
    )
    assert response.status == health_pb2.HealthCheckResponse.SERVING


def test_REQ_011_invalid_plan_maps_to_invalid_argument(grpc_server) -> None:
    with pytest.raises(grpc.RpcError) as caught:
        grpc_server.stub.SearchPlan(retrieval_pb2.SearchPlanRequest(), timeout=1)
    assert caught.value.code() is grpc.StatusCode.INVALID_ARGUMENT
```

Also cover not-ready runtime → health `NOT_SERVING` and RPC `UNAVAILABLE`,
missing selected video → `NOT_FOUND`, oversized input → `RESOURCE_EXHAUSTED`,
and unexpected runtime error → `INTERNAL` without stack-trace detail.

- [ ] **Step 2: Run server tests and confirm missing server**

```bash
aic/bin/python -m pytest tests/retrieval_service/test_server.py -q
```

Expected: FAIL because server/servicer modules do not exist.

- [ ] **Step 3: Implement validation and error mapping in the servicer**

Each handler must:

1. reject an unavailable runtime with `context.abort(UNAVAILABLE, ...)`;
2. decode/validate protobuf before calling the runtime;
3. catch `ValueError` as `INVALID_ARGUMENT`, `KeyError` as `NOT_FOUND`, bounded
   input overflow as `RESOURCE_EXHAUSTED`, and unexpected exceptions as
   `INTERNAL`;
4. log exception type internally without returning stack traces;
5. convert the domain result through `codec.py`.

Do not catch `BaseException` and do not retry.

- [ ] **Step 4: Implement the one-process server and standard health service**

`create_server` must configure both channel limits:

```python
options = (
    ("grpc.max_send_message_length", max_message_bytes),
    ("grpc.max_receive_message_length", max_message_bytes),
)
server = grpc.server(ThreadPoolExecutor(max_workers=4), options=options)
```

Register `RetrievalServicer` and `grpc_health.v1.HealthServicer`. Set both the
empty service name and the fully-qualified retrieval service to `SERVING` only
for a ready runtime.

- [ ] **Step 5: Implement CLI lifecycle**

Parse `--host` (default `127.0.0.1`) and `--port` (default `8002`). Load the
repository environment, configure logging, attempt `RetrievalRuntime.load`,
start the listener even when runtime loading fails, and wait for termination.
On `KeyboardInterrupt`, set health to `NOT_SERVING` and call
`server.stop(grace=5)`.

- [ ] **Step 6: Run server and codec tests**

```bash
aic/bin/python -m pytest \
  tests/retrieval_service/test_server.py \
  tests/retrieval_service/test_codec.py -q
```

Expected: all pass.

- [ ] **Step 7: Commit Task 5**

```bash
git add src/hcmai/retrieval_service/servicer.py \
  src/hcmai/retrieval_service/server.py \
  tests/retrieval_service/test_server.py
git commit -m "feat: serve retrieval over grpc"
```

---

### Task 6: Build the reconnecting backend client and remote adapters

**Requirements:** REQ-003, REQ-004, REQ-007, REQ-009, REQ-011, REQ-012

**Files:**
- Create: `src/hcmai/retrieval_service/config.py`
- Create: `src/hcmai/retrieval_service/errors.py`
- Create: `src/hcmai/retrieval_service/client.py`
- Create: `src/hcmai/retrieval_service/remote.py`
- Create: `tests/retrieval_service/test_client.py`
- Create: `tests/retrieval_service/test_remote_adapters.py`

**Interfaces:**
- Produces: `RetrievalClientSettings.from_env()`
- Produces: `RetrievalGrpcClient`
- Produces: `RemoteTemporalSearchService`
- Produces: `RemoteImageSearchService`
- Produces typed errors: unavailable, invalid request, not found, too large, protocol/internal.

- [ ] **Step 1: Write failing configuration and status-mapping tests**

Cover defaults and invalid values:

```python
def test_settings_default_to_loopback_service() -> None:
    settings = RetrievalClientSettings.from_env({})
    assert settings.target == "127.0.0.1:8002"
    assert settings.timeout_seconds == 120
    assert settings.health_timeout_seconds == 1
    assert settings.max_message_bytes == 64 * 1024 * 1024


@pytest.mark.parametrize("code", [grpc.StatusCode.UNAVAILABLE,
                                  grpc.StatusCode.DEADLINE_EXCEEDED])
def test_REQ_011_connectivity_codes_are_unavailable(code) -> None:
    assert map_rpc_error(_rpc_error(code)).category == "unavailable"
```

Reject targets with `http://`/`https://`, non-positive deadlines, and message
limits outside 1–512 MiB.

- [ ] **Step 2: Run client tests and confirm missing modules**

```bash
aic/bin/python -m pytest \
  tests/retrieval_service/test_client.py \
  tests/retrieval_service/test_remote_adapters.py -q
```

Expected: FAIL because the client and adapters do not exist.

- [ ] **Step 3: Implement settings and typed errors**

Use a frozen dataclass:

```python
@dataclass(frozen=True, slots=True)
class RetrievalClientSettings:
    target: str = "127.0.0.1:8002"
    timeout_seconds: float = 120.0
    health_timeout_seconds: float = 1.0
    max_message_bytes: int = 64 * 1024 * 1024
```

Read the four exact environment names from the spec. Error messages must name
the variable but never echo a secret or binary payload.

- [ ] **Step 4: Implement one long-lived synchronous gRPC client**

Construct one insecure channel with matching send/receive limits and retain the
generated application stub plus health stub. Every method passes an explicit
`timeout=` and converts `grpc.RpcError` through the typed taxonomy. Do not set a
retry service config.

Provide:

```python
probe() -> RemoteRetrievalStatus
get_capabilities() -> RemoteRetrievalStatus
search_plan(...) -> TemporalSearchArtifact
search_events(...) -> TemporalSearchResult
score_video(...) -> SelectedVideoScoreResult
search_image(...) -> RemoteImageSearchResult
close() -> None
```

`probe()` returns a non-throwing status for startup/health; application calls
raise typed errors.

- [ ] **Step 5: Implement remote temporal and image adapters**

`RemoteTemporalSearchService` exposes the `TemporalSearchGateway` interface,
holds backend Corpus for validation/materialization, and delegates network work
to `RetrievalGrpcClient`. Its local `decode_video` uses the pure decoder helper
from Task 3 and never loads evidence or indexes. `get_scoring_revision()` calls
the bounded capabilities RPC and rejects a blank/not-ready revision.

`RemoteImageSearchService` exposes the existing `SUPPORTED_MEDIA_TYPES`, upload
limits, and `search(payload, content_type, top_k) -> ImageSearchResponse` shape.
It validates byte/media bounds before sending, validates each returned canonical
candidate against Corpus, materializes through `SearchMaterializer`, and combines
remote query/retrieval timings with local materialization timing.

Define `RemoteImageSearchResult` and `RemoteImageCandidate` as frozen, slotted
transport-neutral dataclasses in `remote.py`; do not expose protobuf message
objects to orchestration code.

- [ ] **Step 6: Verify channel reconnect behavior**

Create a client channel to a reserved loopback port before starting the test
server. Confirm the initial probe is unavailable, start the fake server on the
same port, wait using `grpc.channel_ready_future(client.channel).result(timeout=5)`,
then confirm the next application call succeeds without reconstructing client.

- [ ] **Step 7: Run client and adapter tests**

```bash
aic/bin/python -m pytest \
  tests/retrieval_service/test_client.py \
  tests/retrieval_service/test_remote_adapters.py -q
```

Expected: all pass.

- [ ] **Step 8: Commit Task 6**

```bash
git add src/hcmai/retrieval_service/config.py \
  src/hcmai/retrieval_service/errors.py \
  src/hcmai/retrieval_service/client.py \
  src/hcmai/retrieval_service/remote.py \
  tests/retrieval_service/test_client.py \
  tests/retrieval_service/test_remote_adapters.py
git commit -m "feat: add grpc retrieval client"
```

---

### Task 7: Remove local retrieval loading from FastAPI composition

**Requirements:** REQ-001, REQ-003, REQ-004, REQ-011, REQ-013, REQ-014

**Files:**
- Modify: `src/hcmai/orchestration/setup.py`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Modify: `src/hcmai/orchestration/utils/health.py`
- Modify: `src/hcmai/api/routers/system.py`
- Modify: `src/hcmai/api/routers/search.py`
- Modify: `src/hcmai/api/routers/trake.py`
- Modify: `tests/orchestration/test_pipeline.py`
- Modify: `tests/orchestration/test_health.py`
- Modify: `tests/orchestration/test_kis_multimodal_prerequisites.py`
- Modify: `tests/api/test_dres_result_logging.py`
- Create: `tests/orchestration/test_remote_setup.py`

**Interfaces:**
- FastAPI `SearchService` receives high-level temporal/image adapters rather than local indexes.
- `/health` exposes `remote_retrieval` status while keeping existing keys compatible.
- Public search routes retain response contracts and map gateway failures consistently.

- [ ] **Step 1: Write the failing no-local-loader startup test**

```python
def test_REQ_001_backend_setup_never_loads_local_retrieval(monkeypatch) -> None:
    monkeypatch.setattr(
        "hcmai.orchestration.retrieval_setup.load_retrieval",
        lambda *args, **kwargs: pytest.fail("FastAPI attempted local index loading"),
    )
    monkeypatch.setattr(RetrievalGrpcClient, "probe", lambda self: _unavailable_status())

    service = load_search_service([])

    assert service.remote_retrieval is not None
    assert service.literal_text is not None
```

Also create a FastAPI lifespan test with an unused gRPC port and assert `/health`
returns successfully while `/api/v1/filter` remains usable and a semantic search
returns 503.

- [ ] **Step 2: Run focused setup tests and confirm local loading occurs**

```bash
aic/bin/python -m pytest \
  tests/orchestration/test_remote_setup.py \
  tests/orchestration/test_pipeline.py \
  tests/orchestration/test_health.py -q
```

Expected: FAIL because `load_search_service` still calls local retrieval loaders.

- [ ] **Step 3: Change `SearchService` to high-level dependencies**

Replace low-level constructor inputs (`retrieval`, `temporal_evidence`,
`image_encoder`, `visual_retriever`, legacy retrieval `llm`) with:

```python
temporal: TemporalSearchGateway | None
image_search: RemoteImageSearchService | ImageSearchService | None
remote_retrieval: RetrievalGrpcClient | None
```

Construct KIS, TRAKE, and event-trail from `temporal`. Keep Corpus, literal
filter, semantic resolver/rewriter/translator, assets, and session stores local.
`close()` closes the gRPC channel. `_ensure_search_ready()` checks Corpus and
the temporal gateway, not local evidence objects.

Translate typed remote errors at the `SearchService` boundary:

- unavailable/deadline → `SearchServiceUnavailableError`;
- invalid argument → `InvalidQueryInputError`;
- not found → `KeyError`;
- oversized image → `ImageQueryTooLargeError`;
- protocol/internal → new `SearchServiceGatewayError`.

Do not edit the dirty KIS router; its existing catch-all already returns 502.
Add explicit 502 branches to image and TRAKE routers.

- [ ] **Step 4: Rewrite backend setup to construct only remote adapters**

`load_search_service` must:

1. load app config and Corpus;
2. load semantic LLM client, translator, KIS resolvers/rewriter, literal index,
   and KIS asset store;
3. construct `RetrievalClientSettings`, `RetrievalGrpcClient`,
   `RemoteTemporalSearchService`, and `RemoteImageSearchService`;
4. perform one non-fatal probe and append exactly one clear startup message when
   unavailable;
5. never load model config, legacy remote inference, local retrieval, visual
   retriever, image encoder, or temporal evidence.

- [ ] **Step 5: Make health probing bounded and non-blocking to FastAPI**

`build_health_report` receives an optional already-computed remote status and
projects:

```python
"remote_retrieval": {
    "configured": True,
    "reachable": status.reachable,
    "ready": status.ready,
    "target": status.target,
    "scoring_revision": status.scoring_revision,
    "active_modalities": list(status.active_modalities),
}
```

Update `system.py` to call `service.health(...)` via `run_in_threadpool` because
the readiness probe is network I/O. Preserve existing health keys for frontend
compatibility.

- [ ] **Step 6: Update constructor fixtures and run backend tests**

Update tests to inject fake temporal/image/remote clients at the new boundary;
do not rebuild low-level retrieval fakes inside `SearchService` tests.

Run:

```bash
aic/bin/python -m pytest \
  tests/orchestration/test_remote_setup.py \
  tests/orchestration/test_pipeline.py \
  tests/orchestration/test_health.py \
  tests/orchestration/test_kis_multimodal_prerequisites.py \
  tests/api/test_dres_result_logging.py -q
```

Expected: all pass.

- [ ] **Step 7: Commit Task 7 without staging dirty user files**

```bash
git add src/hcmai/orchestration/setup.py \
  src/hcmai/orchestration/pipeline.py \
  src/hcmai/orchestration/utils/health.py \
  src/hcmai/api/routers/system.py \
  src/hcmai/api/routers/search.py \
  src/hcmai/api/routers/trake.py \
  tests/orchestration/test_pipeline.py \
  tests/orchestration/test_health.py \
  tests/orchestration/test_kis_multimodal_prerequisites.py \
  tests/orchestration/test_remote_setup.py \
  tests/api/test_dres_result_logging.py
git commit -m "refactor: use remote retrieval in backend"
```

---

### Task 8: Bind event-trail and exploration to retrieval revisions

**Requirements:** REQ-008, REQ-009

**Files:**
- Modify: `src/hcmai/orchestration/pipeline.py`
- Modify: `src/hcmai/orchestration/workflows/temporal_exploration.py`
- Modify: `src/hcmai/api/routers/exploration.py`
- Modify: `tests/event_trail/test_kis_integration.py`
- Modify: `tests/api/test_exploration_router.py`
- Modify: `tests/orchestration/test_temporal_exploration.py`
- Modify: `tests/orchestration/workflows/test_temporal_exploration.py`

**Interfaces:**
- KIS `EvidenceSnapshot.scoring_revision` comes from `SearchPlan`, not FastAPI process UUID.
- Each exploration registry entry stores the revision returned by `ScoreVideo`.
- A changed retrieval revision rejects stale state.

- [ ] **Step 1: Write failing retrieval-revision tests**

```python
def test_REQ_008_kis_snapshot_uses_remote_scoring_revision(service, request) -> None:
    service.kis.temporal.search_plan_artifact.return_value.scoring_revision = "remote-v7"
    response = service.search_kis(request)
    snapshot = service.event_trail_snapshots.get(response.evidence_snapshot_id)
    assert snapshot.scoring_revision == "remote-v7"


def test_REQ_008_exploration_returns_score_video_revision(client) -> None:
    response = client.post("/api/v1/exploration", json=_open_payload())
    assert response.json()["scoring_revision"] == "remote-v7"
```

Add a test where capability revision and `ScoreVideo` revision differ; opening
must fail with 409/503 and must not publish a registry handle.

- [ ] **Step 2: Run focused tests and confirm they expose local UUID behavior**

```bash
aic/bin/python -m pytest \
  tests/event_trail/test_kis_integration.py \
  tests/api/test_exploration_router.py \
  tests/orchestration/test_temporal_exploration.py -q
```

Expected: FAIL because revisions are currently owned by FastAPI/registry.

- [ ] **Step 3: Use artifact revision for event-trail snapshots**

Remove `SearchService.event_trail_scoring_revision`. Require a nonblank remote
revision whenever a temporal artifact is present and store it in
`EvidenceSnapshot`. A missing revision from a real remote adapter is a protocol
error; unit fakes must provide one explicitly.

- [ ] **Step 4: Make exploration registry revisions entry-scoped**

Remove the registry-wide UUID. Change reservation/publication to:

```python
handle = registry.reserve()
registry.publish(handle, branch, scoring_revision=branch.scoring_revision)
```

`_open_branch` obtains the current revision from the temporal gateway, builds
the binding, and verifies that `ScoreVideo` returns the same revision before
publishing. Every later action compares the client revision with the entry and
binding revisions. A mismatch never mutates history.

- [ ] **Step 5: Run event-trail and exploration tests**

```bash
aic/bin/python -m pytest \
  tests/event_trail \
  tests/api/test_event_trail_routes.py \
  tests/api/test_exploration_router.py \
  tests/orchestration/test_temporal_exploration.py \
  tests/orchestration/workflows/test_temporal_exploration.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit Task 8**

```bash
git add src/hcmai/orchestration/pipeline.py \
  src/hcmai/orchestration/workflows/temporal_exploration.py \
  src/hcmai/api/routers/exploration.py \
  tests/event_trail/test_kis_integration.py \
  tests/api/test_exploration_router.py \
  tests/orchestration/test_temporal_exploration.py \
  tests/orchestration/workflows/test_temporal_exploration.py
git commit -m "fix: bind temporal state to retrieval revision"
```

---

### Task 9: Prove transport equivalence and preserve public behavior

**Requirements:** REQ-004, REQ-005, REQ-006, REQ-007, REQ-009, REQ-014

**Files:**
- Create: `tests/retrieval_service/test_integration.py`
- Modify: `tests/api/test_kis_router.py`
- Modify: `tests/api/test_dres_result_logging.py`
- Modify: `tests/orchestration/workflows/test_kis_pipeline.py`
- Modify: `tests/orchestration/workflows/test_temporal_search_artifact.py`
- Modify: `tests/test_kis_acceptance_smoke.py`

**Interfaces:**
- No new production interface; this task verifies that the gRPC boundary is behavior-preserving.

- [ ] **Step 1: Write the in-process equivalence test**

Build a hand-checkable two-video fake runtime using real domain objects. Run the
same plan directly through the fake runtime and through
server → stub/client → remote adapter. Assert exact equality for:

```python
assert remote.result.paths == local.result.paths
assert remote.decoder_config == local.decoder_config
assert [row.video_id for row in remote.video_scores] == [
    row.video_id for row in local.video_scores
]
np.testing.assert_array_equal(remote.video_scores[0].frame_idx,
                              local.video_scores[0].frame_idx)
np.testing.assert_array_equal(remote.video_scores[0].timestamps_ms,
                              local.video_scores[0].timestamps_ms)
np.testing.assert_allclose(remote.video_scores[0].scores,
                           local.video_scores[0].scores,
                           rtol=0, atol=0)
```

Add image candidate and TRAKE path equivalence cases.

- [ ] **Step 2: Run integration test and observe any contract drift**

```bash
aic/bin/python -m pytest tests/retrieval_service/test_integration.py -q
```

Expected before final fixture/adapter adjustments: at least one assertion fails
for any field not yet preserved. Fix production codecs/adapters, not assertions.

- [ ] **Step 3: Update public regression fixtures to the high-level boundary**

Keep public requests and responses unchanged. Update only dependency injection:
existing router/acceptance tests receive a fake `SearchService` or remote
gateway, never a local FAISS/index loader.

- [ ] **Step 4: Run all retrieval-facing public tests**

```bash
aic/bin/python -m pytest \
  tests/retrieval_service \
  tests/api/test_kis_router.py \
  tests/api/test_dres_result_logging.py \
  tests/api/test_event_trail_routes.py \
  tests/api/test_exploration_router.py \
  tests/orchestration/workflows \
  tests/test_kis_acceptance_smoke.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit Task 9**

```bash
git add tests/retrieval_service/test_integration.py \
  tests/api/test_kis_router.py \
  tests/api/test_dres_result_logging.py \
  tests/orchestration/workflows/test_kis_pipeline.py \
  tests/orchestration/workflows/test_temporal_search_artifact.py \
  tests/test_kis_acceptance_smoke.py \
  src/hcmai/retrieval_service
git commit -m "test: verify grpc retrieval equivalence"
```

---

### Task 10: Document operation and run final verification

**Requirements:** REQ-003, REQ-013, REQ-015

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `src/hcmai/orchestration/README.md`
- Create: `src/hcmai/retrieval_service/README.md`

**Interfaces:**
- Documents the stable two-process local runbook and health checks.

- [ ] **Step 1: Write a failing documentation assertion test**

Add to `tests/retrieval_service/test_server.py`:

```python
from pathlib import Path


def test_REQ_015_readme_documents_separate_processes() -> None:
    readme = Path("README.md").read_text()
    assert "hcmai.retrieval_service.server" in readme
    assert "--port 8002" in readme
    assert "uvicorn hcmai.app:app" in readme
    assert "--reload" in readme
```

- [ ] **Step 2: Run the assertion and confirm docs are missing**

```bash
aic/bin/python -m pytest \
  tests/retrieval_service/test_server.py::test_REQ_015_readme_documents_separate_processes -q
```

Expected: FAIL until the runbook is updated.

- [ ] **Step 3: Update environment and run commands**

Add to `.env.example`:

```dotenv
HCMAI_RETRIEVAL_TARGET=127.0.0.1:8002
HCMAI_RETRIEVAL_TIMEOUT_SECONDS=120
HCMAI_RETRIEVAL_HEALTH_TIMEOUT_SECONDS=1
HCMAI_RETRIEVAL_MAX_MESSAGE_MIB=64
```

Document two terminals:

```bash
# Terminal 1: stable retrieval process; one process, no --reload
aic/bin/python -m hcmai.retrieval_service.server --host 127.0.0.1 --port 8002

# Terminal 2: reloadable public backend
aic/bin/python -m uvicorn hcmai.app:app --host 127.0.0.1 --port 8000 --reload
```

State explicitly that retrieval must not be launched multiple times/workers and
that backend startup warnings are expected when retrieval is absent.

- [ ] **Step 4: Run focused and full automated verification**

```bash
aic/bin/python -m pytest tests/retrieval_service -q
aic/bin/python -m pytest tests/orchestration tests/event_trail tests/api -q
aic/bin/python -m pytest -q
```

Record exact pass/fail counts. If unrelated pre-existing failures arise from the
dirty worktree, rerun the affected test on the baseline/isolated worktree before
classifying it; do not modify unrelated user changes.

- [ ] **Step 5: Run import and CLI smoke checks**

```bash
aic/bin/python -m hcmai.retrieval_service.server --help
aic/bin/python -c "from hcmai.app import app; print(app.title)"
```

Expected: both commands exit 0 without loading production indexes.

- [ ] **Step 6: Run an operational two-process smoke test when artifacts exist**

Start retrieval, wait for standard gRPC health `SERVING`, start FastAPI, verify
`GET /health` reports remote retrieval ready, run one KIS/TRAKE/image query, then
restart only FastAPI and confirm the retrieval PID and scoring revision remain
unchanged. If production artifacts/providers are unavailable, report this step
as not run; do not claim it passed from unit tests.

- [ ] **Step 7: Check scope, formatting, and accidental staging**

```bash
git diff --check
git status --short
git diff --stat
```

Confirm `KNOWLEDGE.md` is unchanged because no algorithmic/research finding was
introduced. Confirm user-owned frontend/VBS files and untracked `uv.lock` remain
unstaged.

- [ ] **Step 8: Commit documentation**

```bash
git add .env.example README.md \
  src/hcmai/orchestration/README.md \
  src/hcmai/retrieval_service/README.md \
  tests/retrieval_service/test_server.py
git commit -m "docs: add grpc retrieval runbook"
```

---

## Completion Checklist

- [ ] Every REQ-001 through REQ-015 has at least one named test.
- [ ] FastAPI setup contains no call path to local retrieval loaders.
- [ ] Retrieval service runs in one non-Uvicorn process on port 8002.
- [ ] KIS, TRAKE, image search, event-trail, exploration, filter, and health tests pass.
- [ ] Canonical identity and binary matrix validation tests pass.
- [ ] Backend starts and logs clearly while retrieval is unavailable.
- [ ] A later retrieval startup is usable through the existing client channel.
- [ ] Public HTTP contracts remain unchanged.
- [ ] Full verification output is recorded before any completion claim.
- [ ] No unrelated dirty-worktree files are staged or reverted.
