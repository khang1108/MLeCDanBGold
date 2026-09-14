# KIS Semantic Intent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace deterministic raw-query splitting with provider-agnostic SLM semantic intent resolution, feed the resolved event graph into the existing temporal DP, and finish the Unified KIS Panel without retaining superseded KIS/query-preparation code.

**Architecture:** Ordered KIS clues are resolved by `KISIntentResolver` through a capability-level `LLMClient` into a validated semantic graph of entities, event nodes, entity bindings, and `BEFORE` edges. Retrieval/query preparation operates only on the resolved event texts; the current `TemporalSearchService`/DP receives their canonical order unchanged. Provider endpoints and model names come from `.env`, while domain prompts remain version-controlled.

**Tech Stack:** Python 3, Pydantic/FastAPI/httpx, existing HCMAI hybrid temporal retrieval + NumPy DP, React/Jest/Testing Library.

**Spec:** `docs/superpowers/specs/2026-09-14-kis-semantic-intent-design.md`

## Global Constraints

- `KISIntentResolver` is the only production KIS intent resolver; there is no deterministic fallback.
- The resolver must use a provider-agnostic `LLMClient`; KIS code must not import Qwen/provider-specific classes.
- `.env` selects LLM/text-embedding base URL, API key, model, and timeout. Prompts remain in source control.
- Resolver output is a full semantic graph: entities, event nodes, event/entity bindings, and explicit `BEFORE` edges.
- Current DP remains a strict chronological decoder. Unsupported overlap/simultaneity is merged into one event by the resolver rather than encoded as an unsupported relation.
- Query Preparation runs after intent resolution and may rewrite text only; it may not change event count/order/identity.
- The frontend continues to own ordered KIS clue history; do not add Redis/SQLite mutable search sessions in this phase.
- New abstraction means replacement: migrate every production caller and delete the superseded implementation in the same plan.
- Do not preserve legacy KIS/query parsing merely for hypothetical compatibility. Git history is the fallback for experiments, not dead production code.
- EventTrail anchor/reject/verify actions remain out of scope.

---

## File Structure

### Create

- `src/hcmai/inference/__init__.py` — public inference capability exports.
- `src/hcmai/inference/config.py` — environment-backed endpoint/model configuration.
- `src/hcmai/inference/http.py` — shared authenticated JSON transport.
- `src/hcmai/inference/llm.py` — `LLMClient` protocol + OpenAI-compatible implementation.
- `src/hcmai/inference/embeddings.py` — shared text `EmbeddingClient` protocol + OpenAI-compatible implementation.
- `src/hcmai/kis/__init__.py` — KIS domain exports.
- `src/hcmai/kis/models.py` — semantic graph models + validators.
- `src/hcmai/kis/prompts.py` — version-controlled resolver prompt.
- `src/hcmai/kis/resolver.py` — `KISIntentResolver`.
- `src/hcmai/temporal/events.py` — shared explicit-event normalization after planner removal.
- `src/hcmai/api/routers/kis.py` — canonical revisioned KIS endpoint.
- `frontend/src/api/kis.js` — revisioned KIS API call.
- `frontend/src/features/kis/session.js` — pure live KIS session reducer/helpers.
- `frontend/src/features/kis/components/KisPanel.jsx` — Unified KIS Panel.

### Modify

- `src/hcmai/api/contracts/kis.py`
- `src/hcmai/api/contracts/search.py`
- `src/hcmai/api/contracts/query_candidates.py`
- `src/hcmai/api/contracts/__init__.py`
- `src/hcmai/api/routers/__init__.py`
- `src/hcmai/api/routers/search.py`
- `src/hcmai/api/routers/query_candidates.py`
- `src/hcmai/app.py`
- `src/hcmai/common/config.py`
- `src/hcmai/orchestration/pipeline.py`
- `src/hcmai/orchestration/setup.py`
- `src/hcmai/orchestration/workflows/kis.py`
- `src/hcmai/orchestration/workflows/temporal_search.py`
- `src/hcmai/query_preparation/models.py`
- `src/hcmai/query_preparation/service.py`
- `src/hcmai/query_preparation/__init__.py`
- `src/hcmai/retrieval/embedding/adapters/remote.py`
- `src/hcmai/retrieval/embedding/pipeline.py`
- `src/hcmai/temporal/__init__.py`
- `frontend/.env.example`
- `frontend/src/api/search.js`
- `frontend/src/features/search/components/SearchWorkspace.jsx`
- `frontend/src/features/search/components/SearchWorkspace.test.jsx`
- `frontend/src/features/workspace/queryHistory.js`

### Delete after caller migration

- `src/hcmai/orchestration/workflows/kis_intent.py`
- `src/hcmai/query_preparation/adapters/qwen.py`
- `src/hcmai/temporal/planner.py`

Remove obsolete methods from, but do not necessarily delete wholesale:

- `llm/remote/client.py` — remove query-preparation/text-embedding methods only after no callers remain; retain unrelated OCR/caption/ASR/private capabilities.
- `llm/pipeline.py` — remove query-preparation façade methods after migration.

---

### Task 1: Provider-agnostic LLM and text-embedding clients

**Files:**
- Create: `src/hcmai/inference/config.py`
- Create: `src/hcmai/inference/http.py`
- Create: `src/hcmai/inference/llm.py`
- Create: `src/hcmai/inference/embeddings.py`
- Create: `src/hcmai/inference/__init__.py`
- Modify: `.env.example` or repository backend env example if present
- Test: `tests/inference/test_clients.py`

**Interfaces:**
- Produces: `ModelEndpointConfig`, `LLMClient.generate_structured(...)`, `EmbeddingClient.embed_text(...)`.
- Later tasks depend only on these capability interfaces.

**Legacy impact:**
- This task does not delete old callers yet; Tasks 3 and 4 migrate them first.
- Do not add a second provider-specific KIS client.

- [x] **Step 1: Write failing environment-config tests**

```python
# tests/inference/test_clients.py
import os
import unittest
from unittest.mock import patch

from hcmai.inference.config import load_llm_endpoint, load_embedding_endpoint


class InferenceConfigTest(unittest.TestCase):
    def test_loads_llm_endpoint_from_environment(self) -> None:
        with patch.dict(os.environ, {
            "HCMAI_LLM_BASE_URL": "https://api.example/v1",
            "HCMAI_LLM_API_KEY": "secret",
            "HCMAI_LLM_MODEL": "qwen3-4b",
            "HCMAI_LLM_TIMEOUT_SECONDS": "17",
        }, clear=False):
            config = load_llm_endpoint()
        self.assertEqual(config.base_url, "https://api.example/v1")
        self.assertEqual(config.api_key, "secret")
        self.assertEqual(config.model, "qwen3-4b")
        self.assertEqual(config.timeout_seconds, 17)

    def test_loads_embedding_endpoint_independently(self) -> None:
        with patch.dict(os.environ, {
            "HCMAI_EMBEDDING_BASE_URL": "https://embed.example/v1",
            "HCMAI_EMBEDDING_MODEL": "bge-m3",
        }, clear=False):
            config = load_embedding_endpoint()
        self.assertEqual(config.base_url, "https://embed.example/v1")
        self.assertEqual(config.model, "bge-m3")
```

- [x] **Step 2: Run the tests and verify missing-module failure**

```bash
PYTHONPATH=src python -m unittest tests.inference.test_clients -v
```

Expected: import failure for `hcmai.inference`.

- [x] **Step 3: Implement endpoint configuration**

```python
# src/hcmai/inference/config.py
from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class ModelEndpointConfig:
    base_url: str
    api_key: str | None
    model: str
    timeout_seconds: float


def _load(prefix: str, *, default_timeout: float = 30.0) -> ModelEndpointConfig:
    base_url = os.environ[f"HCMAI_{prefix}_BASE_URL"].rstrip("/")
    model = os.environ[f"HCMAI_{prefix}_MODEL"].strip()
    api_key = os.getenv(f"HCMAI_{prefix}_API_KEY") or None
    timeout = float(os.getenv(f"HCMAI_{prefix}_TIMEOUT_SECONDS", str(default_timeout)))
    if not model:
        raise ValueError(f"HCMAI_{prefix}_MODEL must not be blank")
    if timeout <= 0:
        raise ValueError(f"HCMAI_{prefix}_TIMEOUT_SECONDS must be positive")
    return ModelEndpointConfig(base_url, api_key, model, timeout)


def load_llm_endpoint() -> ModelEndpointConfig:
    return _load("LLM")


def load_embedding_endpoint() -> ModelEndpointConfig:
    return _load("EMBEDDING")
```

- [x] **Step 4: Write failing structured-generation and embedding transport tests**

```python
from pydantic import BaseModel
from unittest.mock import Mock

from hcmai.inference.config import ModelEndpointConfig
from hcmai.inference.llm import LLMClient
from hcmai.inference.embeddings import EmbeddingClient

class Sample(BaseModel):
    value: str


def test_structured_generation_validates_response_model():
    transport = Mock()
    transport.post_json.return_value = {
        "choices": [{"message": {"content": '{"value":"ok"}'}}]
    }
    client = LLMClient(
        ModelEndpointConfig("https://x/v1", "k", "model", 10),
        transport=transport,
    )
    assert client.generate_structured(
        [{"role": "user", "content": "test"}], Sample
    ) == Sample(value="ok")


def test_embedding_client_preserves_input_order():
    transport = Mock()
    transport.post_json.return_value = {
        "data": [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ],
        "model": "embed",
    }
    client = EmbeddingClient(
        ModelEndpointConfig("https://x/v1", "k", "embed", 10),
        transport=transport,
    )
    assert client.embed_text(["a", "b"]).vectors == ((1.0, 0.0), (0.0, 1.0))
```

- [x] **Step 5: Implement shared transport and capability clients**

`LLMClient` provides structured generation. `EmbeddingClient` returns a typed internal batch instead of leaking provider JSON.

```python
# src/hcmai/inference/llm.py
from collections.abc import Sequence
from typing import Protocol, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class LLMClient(Protocol):
    def generate_structured(
        self,
        messages: Sequence[dict[str, str]],
        response_model: type[T],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> T: ...
```

```python
# src/hcmai/inference/embeddings.py
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True, slots=True)
class TextEmbeddingBatch:
    model: str
    vectors: tuple[tuple[float, ...], ...]

class EmbeddingClient(Protocol):
    def embed_text(self, texts: list[str]) -> TextEmbeddingBatch: ...
```

Use `Authorization: Bearer <api_key>` only when an API key is configured. `generate_structured` sends a JSON-schema response format when supported and always validates the returned content with the supplied Pydantic model before returning.

- [x] **Step 6: Run Task 1 tests**

```bash
PYTHONPATH=src python -m unittest tests.inference.test_clients -v
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add src/hcmai/inference tests/inference .env.example
git commit -m "feat: add provider-agnostic inference clients"
```

---

### Task 2: Replace Task-1 deterministic KIS intent with the semantic graph and resolver

**Files:**
- Create: `src/hcmai/kis/models.py`
- Create: `src/hcmai/kis/prompts.py`
- Create: `src/hcmai/kis/resolver.py`
- Create: `src/hcmai/kis/__init__.py`
- Modify: `src/hcmai/api/contracts/kis.py`
- Delete: `src/hcmai/orchestration/workflows/kis_intent.py`
- Test: `tests/kis/test_models.py`
- Test: `tests/kis/test_resolver.py`

**Interfaces:**
- Consumes: `LLMClient` from Task 1.
- Produces: `KISIntentResolver.resolve(inputs: Sequence[str]) -> KISIntent`.
- `KISIntent` becomes a domain model imported by HTTP response contracts.

**Legacy impact:**
- Replaces and deletes `KISIntentBuilder`.
- Removes the old string-list `KISIntent.events` contract.
- No fallback path survives.

- [x] **Step 1: Write semantic graph validation tests**

```python
# tests/kis/test_models.py
import unittest
from pydantic import ValidationError

from hcmai.kis.models import KISIntent


VALID = {
    "revision": 2,
    "inputs": ["A woman is in a kitchen.", "She talks to a man."],
    "language": "en",
    "query_text": "A woman talks to a man in a kitchen.",
    "entities": [
        {"id": "X1", "kind": "person", "description": "woman in a kitchen"},
        {"id": "X2", "kind": "person", "description": "man talking with X1"},
    ],
    "events": [{
        "id": "E1",
        "text": "A woman talks to a man in a kitchen",
        "bindings": [
            {"entity_id": "X1", "role": "speaker"},
            {"entity_id": "X2", "role": "conversation partner"},
        ],
    }],
    "temporal_edges": [],
}

class KISIntentModelTest(unittest.TestCase):
    def test_accepts_resolved_entity_event_graph(self):
        self.assertEqual(KISIntent.model_validate(VALID).events[0].id, "E1")

    def test_rejects_unknown_entity_binding(self):
        invalid = {**VALID, "events": [{
            **VALID["events"][0],
            "bindings": [{"entity_id": "X9", "role": "speaker"}],
        }]}
        with self.assertRaises(ValidationError):
            KISIntent.model_validate(invalid)

    def test_rejects_temporal_edge_against_canonical_order(self):
        invalid = {**VALID,
            "events": [
                {"id":"E1","text":"first","bindings":[]},
                {"id":"E2","text":"second","bindings":[]},
            ],
            "temporal_edges": [{"source":"E2","relation":"before","target":"E1"}],
        }
        with self.assertRaises(ValidationError):
            KISIntent.model_validate(invalid)
```

- [x] **Step 2: Run and verify failure**

```bash
PYTHONPATH=src python -m unittest tests.kis.test_models -v
```

Expected: missing `hcmai.kis.models`.

- [x] **Step 3: Implement graph models and cross-reference validation**

Use `Annotated[str, StringConstraints(...)]` IDs and one `model_validator` on `KISIntent` to check unique IDs, sequential event IDs, binding references, edge references, cycles, and edge/list-order consistency. Limit `events` with `DEFAULT_MAX_TEMPORAL_EVENT_COUNT`.

The public model names must be exactly:

```python
KISEntity
KISEntityBinding
KISEvent
KISTemporalEdge
KISIntent
```

- [x] **Step 4: Write resolver tests with a fake LLM client**

```python
# tests/kis/test_resolver.py
import unittest
from unittest.mock import Mock

from hcmai.kis.models import KISIntent
from hcmai.kis.resolver import KISIntentResolver


class ResolverTest(unittest.TestCase):
    def test_resolver_accepts_reordered_temporal_graph(self):
        llm = Mock()
        llm.generate_structured.return_value = KISIntent(
            revision=2,
            inputs=["A man enters a room.", "Before that, he talks to a woman."],
            language="en",
            query_text="A man talks to a woman before entering a room.",
            entities=[
                {"id":"X1","kind":"person","description":"man"},
                {"id":"X2","kind":"person","description":"woman"},
                {"id":"X3","kind":"place","description":"room"},
            ],
            events=[
                {"id":"E1","text":"A man talks to a woman","bindings":[
                    {"entity_id":"X1","role":"speaker"},
                    {"entity_id":"X2","role":"conversation partner"},
                ]},
                {"id":"E2","text":"The man enters a room","bindings":[
                    {"entity_id":"X1","role":"person entering"},
                    {"entity_id":"X3","role":"destination"},
                ]},
            ],
            temporal_edges=[{"source":"E1","relation":"before","target":"E2"}],
        )
        resolver = KISIntentResolver(llm)
        intent = resolver.resolve([
            "A man enters a room.",
            "Before that, he talks to a woman.",
        ])
        self.assertEqual([event.id for event in intent.events], ["E1", "E2"])
        llm.generate_structured.assert_called_once()

    def test_provider_failure_is_not_hidden_by_deterministic_fallback(self):
        llm = Mock()
        llm.generate_structured.side_effect = RuntimeError("provider down")
        with self.assertRaisesRegex(RuntimeError, "provider down"):
            KISIntentResolver(llm).resolve(["A woman is in a kitchen."])
```

- [x] **Step 5: Implement resolver prompt/profile and `KISIntentResolver`**

`resolver.py` must normalize whitespace, reject empty inputs, and call the LLM once with the entire clue history. Do not call `plan_query_events`.

```python
class KISIntentResolver:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def resolve(self, inputs: Sequence[str]) -> KISIntent:
        normalized = tuple(" ".join(value.split()) for value in inputs)
        if not normalized or any(not value for value in normalized):
            raise ValueError("KIS inputs must contain non-empty text")
        messages = build_kis_intent_messages(normalized)
        intent = self._llm.generate_structured(messages, KISIntent)
        if intent.revision != len(normalized) or intent.inputs != list(normalized):
            raise ValueError("resolver changed KIS revision or raw clue history")
        return intent
```

- [x] **Step 6: Move `KISIntent` ownership out of the API contract and delete the deterministic builder**

`src/hcmai/api/contracts/kis.py` imports `KISIntent` from `hcmai.kis.models`. Delete `src/hcmai/orchestration/workflows/kis_intent.py` after:

```bash
rg "KISIntentBuilder|workflows\.kis_intent" src llm frontend
```

returns no production caller.

- [x] **Step 7: Run Task 2 tests and commit**

```bash
PYTHONPATH=src python -m unittest tests.kis.test_models tests.kis.test_resolver -v
git add src/hcmai/kis src/hcmai/api/contracts/kis.py tests/kis
git rm src/hcmai/orchestration/workflows/kis_intent.py
git commit -m "feat: resolve KIS clues into semantic intent graphs"
```

---

### Task 3: Migrate Query Preparation to `LLMClient` and delete the Qwen-specific runtime adapter

**Files:**
- Modify: `src/hcmai/query_preparation/models.py`
- Modify: `src/hcmai/query_preparation/service.py`
- Modify: `src/hcmai/query_preparation/__init__.py`
- Delete: `src/hcmai/query_preparation/adapters/qwen.py`
- Modify: `src/hcmai/orchestration/setup.py`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Test: `tests/query_preparation/test_service.py`

**Interfaces:**
- Consumes: `LLMClient`.
- Produces: unchanged `QueryCandidateSet` result shape plus `translate_literal(events, language)`.
- KIS resolver remains the only component allowed to alter event semantics.

**Legacy impact:**
- Delete `QueryPreparationAdapter` protocol and `QwenQueryPreparationAdapter`.
- Remove query-preparation methods from `LLMService`/`InferenceClient` in Task 4 after reference audit.

- [x] **Step 1: Write tests that prove query preparation cannot change resolved event cardinality/order**

Use a mock `LLMClient` returning structured translation/candidate payloads. Add an English no-op test so `language="en"` does not pay a translation call.

- [x] **Step 2: Introduce internal Pydantic response models owned by Query Preparation**

```python
class LiteralTranslation(BaseModel):
    events: list[NonBlank]

class CandidateBundle(BaseModel):
    literal_en: list[NonBlank]
    candidates: list[list[NonBlank]]
```

`QueryPreparationService` calls `llm.generate_structured(...)` directly with its own version-controlled prompts.

- [x] **Step 3: Preserve existing invariant validation**

Keep `_validate_bundle` and exact-token preservation. Change only inference ownership; do not relax event-alignment rules.

- [x] **Step 4: Replace setup wiring**

`_load_query_preparation` receives `LLMClient`, not `LLMService/QwenQueryPreparationAdapter`.

- [x] **Step 5: Delete Qwen-specific adapter after zero-reference check**

```bash
rg "QwenQueryPreparationAdapter|QueryPreparationAdapter" src llm
```

Expected after migration: no production references.

```bash
git rm src/hcmai/query_preparation/adapters/qwen.py
```

- [x] **Step 6: Run tests and commit**

```bash
PYTHONPATH=src python -m unittest tests.query_preparation.test_service -v
git commit -am "refactor: make query preparation provider agnostic"
```

---

### Task 4: Migrate text embeddings to the shared `EmbeddingClient` and remove overlapping monolithic inference methods

**Files:**
- Modify: `src/hcmai/retrieval/embedding/adapters/remote.py`
- Modify: `src/hcmai/retrieval/embedding/pipeline.py`
- Modify: `src/hcmai/orchestration/setup.py`
- Modify: `llm/remote/client.py`
- Modify: `llm/pipeline.py`
- Test: `tests/retrieval/embedding/test_remote_adapter.py`

**Interfaces:**
- Consumes: `EmbeddingClient.embed_text` from Task 1.
- Produces: existing `TextEmbeddingAdapter.encode_text(...)` behavior and index-shape/model validation.

**Legacy impact:**
- Delete duplicate `EmbeddingClient` protocol from `retrieval/embedding/adapters/remote.py`.
- Remove `InferenceClient.embed_text` and `LLMService.embed_text` only after all text-embedding callers use the shared client.
- Keep private image embedding methods until a separate `VisionEmbeddingClient` exists; they are a distinct capability, not duplicate legacy.

- [x] **Step 1: Adapt remote text-embedding tests to `TextEmbeddingBatch`**

The test must check count, dimension, finite values, normalization, and configured model identity exactly as the existing adapter does.

- [x] **Step 2: Change `RemoteEmbeddingAdapter` to import `EmbeddingClient`**

Do not move model/index validation into the provider client. Provider client validates wire format; adapter validates HCMAI checkpoint/dimension/normalization semantics.

- [x] **Step 3: Wire `_query_encoder` from `load_embedding_endpoint()`**

Local encoders remain available when configured. Remote text encoding uses `EmbeddingClient` built from `.env` instead of reusing the all-purpose `LLMService` object.

- [x] **Step 4: Remove overlapping monolithic methods after reference audit**

```bash
rg "\.embed_text\(|def embed_text" src llm | sort
```

Remove only the methods made unreachable by the new text client. Do not delete OCR/caption/ASR/image-specific transport.

- [x] **Step 5: Run embedding regression tests and commit**

```bash
PYTHONPATH=src python -m unittest tests.retrieval.embedding.test_remote_adapter -v
git commit -am "refactor: use shared text embedding client"
```

---

### Task 5: Remove raw-query planning and make KISPipeline consume `KISIntent`

**Files:**
- Create: `src/hcmai/temporal/events.py`
- Modify: `src/hcmai/temporal/__init__.py`
- Modify: `src/hcmai/orchestration/workflows/temporal_search.py`
- Modify: `src/hcmai/orchestration/workflows/kis.py`
- Modify: `src/hcmai/api/contracts/query_candidates.py`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Delete: `src/hcmai/temporal/planner.py`
- Test: `tests/orchestration/workflows/test_kis_pipeline.py`
- Test: `tests/temporal/test_events.py`

**Interfaces:**
- Consumes: validated `KISIntent` and meaning-preserving retrieval-event bundle.
- Produces: existing `SearchResult` paths where position `i` maps to `intent.events[i]`.

**Legacy impact:**
- Removes `plan_query_events`/`split_query_events` from production.
- Moves only reusable `normalize_event_texts` to `temporal/events.py`.
- Raw query candidate generation is removed; candidate generation receives explicit resolved events.

- [x] **Step 1: Move explicit-event normalization with unchanged behavior**

Copy the existing `normalize_event_texts` behavior into `temporal/events.py` and update `TemporalSearchService` imports. Write regression tests first.

- [x] **Step 2: Write KIS pipeline test that fails if raw planning is attempted**

Patch/omit any planner entirely; construct a `KISIntent` with two events, call the new pipeline API, and assert `TemporalSearchService.search` receives exactly:

```python
("A woman talks to a man", "The woman takes a white plate")
```

with the retrieval translation bundle aligned by position.

- [x] **Step 3: Replace `KISPipeline.execute(SearchRequest)` with one explicit intent method**

Target signature:

```python
def execute(
    self,
    *,
    intent: KISIntent,
    retrieval_events: Sequence[str],
    use_dense: bool,
    use_bm25: bool,
    top_k: int,
    query_ms: float = 0.0,
) -> KISSearchExecution:
```

`original_events = tuple(event.text for event in intent.events)`.

`caption_events = original_events if use_bm25 else None`.

There must be no import of `plan_query_events` in `workflows/kis.py`.

- [x] **Step 4: Remove raw-query mode from Query Candidates**

`QueryCandidatesRequest` becomes:

```python
class QueryCandidatesRequest(BaseModel):
    events: list[NonBlank] = Field(min_length=1)
    language: Literal["vi", "en"]
```

`SearchService.generate_query_candidates` must no longer call a planner.

- [x] **Step 5: Delete planner after a zero-reference search**

```bash
rg "plan_query_events|split_query_events|temporal\.planner" src llm frontend
```

Expected: no production reference except the file being deleted.

```bash
git rm src/hcmai/temporal/planner.py
```

- [x] **Step 6: Run tests and commit**

```bash
PYTHONPATH=src python -m unittest \
  tests.temporal.test_events \
  tests.orchestration.workflows.test_kis_pipeline -v
git commit -am "refactor: drive KIS temporal search from resolved intent"
```

---

### Task 6: Build the canonical revisioned KIS orchestration/API and delete old text `/api/v1/search`

**Files:**
- Modify: `src/hcmai/api/contracts/kis.py`
- Modify: `src/hcmai/api/contracts/search.py`
- Create: `src/hcmai/api/routers/kis.py`
- Modify: `src/hcmai/api/routers/search.py`
- Modify: `src/hcmai/api/routers/__init__.py`
- Modify: `src/hcmai/app.py`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Modify: `src/hcmai/orchestration/setup.py`
- Test: `tests/api/test_kis_router.py`
- Test: `tests/orchestration/test_kis_revision.py`

**Interfaces:**
- `SearchService.search_kis_revision(request: KISRevisionSearchRequest) -> KISRevisionSearchResponse`.
- `POST /api/v1/kis/search` becomes the only text KIS HTTP route.

**Legacy impact:**
- Delete `SearchRequest` once old text route/callers are migrated.
- Delete `SearchService.search_kis(SearchRequest)`.
- Remove only the text `/api/v1/search` handler; preserve image/filter endpoints.
- Response no longer duplicates `query`, `events`, `revision`, or `inputs` outside `intent`.

- [x] **Step 1: Simplify revisioned response contract**

Target:

```python
class KISRevisionSearchResponse(BaseModel):
    intent: KISIntent
    dense_events: list[NonBlank] | None
    bm25_events: list[NonBlank] | None
    use_dense: bool
    use_bm25: bool
    results: list[SearchResult]
    latency: SearchLatency
```

- [x] **Step 2: Implement orchestration sequence**

`search_kis_revision` must:

1. reject `has_revision_conflict` before inference;
2. resolve all request clues with `KISIntentResolver`;
3. extract canonical event texts;
4. call Query Preparation *after* resolution to obtain literal English dense events (`language="en"` may reuse original texts without a translation request);
5. run `KISPipeline.execute(...)`;
6. return the same `KISIntent` graph used for retrieval.

No planner or deterministic builder is allowed in this method.

- [x] **Step 3: Wire resolver and clients during setup**

`SearchService` constructor receives `intent_resolver: KISIntentResolver` and `query_preparation: QueryPreparationService` explicitly. Do not hide model-client construction inside KIS pipeline classes.

- [x] **Step 4: Add FastAPI error mapping**

- revision conflict -> 409;
- invalid request/graph -> 422;
- LLM provider/network/invalid structured response -> 502 or existing inference-unavailable boundary;
- retrieval unavailable -> 503.

- [x] **Step 5: Preserve DRES logging with canonical intent text**

Use `response.intent.query_text` as the logged query value. Result ranks still use the returned canonical frame results.

- [x] **Step 6: Remove old text search after route tests pass**

```bash
rg "SearchRequest|search_kis\(|/api/v1/search" src frontend
```

Migrate remaining text-KIS callers first. Then remove the text handler and obsolete request type. Do not remove image search/filter functionality that happens to share `search.py`.

- [x] **Step 7: Run backend API tests and commit**

```bash
PYTHONPATH=src python -m unittest \
  tests.api.test_kis_router \
  tests.orchestration.test_kis_revision -v
git commit -am "feat: add semantic revisioned KIS search API"
```

---

### Task 7: Unified KIS Panel and revisioned frontend session

**Files:**
- Create: `frontend/src/api/kis.js`
- Create: `frontend/src/features/kis/session.js`
- Create: `frontend/src/features/kis/session.test.js`
- Create: `frontend/src/features/kis/components/KisPanel.jsx`
- Create: `frontend/src/features/kis/components/KisPanel.test.jsx`
- Modify: `frontend/src/features/search/components/SearchWorkspace.jsx`
- Modify: `frontend/src/features/search/components/SearchWorkspace.test.jsx`
- Modify: `frontend/src/api/search.js`

**Interfaces:**
- Consumes: `POST /api/v1/kis/search` response from Task 6.
- Produces: one live KIS state object independent of Query History.

**Legacy impact:**
- Text KIS no longer uses `searchFrames` or `eventDescription` as the whole session model.
- `api/search.js` retains only non-KIS search operations still used (for example image search).
- Do not fold clue/session state into `activeQuerySession`; that object remains history/replay state until explicitly refactored later.

- [x] **Step 1: Write pure session tests**

Test these transitions:

```text
initial -> submit Q1 -> request revision 1 -> commit
revision 1 -> submit Q2 -> request [Q1,Q2] -> commit revision 2
request failure -> keep revision 1 and keep Q2 as draft
reset -> inputs=[], revision=0, intent=null
```

- [x] **Step 2: Implement `createKisSessionState` and reducer/helpers**

Use explicit immutable state rather than adding more independent `useState` values to `SearchWorkspace`.

- [x] **Step 3: Implement API helper**

`searchKis({ inputs, expectedRevision, useDense, useBm25, topK, userId, signal })` posts to `/api/v1/kis/search` and preserves the DRES user header behavior.

- [x] **Step 4: Implement `KisPanel`**

Required presentation:

- input placeholder `Search or add another clue…`;
- compact committed clue history;
- canonical query text from `intent.query_text`;
- event list `E1..En` from `intent.events`;
- entity list/chips only as lightweight inspectable metadata;
- `New Search` reset action.

No chat-agent prose and no EventTrail action controls in this task.

- [x] **Step 5: Integrate `SearchWorkspace`**

Replace the current text submission branch with the session helper/API. Preserve image search/filter/replay/result rendering. Successful result rendering still uses `FramesBox`.

- [x] **Step 6: Delete old text KIS API call after zero references**

```bash
rg "searchFrames\(" frontend/src
```

If only KIS used it, delete that export and its old tests. Keep `searchFramesByImage` or move it to an image-specific API module if that is the remaining responsibility.

- [x] **Step 7: Run frontend tests and commit**

```bash
cd frontend
npm test -- --runInBand \
  src/features/kis/session.test.js \
  src/features/kis/components/KisPanel.test.jsx \
  src/features/search/components/SearchWorkspace.test.jsx

git add src
 git commit -m "feat: add unified revisioned KIS panel"
```

---

### Task 8: Persist semantic intent in history without making history the live session

**Files:**
- Modify: `frontend/src/features/workspace/queryHistory.js`
- Modify: `frontend/src/features/workspace/queryHistory.test.js`
- Modify: `frontend/src/features/search/components/SearchWorkspace.jsx`
- Modify backend history contract only if current JSON validation rejects the new snapshot field.

**Interfaces:**
- `buildKisSnapshot(results, { intent, latency, warnings, ... })` stores the full semantic graph.
- Replay reads the graph for display but does not mutate live KIS session state.

**Legacy impact:**
- Replace snapshot `events: string[]` as the semantic source of truth with `intent`.
- Remove duplicate event-only snapshot metadata after replay callers use `snapshot.intent.events`.

- [x] **Step 1: Add snapshot test with entity/event graph**

Assert round-trip preservation of:

```text
intent.inputs
intent.entities
intent.events
intent.temporal_edges
```

and existing result frame/timestamp metadata.

- [x] **Step 2: Modify snapshot normalization**

Validate intent as an object and retain it unchanged apart from defensive cloning. Stop separately requiring a top-level `events` field.

- [x] **Step 3: Update Replay consumers and remove duplicate `events` snapshot usage**

Search with:

```bash
rg "result_snapshot.*events|snapshot\.events|\.events" frontend/src/features/workspace frontend/src/features/search
```

Migrate only KIS-history semantics; do not touch unrelated event arrays in temporal exploration.

- [x] **Step 4: Run history/replay tests and commit**

```bash
cd frontend
npm test -- --runInBand \
  src/features/workspace/queryHistory.test.js \
  src/features/search/components/SearchWorkspace.test.jsx

git commit -am "feat: persist KIS semantic intent in query history"
```

---

### Task 9: Legacy-removal audit, configuration documentation, and regression gate

**Files:**
- Modify: backend `.env.example`/deployment docs available in the repository
- Modify: `frontend/.env.example` only if frontend KIS URL configuration changes
- Modify: README/runbook files that still document `/api/v1/search` as text KIS
- No new runtime abstraction in this task.

**Interfaces:** none; this is the completion gate.

**Legacy impact:** this task proves the replacement map is complete.

- [ ] **Step 1: Run mandatory zero-reference audits**

Each command must return no production matches (tests documenting deletion may be excluded explicitly):

```bash
rg "KISIntentBuilder|workflows\.kis_intent" src llm frontend
rg "plan_query_events|split_query_events" src llm frontend
rg "QwenQueryPreparationAdapter|QueryPreparationAdapter" src llm frontend
rg "SearchRequest" src/hcmai frontend/src
rg '"/api/v1/search"|`/api/v1/search`|/api/v1/search' frontend/src src/hcmai
```

Any remaining match must be classified as a current independent responsibility or removed; do not leave compatibility shims without a named external consumer.

- [ ] **Step 2: Audit duplicate inference HTTP ownership**

```bash
rg "translate_query_events|generate_query_candidates|def embed_text" src llm
```

Expected final state:

- query preparation calls `LLMClient`;
- text query embeddings call `EmbeddingClient`;
- remaining methods in the private inference service are only capabilities not replaced by those clients.

- [ ] **Step 3: Document environment switching**

Document one self-host example and one third-party example using the same variables:

```env
HCMAI_LLM_BASE_URL=https://api.iamphuckhang.dev/v1
HCMAI_LLM_API_KEY=...
HCMAI_LLM_MODEL=Qwen/Qwen3-4B

HCMAI_EMBEDDING_BASE_URL=https://api.iamphuckhang.dev/v1
HCMAI_EMBEDDING_API_KEY=...
HCMAI_EMBEDDING_MODEL=google/siglip2-base-patch16-224
```

Changing providers must not require editing KIS/query-preparation source.

- [ ] **Step 4: Run backend regression suite**

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

At minimum verify KIS semantic resolver, query preparation, temporal DP, TRAKE, image search, and history-facing contracts.

- [ ] **Step 5: Run frontend regression suite**

```bash
cd frontend
npm test -- --runInBand
```

- [ ] **Step 6: Manual acceptance smoke test**

Use these three cases:

1. KIS-T one-shot: `A woman talks to a man in a kitchen, then takes a white plate.`
2. KIS-C revision 1: `A man enters a room.`; revision 2: `Before that, he talks to a woman.` Verify UI event order becomes talk -> enter.
3. Correction: first clue says `red shirt`; later clue says `actually orange, not red`. Verify canonical intent/event text contains the corrected color and raw clue history still contains both clues.

For each case verify returned `frame_ids` length equals `len(intent.events)`.

- [ ] **Step 7: Commit completion gate**

```bash
git add .
git commit -m "chore: complete semantic KIS migration and remove legacy paths"
```

---

## Self-review results

### Spec coverage

- Provider-switchable LLM/text embedding: Tasks 1, 3, 4, 9.
- Full semantic graph Decision C: Task 2.
- No deterministic fallback: Task 2 tests and deletion.
- Resolver before query preparation: Tasks 3, 6.
- DP consumes resolved ordered events only: Task 5.
- Unified revisioned KIS API and panel: Tasks 6, 7.
- History separation: Task 8.
- Legacy removal rule: every task contains a `Legacy impact` section; Task 9 is the final zero-reference gate.
- EventTrail remains excluded.

### Type consistency

- `KISIntentResolver.resolve(...) -> KISIntent` is the only KIS semantic-resolution interface.
- `KISIntent.events[i].text` is the canonical DP/BM25 event string.
- Dense retrieval rewrites remain positional arrays matching `KISIntent.events` exactly.
- `KISRevisionSearchResponse.intent` is the only public query/event source of truth.

### Intentional non-goals

- No DB-backed mutable KIS session.
- No entity-consistency visual verifier yet; entity IDs represent resolver semantics only.
- No arbitrary temporal graph decoder; current graph normalizes to a strict `BEFORE` chain.
- No EventTrail actions.
