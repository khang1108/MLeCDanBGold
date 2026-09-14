# KIS Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize the unified revisioned KIS flow from generic provider inference through semantic intent resolution, aligned retrieval/DP, and transactional frontend state, while removing superseded query-expansion/query-preparation code before EventTrail work starts.

**Architecture:** Raw KIS clues are resolved once by `KISIntentResolver` into model-owned `KISResolution`, then canonicalized by the server into `KISIntent`. Retrieval derives an immutable `KISRetrievalPlan` from that intent; `EventTranslator` may translate each event one-to-one for Dense search, while BM25 keeps canonical text. The existing temporal scorer and DP remain the alignment engine. Frontend revisions are transactional: pending clues never destroy the last committed intent/results/exploration state.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, httpx, Transformers/PyTorch for self-hosted Qwen text generation, existing retrieval/DP stack, React 19 + react-scripts/Jest.

**Spec:** `docs/vbs/2026-09-14-kis-stabilization-design.md`

## Global Constraints

- This plan targets the current `vbs` branch (post `feat/ui-vbs` merge, post KIS semantic-intent migration commit `be790e0`). The prior semantic-intent plan's legacy deletions (`KISIntentBuilder`, `plan_query_events`, `split_query_events`, legacy raw-query path) are already complete and do not need repeating.
- Do **not** add EventTrail, anchor/reject/replace actions, VLM temporal verification, graph-aware decoding, entity re-identification, or query expansion/fusion in this batch.
- `KISIntentResolver` is the only production KIS intent-construction path. There is no deterministic fallback.
- The LLM owns semantic resolution only; the server owns raw inputs, revision, `Xn`/`En` IDs, entity bindings, and the complete adjacent `before` chain.
- Current DP semantics remain a strict ordered chain. The resolver must emit events in chronological order; overlapping actions are merged into one event until a richer decoder exists.
- `KISIntent` is immutable semantic truth. Translation/retrieval preparation may not change event cardinality, order, IDs, bindings, entities, or graph meaning.
- Query expansion/candidate generation is deleted rather than deprecated.
- New abstractions must replace old ones in the same plan: trace callers, migrate them, then delete obsolete symbols/routes/packages.
- `HCMAI_LLM_BASE_URL` and `HCMAI_EMBEDDING_BASE_URL` are API roots including `/v1`.
- `HCMAI_EMBEDDING_MODE` explicitly selects `remote` or `local`; mode is never inferred from endpoint presence or a caught exception.
- A configured remote embedding/LLM failure must fail visibly; no silent remote-to-local execution switch is allowed.
- KIS revision state is transactional. A pending revision never clears the last committed intent/results/exploration state.
- History persistence is best-effort and outside the live search critical path.
- Existing temporal exploration must continue to work during stabilization, but it must consume a dedicated canonical exploration seed rather than reconstructing state from top-level `dense_events`/`bm25_events` debug fields.

---

## File Structure / Target Ownership

### Create

- `llm/local/text_generation.py` — process-local Qwen text generator used by the generic generation route.
- `llm/server/routers/generation.py` — OpenAI-compatible `POST /v1/chat/completions` subset.
- `src/hcmai/inference/errors.py` — provider/client error taxonomy.
- `src/hcmai/retrieval/translation/__init__.py`
- `src/hcmai/retrieval/translation/cache.py`
- `src/hcmai/retrieval/translation/models.py`
- `src/hcmai/retrieval/translation/prompts.py`
- `src/hcmai/retrieval/translation/service.py` — `EventTranslator` only.
- `src/hcmai/retrieval/kis_plan.py` — aligned `KISRetrievalPlan` / `RetrievalEvent` models and builder.
- `tests/llm/test_generation_router.py`
- `tests/llm/test_embedding_router.py`
- `tests/hcmai/inference/test_clients.py`
- `tests/hcmai/kis/test_resolver.py`
- `tests/hcmai/retrieval/test_translation.py`
- `tests/hcmai/retrieval/test_kis_plan.py`
- `tests/hcmai/orchestration/test_kis_revision.py`
- `tests/hcmai/orchestration/test_readiness.py`
- root `.gitignore` if the actual repository does not already provide equivalent rules.

### Modify

- `llm/config.py`
- `llm/config.yaml`
- `llm/contracts.py`
- `llm/local/adapter.py`
- `llm/local/readiness.py`
- `llm/pipeline.py`
- `llm/server/routers/__init__.py`
- `llm/server/routers/embeddings.py`
- `llm/scripts/deploy_cloudflared_private.sh`
- `llm/scripts/run_local.sh`
- `llm/README.md`
- `src/hcmai/inference/config.py`
- `src/hcmai/inference/http.py`
- `src/hcmai/inference/llm.py`
- `src/hcmai/inference/embeddings.py`
- `src/hcmai/inference/__init__.py`
- `src/hcmai/kis/models.py`
- `src/hcmai/kis/prompts.py`
- `src/hcmai/kis/resolver.py`
- `src/hcmai/common/config.py`
- `src/hcmai/orchestration/setup.py`
- `src/hcmai/orchestration/retrieval_setup.py`
- `src/hcmai/orchestration/pipeline.py`
- `src/hcmai/orchestration/health.py`
- `src/hcmai/orchestration/workflows/kis.py`
- `src/hcmai/api/contracts/kis.py`
- `src/hcmai/api/contracts/latency.py`
- `src/hcmai/api/contracts/__init__.py`
- `src/hcmai/api/routers/kis.py`
- `src/hcmai/api/routers/__init__.py`
- `src/hcmai/app.py`
- `frontend/src/api/kis.js`
- `frontend/src/api/kis.test.js`
- `frontend/src/features/kis/session.js`
- `frontend/src/features/kis/session.test.js`
- `frontend/src/features/kis/components/KisPanel.jsx`
- `frontend/src/features/kis/components/KisPanel.test.jsx`
- `frontend/src/features/search/components/SearchWorkspace.jsx`
- `frontend/src/features/search/components/SearchWorkspace.test.jsx`
- `frontend/src/features/alignment/hooks/useTemporalExploration.js`
- `frontend/src/features/alignment/hooks/useTemporalExploration.test.js`
- `frontend/src/features/workspace/queryHistory.js`
- `frontend/src/features/workspace/queryHistory.test.js`
- `frontend/src/App.jsx`
- `frontend/src/App.test.jsx`
- affected KIS/chat CSS in the existing style file that currently defines `.kis-chat-*` classes.

### Delete after caller migration

- `src/hcmai/query_preparation/__init__.py`
- `src/hcmai/query_preparation/cache.py`
- `src/hcmai/query_preparation/models.py`
- `src/hcmai/query_preparation/service.py`
- `src/hcmai/api/contracts/query_candidates.py`
- `src/hcmai/api/routers/query_candidates.py`
- obsolete exports/registration for `/api/v1/query-candidates`.
- any deployed inference-server `/query-preparation/translate` and `/query-preparation/candidates` source discovered during deployment reconciliation after `/v1/chat/completions` is verified.
- generated `__pycache__/`, `*.pyc`, `.pytest_cache/`, `frontend/build/` from source tracking/snapshots.

---

## Task 1: Add Generic OpenAI-Compatible Text Generation to the Self-Hosted Inference Service

**Files:**
- Create: `llm/local/text_generation.py`
- Create: `llm/server/routers/generation.py`
- Modify: `llm/config.py`
- Modify: `llm/config.yaml`
- Modify: `llm/contracts.py`
- Modify: `llm/local/adapter.py`
- Modify: `llm/local/readiness.py`
- Modify: `llm/pipeline.py`
- Modify: `llm/server/routers/__init__.py`
- Modify: `llm/scripts/deploy_cloudflared_private.sh`
- Modify: `llm/scripts/run_local.sh`
- Modify: `llm/README.md`
- Test: `tests/llm/test_generation_router.py`

**Interfaces:**
- Produces: `HostedTextGenerationConfig`, `TextGenerationAdapter.generate(messages, response_schema, temperature, max_tokens) -> str`, `LLMService.generate_chat(...) -> str`, and `POST /v1/chat/completions`.
- Consumers: `src/hcmai/inference/llm.py` from Task 2 and all domain LLM services.

**Legacy impact:**
- Replaces the deployed task-specific query-preparation generation capability as the generic text-generation entry point.
- Do not delete deployed `/query-preparation/*` routes until this generic route passes the contract test and the deployed checkout is searched for their actual source.
- Rename deployment capability flag from `HCMAI_ENABLE_QUERY_PREPARATION` to `HCMAI_ENABLE_TEXT_GENERATION` in this repository.

- [ ] **Step 1: Write the failing generation-router tests**

Create `tests/llm/test_generation_router.py` with an injectable fake runtime so the route can be tested without loading Qwen:

```python
from fastapi.testclient import TestClient

from llm.server.app import create_llm_app


class FakeRuntime:
    def load(self):
        pass

    def close(self):
        pass

    def readiness(self):
        class Model:
            loaded = True
            checkpoint = "Qwen/Qwen3-4B"
            revision = "rev"

        class Ready:
            ready = True
            models = {"text_generation": Model()}

        return Ready()

    def generate_chat(self, messages, *, response_schema, temperature, max_tokens):
        assert messages[-1]["content"] == "resolve this"
        assert response_schema["type"] == "object"
        return '{"answer":"ok"}'


def test_chat_completions_returns_openai_shape():
    client = TestClient(create_llm_app(FakeRuntime()))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-4B",
            "messages": [{"role": "user", "content": "resolve this"}],
            "temperature": 0,
            "max_tokens": 128,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Answer",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "Qwen/Qwen3-4B"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == '{"answer":"ok"}'


def test_chat_completions_rejects_unready_text_model():
    runtime = FakeRuntime()
    runtime.readiness = lambda: type(
        "Ready",
        (),
        {"models": {"text_generation": type("Model", (), {"loaded": False})()}},
    )()
    client = TestClient(create_llm_app(runtime))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-4B",
            "messages": [{"role": "user", "content": "x"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "X", "schema": {"type": "object"}},
            },
        },
    )
    assert response.status_code == 503
```

- [ ] **Step 2: Run the router tests and verify they fail**

Run:

```bash
PYTHONPATH=src:. python -m pytest tests/llm/test_generation_router.py -v
```

Expected: FAIL because `/v1/chat/completions` and its contracts do not exist.

- [ ] **Step 3: Add typed hosted text-generation configuration**

In `llm/config.py`, add:

```python
class HostedTextGenerationConfig(BaseModel):
    checkpoint: str = "Qwen/Qwen3-4B"
    revision: str | None = "1cfa9a7208912126459214e8b04321603b3df60c"
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_input_tokens: int = Field(default=4096, ge=256)
    max_new_tokens: int = Field(default=2048, ge=64)


class LLMServiceConfig(BaseModel):
    ...
    text_generation: HostedTextGenerationConfig = Field(
        default_factory=HostedTextGenerationConfig
    )
```

Add the matching `text_generation:` section to `llm/config.yaml`.

- [ ] **Step 4: Implement the process-local text generator**

Create `llm/local/text_generation.py` with one lazy-loaded owner. It must not know KIS or translation prompts:

```python
import json
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.config import HostedTextGenerationConfig


class TextGenerationAdapter:
    def __init__(self, config: HostedTextGenerationConfig) -> None:
        self.config = config
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.checkpoint,
            revision=self.config.revision,
        )
        dtype = getattr(torch, self.config.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.checkpoint,
            revision=self.config.revision,
            torch_dtype=dtype,
        ).to(self.config.device)
        self.model.eval()

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        response_schema: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str:
        self.load()
        assert self.tokenizer is not None and self.model is not None
        schema = json.dumps(response_schema, ensure_ascii=False, separators=(",", ":"))
        augmented = list(messages) + [{
            "role": "system",
            "content": (
                "Return only one JSON object matching this JSON Schema exactly. "
                f"Do not use markdown fences. Schema: {schema}"
            ),
        }]
        prompt = self.tokenizer.apply_chat_template(
            augmented,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_tokens,
        ).to(self.model.device)
        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": min(max_tokens, self.config.max_new_tokens),
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
        output = self.model.generate(**inputs, **generation_kwargs)
        generated = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
```

Do not parse domain schemas in this adapter; the HCMAI backend remains the domain validator.

- [ ] **Step 5: Wire `LocalAdapter` and readiness to the text generator**

In `llm/local/adapter.py`:

```python
self.enable_text_generation = enable_text_generation
self.text_generator = (
    TextGenerationAdapter(config.text_generation)
    if enable_text_generation
    else None
)
```

Load it inside `load()` and expose:

```python
def generate_chat(self, messages, *, response_schema, temperature, max_tokens):
    if self.text_generator is None:
        raise RuntimeError("text generation model is disabled")
    return self.text_generator.generate(
        messages,
        response_schema=response_schema,
        temperature=temperature,
        max_tokens=max_tokens,
    )
```

In `from_environment()`, read `HCMAI_ENABLE_TEXT_GENERATION`.

In `llm/local/readiness.py`, add model status `text_generation`, include it in `ready`, and set `structured_parsing=text_generation_loaded`.

- [ ] **Step 6: Add generic chat-completion contracts and router**

In `llm/contracts.py`, add strict models for the supported OpenAI subset:

```python
class ChatMessage(_HTTPContract):
    role: Literal["system", "user", "assistant"]
    content: _NonEmptyString


class JsonSchemaSpec(_HTTPContract):
    name: _NonEmptyString
    schema_: dict[str, Any] = Field(alias="schema")


class ResponseFormat(_HTTPContract):
    type: Literal["json_schema"]
    json_schema: JsonSchemaSpec


class ChatCompletionRequest(_HTTPContract):
    model: _NonEmptyString
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1)
    response_format: ResponseFormat
```

Create `llm/server/routers/generation.py` and return an OpenAI-compatible response containing `model`, `choices[0].message.role`, and `choices[0].message.content`. Reject a requested `model` that differs from the loaded checkpoint rather than silently executing another model.

Register the router in `llm/server/routers/__init__.py`.

- [ ] **Step 7: Delegate generation through `LLMService`**

In `llm/pipeline.py`:

```python
def generate_chat(self, messages, *, response_schema, temperature, max_tokens):
    method = getattr(self.adapter, "generate_chat", None)
    if method is None:
        raise RuntimeError("text generation is not supported by this provider")
    return method(
        messages,
        response_schema=response_schema,
        temperature=temperature,
        max_tokens=max_tokens,
    )
```

- [ ] **Step 8: Update deployment flags/docs and reconcile deployed query-preparation routes**

Replace `HCMAI_ENABLE_QUERY_PREPARATION` / `--query-preparation` in the repository scripts with `HCMAI_ENABLE_TEXT_GENERATION` / `--text-generation`. Update `llm/README.md` to document `/v1/chat/completions`.

Reconcile the deployed inference server. The deployed checkout may contain query-preparation routes (`/query-preparation/translate`, `/query-preparation/candidates`) whose source does not exist in this repository. Run on the **deployed checkout** (SSH to inference host or check its local git repo):

```bash
rg -n "query-preparation|query_preparation|ENABLE_QUERY_PREPARATION" llm src
```

If the deployed checkout is not accessible during this task, record this as a deferred action for Task 4 and proceed — the generic route can be verified independently.

Do not delete the deployed task-specific router yet; record its exact source path in the implementation notes for Task 4. If the deployed checkout contains source not present in this repository, port the generic generation changes into that checkout before route removal.

- [ ] **Step 9: Run generation tests**

Run:

```bash
PYTHONPATH=src:. python -m pytest tests/llm/test_generation_router.py -v
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add llm tests/llm/test_generation_router.py
git commit -m "feat: expose generic text generation API"
```

---

## Task 2: Normalize Provider Clients, Model Identity, and Inference Errors

**Files:**
- Create: `src/hcmai/inference/errors.py`
- Modify: `src/hcmai/inference/http.py`
- Modify: `src/hcmai/inference/llm.py`
- Modify: `src/hcmai/inference/embeddings.py`
- Modify: `src/hcmai/inference/__init__.py`
- Test: `tests/hcmai/inference/test_clients.py`

**Interfaces:**
- Produces: `InferenceError`, `InferenceUnavailableError`, `InferenceAuthError`, `InferenceResponseError`; `LLMClient.model`; `EmbeddingClient.model`.
- Consumers: `KISIntentResolver`, `EventTranslator`, setup/readiness, and router error mapping.

**Legacy impact:**
- Replaces raw `httpx`/`ValueError` leakage from capability clients.
- Cache identity in later tasks must use `client.model`, never the old query-preparation model config.

- [ ] **Step 1: Write failing client contract tests**

Create `tests/hcmai/inference/test_clients.py` using a fake `HttpTransport` that records URL/payload and raises typed transport errors. Cover:

```python
def test_llm_client_appends_chat_completions_and_exposes_model(): ...
def test_llm_client_preserves_json_schema_payload(): ...
def test_llm_client_maps_invalid_content_to_response_error(): ...
def test_embedding_client_appends_embeddings_and_preserves_input_order(): ...
def test_auth_failure_is_not_reported_as_unavailable(): ...
```

The first assertion must include:

```python
assert transport.calls[0].url == "https://api.example/v1/chat/completions"
assert client.model == "Qwen/Qwen3-4B"
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=src:. python -m pytest tests/hcmai/inference/test_clients.py -v
```

Expected: FAIL because typed errors/model properties are absent.

- [ ] **Step 3: Add the small inference error taxonomy**

Create `src/hcmai/inference/errors.py`:

```python
class InferenceError(RuntimeError):
    pass


class InferenceUnavailableError(InferenceError):
    pass


class InferenceAuthError(InferenceError):
    pass


class InferenceResponseError(InferenceError):
    pass
```

- [ ] **Step 4: Make `HttpTransport` translate HTTP/network errors once**

In `src/hcmai/inference/http.py`, catch:

- timeout/connect/network -> `InferenceUnavailableError`;
- HTTP 401/403 -> `InferenceAuthError`;
- other non-2xx or invalid JSON body -> `InferenceResponseError`.

Keep the transport unaware of KIS/translation semantics.

- [ ] **Step 5: Add model identity and wrap malformed provider output**

In both clients add:

```python
@property
def model(self) -> str:
    return self._endpoint.model
```

`LLMClient.generate_structured()` must convert empty choices, missing content, malformed JSON, and schema-validation failures into `InferenceResponseError` while preserving the original exception as `__cause__`.

`EmbeddingClient.embed_text()` must convert malformed count/index/vector responses into `InferenceResponseError`.

- [ ] **Step 6: Export errors and run tests**

Update `src/hcmai/inference/__init__.py`, then run:

```bash
PYTHONPATH=src:. python -m pytest tests/hcmai/inference/test_clients.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/hcmai/inference tests/hcmai/inference/test_clients.py
git commit -m "refactor: normalize inference client errors"
```

---

## Task 3: Separate Model-Owned `KISResolution` from Server-Owned `KISIntent`

**Files:**
- Modify: `src/hcmai/kis/models.py`
- Modify: `src/hcmai/kis/prompts.py`
- Modify: `src/hcmai/kis/resolver.py`
- Modify: `src/hcmai/kis/__init__.py`
- Test: `tests/hcmai/kis/test_resolver.py`

**Interfaces:**
- Consumes: `LLMClient.generate_structured(..., KISResolution)` from Task 2.
- Produces: canonical `KISIntent` with exact normalized `inputs`, deterministic `revision`, `Xn`/`En` IDs, bindings, and complete adjacent `before` chain.

**Legacy impact:**
- Replaces direct `LLMClient.generate_structured(..., KISIntent)` in `resolver.py`.
- Removes model ownership of revision, raw clues, canonical IDs, and temporal edges.

- [ ] **Step 1: Write resolver tests for semantic-only model output**

Create `tests/hcmai/kis/test_resolver.py` with a fake LLM that returns `KISResolution`. Include tests for:

```python
Q1 = "A woman is standing in a kitchen."
Q2 = "She is talking to a man."
Q3 = "Before taking a white plate, they move to the left."
```

Expected canonical state:

```python
assert intent.revision == 3
assert intent.inputs == [Q1, Q2, Q3]
assert [event.id for event in intent.events] == ["E1", "E2", "E3"]
assert [(e.source, e.target) for e in intent.temporal_edges] == [
    ("E1", "E2"),
    ("E2", "E3"),
]
assert [entity.id for entity in intent.entities] == ["X1", "X2", "X3"]
```

Also test:

- out-of-range entity index -> `KISResolutionError`;
- duplicate entity index in one event -> `KISResolutionError`;
- too many events -> `KISResolutionError`;
- blank client clue -> `ValueError` before calling the model;
- provider unavailable propagates as inference-unavailable, not 422 semantics.

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=src:. python -m pytest tests/hcmai/kis/test_resolver.py -v
```

Expected: FAIL because `KISResolution` and server canonicalization do not exist.

- [ ] **Step 3: Add semantic-only resolution models**

In `src/hcmai/kis/models.py`, add:

```python
class KISResolutionEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["person", "object", "place", "text", "other"]
    description: NonBlank


class KISResolutionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: NonBlank
    entity_indices: list[int] = Field(default_factory=list)


class KISResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    language: Literal["vi", "en"]
    query_text: NonBlank
    entities: list[KISResolutionEntity] = Field(default_factory=list)
    events: list[KISResolutionEvent] = Field(min_length=1)
```

Strengthen `KISIntent.validate_graph()` so a multi-event intent requires exactly the adjacent chain derived by the server, not an arbitrary partial edge set.

- [ ] **Step 4: Change the prompt to request only semantic content**

In `src/hcmai/kis/prompts.py`, remove instructions asking for `revision`, `inputs`, `E*`/`X*` IDs, or temporal edges. Explicitly require:

```text
- Resolve pronouns/coreference.
- Apply later corrections instead of concatenating contradictions.
- Merge simultaneous descriptions of the same event.
- Split genuinely sequential actions.
- Return events in chronological order even when clue order says "before that".
- entity_indices are zero-based references into entities[].
- Do not emit timestamps, candidate IDs, retrieval translations, event IDs, entity IDs, or edges.
```

- [ ] **Step 5: Canonicalize inside `KISIntentResolver`**

Add `KISResolutionError(RuntimeError)` and implement:

```python
resolution = self._llm.generate_structured(messages, KISResolution)
entities = [
    KISEntity(id=f"X{i+1}", kind=e.kind, description=e.description)
    for i, e in enumerate(resolution.entities)
]

events = []
for event_index, event in enumerate(resolution.events):
    if len(set(event.entity_indices)) != len(event.entity_indices):
        raise KISResolutionError("event contains duplicate entity indices")
    bindings = []
    for entity_index in event.entity_indices:
        if not 0 <= entity_index < len(entities):
            raise KISResolutionError("event references an out-of-range entity index")
        bindings.append(
            KISEntityBinding(
                entity_id=entities[entity_index].id,
                role="participant",
            )
        )
    events.append(KISEvent(id=f"E{event_index+1}", text=event.text, bindings=bindings))

edges = [
    KISTemporalEdge(source=f"E{i}", target=f"E{i+1}")
    for i in range(1, len(events))
]
```

Construct `KISIntent` with server-owned `revision=len(normalized)` and exact normalized inputs.

Wrap valid provider output that violates the semantic contract in `KISResolutionError`; do not convert provider-unavailable/auth failures to that domain error.

- [ ] **Step 6: Run resolver tests**

```bash
PYTHONPATH=src:. python -m pytest tests/hcmai/kis/test_resolver.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/hcmai/kis tests/hcmai/kis/test_resolver.py
git commit -m "refactor: canonicalize KIS semantic resolutions"
```

---

## Task 4: Replace Query Preparation with `EventTranslator` and Delete Query Expansion

**Files:**
- Create: `src/hcmai/retrieval/translation/__init__.py`
- Create: `src/hcmai/retrieval/translation/cache.py`
- Create: `src/hcmai/retrieval/translation/models.py`
- Create: `src/hcmai/retrieval/translation/prompts.py`
- Create: `src/hcmai/retrieval/translation/service.py`
- Modify: `src/hcmai/common/config.py`
- Modify: `src/hcmai/api/contracts/__init__.py`
- Modify: `src/hcmai/api/routers/__init__.py`
- Modify: `src/hcmai/app.py`
- Modify: `src/hcmai/orchestration/setup.py`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Modify: `src/hcmai/orchestration/health.py`
- Delete: `src/hcmai/query_preparation/`
- Delete: `src/hcmai/api/contracts/query_candidates.py`
- Delete: `src/hcmai/api/routers/query_candidates.py`
- Test: `tests/hcmai/retrieval/test_translation.py`

**Interfaces:**
- Produces: `EventTranslator.translate(events, language) -> tuple[str, ...]`, `EventTranslationError`.
- Consumers: KIS retrieval-plan builder in Task 5.

**Legacy impact:**
- Deletes `QueryPreparationService`, `generate_candidates`, candidate models, `/api/v1/query-candidates`, and query-expansion config.
- Moves only one-to-one translation behavior under retrieval ownership.

- [ ] **Step 1: Write failing translation tests**

Create `tests/hcmai/retrieval/test_translation.py` covering:

```python
def test_english_events_bypass_llm(): ...
def test_vietnamese_translation_preserves_event_count_and_order(): ...
def test_translation_preserves_required_uppercase_tokens(): ...
def test_cache_identity_changes_when_llm_model_changes(): ...
def test_invalid_translation_raises_event_translation_error(): ...
```

The cache test must construct two fake clients with different `.model` values and assert no cache reuse across models.

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=src:. python -m pytest tests/hcmai/retrieval/test_translation.py -v
```

Expected: FAIL because `hcmai.retrieval.translation` does not exist.

- [ ] **Step 3: Create the focused translation package**

Move the existing literal-translation validator/cache logic into the new package. `models.py` contains only:

```python
class LiteralTranslation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    events: list[NonBlank]
```

`prompts.py` owns the one-to-one translation system prompt. `cache.py` builds keys from:

```python
(operation, llm_model, prompt_version, *normalized_events)
```

There is no model revision copied from `QueryPreparationConfig` and no candidate-generation operation.

- [ ] **Step 4: Implement `EventTranslator`**

In `service.py`:

```python
class EventTranslationError(RuntimeError):
    pass


class EventTranslator:
    def __init__(self, llm: LLMClient, config: EventTranslationConfig, cache=None):
        ...

    def translate(self, events: Sequence[str], *, language: str) -> tuple[str, ...]:
        normalized = normalize_event_texts(events)
        if language == "en":
            return normalized
        key = cache_key(
            operation="translate",
            events=normalized,
            model_name=self._llm.model,
            prompt_version=self._config.prompt_version,
        )
        ...
```

Preserve exact event count/order and required uppercase tokens. Wrap semantic translation-contract violations as `EventTranslationError`; provider-unavailable/auth failures retain their inference error type.

- [ ] **Step 5: Replace `QueryPreparationConfig` with translation-only config**

In `src/hcmai/common/config.py`, replace:

```python
class QueryPreparationConfig(...):
    model_name
    model_revision
    candidate_count
    ...
```

with:

```python
class EventTranslationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt_version: str = Field(default="event-translation-v1", min_length=1)
    cache_enabled: bool = True
    cache_ttl_seconds: float = Field(default=3600, gt=0)
    cache_max_entries: int = Field(default=2048, ge=1)
```

Rename the `AppConfig` field from `query_preparation` to `event_translation` and migrate any repository YAML that contains the old key.

- [ ] **Step 6: Migrate setup/service ownership before deletion**

In `orchestration/setup.py`, replace `_load_query_preparation()` with `_load_event_translator()` and pass `event_translator` into `SearchService`.

In `SearchService`, rename the member and constructor argument; remove `generate_query_candidates()` completely.

Update health capability name from `query_preparation` to `event_translation`.

- [ ] **Step 7: Delete the query-expansion HTTP/domain chain**

Delete:

```text
src/hcmai/api/contracts/query_candidates.py
src/hcmai/api/routers/query_candidates.py
src/hcmai/query_preparation/
```

Remove their imports/exports/router registration from `contracts/__init__.py`, `routers/__init__.py`, `app.py`, and orchestration modules.

Run:

```bash
rg -n "QueryPreparationService|generate_candidates|QueryCandidate|CandidateBundle|query-candidates|query_preparation" src/hcmai
```

Expected: no production references. Any remaining match must be a migration doc that is updated in Task 10.

- [ ] **Step 8: Remove deployed task-specific inference routes after generic generation is proven**

On the deployed inference checkout identified in Task 1 Step 8, remove `/query-preparation/translate` and `/query-preparation/candidates` router/contracts after confirming no other consumer through access logs or repository search. If the deployed checkout was not accessible during Task 1, this step must SSH to the inference host or reference its local git repo. If still not accessible, record as a deferred deployment action and continue — this does not block HCMAI backend stabilization. Do not add an HCMAI-specific client adapter.

- [ ] **Step 9: Run tests**

```bash
PYTHONPATH=src:. python -m pytest tests/hcmai/retrieval/test_translation.py -v
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/hcmai tests/hcmai/retrieval/test_translation.py
git commit -m "refactor: replace query preparation with event translation"
```

---

## Task 5: Introduce `KISRetrievalPlan`, Dedicated Exploration Seed, and Correct Latency Ownership

**Files:**
- Create: `src/hcmai/retrieval/kis_plan.py`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Modify: `src/hcmai/orchestration/workflows/kis.py`
- Modify: `src/hcmai/api/contracts/kis.py`
- Modify: `src/hcmai/api/contracts/latency.py`
- Modify: `src/hcmai/api/routers/kis.py`
- Test: `tests/hcmai/retrieval/test_kis_plan.py`
- Test: `tests/hcmai/orchestration/test_kis_revision.py`

**Interfaces:**
- Consumes: canonical `KISIntent`, optional `EventTranslator`, retrieval-source flags.
- Produces: `KISRetrievalPlan`, `KISExplorationSeed`, stage-separated KIS latency.
- Existing `TemporalSearchService`/DP remain semantically unchanged.

**Legacy impact:**
- Replaces parallel positional `dense_events`, `bm25_events`, and raw `retrieval_events` orchestration.
- Removes top-level `dense_events`/`bm25_events` from `KISRevisionSearchResponse` after frontend migrates to the nested exploration seed.

- [ ] **Step 1: Write failing retrieval-plan tests**

Create `tests/hcmai/retrieval/test_kis_plan.py`:

```python
def test_plan_keeps_event_ids_and_order(): ...
def test_bm25_uses_canonical_text(): ...
def test_dense_uses_translation_for_vietnamese(): ...
def test_dense_uses_canonical_text_for_english_without_translator_call(): ...
def test_plan_rejects_event_alignment_mismatch(): ...
```

The expected plan shape is:

```python
RetrievalEvent(
    event_id="E1",
    canonical_text="...",
    dense_text="..." or None,
    bm25_text="..." or None,
)
```

- [ ] **Step 2: Write failing KIS orchestration test**

Create `tests/hcmai/orchestration/test_kis_revision.py` with deterministic resolver, translator, temporal scorer/materializer doubles. Exercise `SearchService.search_kis_revision()` and assert:

```python
assert response.intent.revision == 3
assert [event.id for event in response.intent.events] == ["E1", "E2", "E3"]
assert response.exploration_seed.events == ["...", "...", "..."]
assert response.latency.intent_ms >= 0
assert response.latency.translation_ms >= 0
assert response.latency.total_ms >= sum([
    response.latency.intent_ms,
    response.latency.translation_ms,
    response.latency.retrieval_ms,
    response.latency.alignment_ms,
    response.latency.materialization_ms,
]) - 1.0
```

- [ ] **Step 3: Run and verify failures**

```bash
PYTHONPATH=src:. python -m pytest \
  tests/hcmai/retrieval/test_kis_plan.py \
  tests/hcmai/orchestration/test_kis_revision.py -v
```

Expected: FAIL because aligned plan/seed/stage latency are absent.

- [ ] **Step 4: Implement `KISRetrievalPlan`**

Create `src/hcmai/retrieval/kis_plan.py` with immutable dataclasses/Pydantic models:

```python
@dataclass(frozen=True, slots=True)
class RetrievalEvent:
    event_id: str
    canonical_text: str
    dense_text: str | None
    bm25_text: str | None


@dataclass(frozen=True, slots=True)
class KISRetrievalPlan:
    events: tuple[RetrievalEvent, ...]

    @property
    def canonical_events(self) -> tuple[str, ...]: ...
    @property
    def dense_events(self) -> tuple[str, ...] | None: ...
    @property
    def bm25_events(self) -> tuple[str, ...] | None: ...
```

Add a builder that accepts `intent`, `event_translator`, `use_dense`, and `use_bm25`, translates one-to-one only when `intent.language != "en"`, and validates exact event ID/order alignment.

**Dense compatibility rule:** Dense translation is required when the intent language is not `"en"` and the embedding model was trained on English text (the current default: CLIP/BGE-M3 with English queries). If the embedding model natively supports the intent language, canonical text is used directly. This decision is made at build time based on `intent.language` and is not inferred from runtime behavior.

- [ ] **Step 5: Make `KISPipeline` consume the plan**

Change `KISPipeline.execute()` from:

```python
intent, retrieval_events, use_dense, use_bm25, ...
```

to:

```python
intent: KISIntent,
plan: KISRetrievalPlan,
top_k: int,
intent_ms: float = 0.0,
translation_ms: float = 0.0,
```

Pass:

```python
original_events=plan.canonical_events
retrieval_events=plan.dense_events or plan.canonical_events
caption_events=plan.bm25_events
use_dense=plan.dense_events is not None
use_bm25=plan.bm25_events is not None
```

Do not change `TemporalSearchService`/DP semantics.

- [ ] **Step 6: Add a dedicated exploration seed contract**

In `api/contracts/kis.py`, define:

```python
class KISExplorationSeed(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: NonBlank
    events: list[NonBlank] = Field(min_length=1)
    retrieval_events: list[NonBlank] = Field(min_length=1)
    caption_events: list[NonBlank] | None
    use_dense: bool
    use_bm25: bool
```

`KISRevisionSearchResponse` becomes:

```python
intent: KISIntent
exploration_seed: KISExplorationSeed
use_dense: bool
use_bm25: bool
results: list[SearchResult]
latency: SearchLatency
```

Remove top-level `dense_events` and `bm25_events`.

This seed exists only because the existing temporal-exploration API must reproduce the committed scoring inputs; it is not shown as product semantics.

- [ ] **Step 7: Split latency stages**

Extend `SearchLatency` with:

```python
intent_ms: float = Field(default=0, ge=0)
translation_ms: float = Field(default=0, ge=0)
```

Keep `query_ms` only if existing non-KIS consumers require it; for KIS set:

```python
query_ms = intent_ms + translation_ms
```

Measure `intent_ms` around resolver only and `translation_ms` around retrieval-plan preparation only. Ensure `total_ms` covers all five stages.

- [ ] **Step 8: Correct router error ownership**

In `api/routers/kis.py`, map:

- request validation / `InvalidQueryInputError` -> 422;
- `RevisionConflictError` -> 409;
- `InferenceUnavailableError` -> 503;
- `InferenceAuthError` -> 502 (provider credentials/configuration are a server integration issue, not user input);
- `InferenceResponseError`, `KISResolutionError`, `EventTranslationError` -> 502;
- `SearchServiceUnavailableError` -> 503.

Remove `ValidationError`/generic resolver `ValueError` from the 422 tuple when those originate after HTTP request validation.

- [ ] **Step 9: Run backend tests**

```bash
PYTHONPATH=src:. python -m pytest \
  tests/hcmai/retrieval/test_kis_plan.py \
  tests/hcmai/orchestration/test_kis_revision.py -v
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/hcmai/retrieval/kis_plan.py src/hcmai/orchestration src/hcmai/api tests/hcmai
git commit -m "refactor: align KIS retrieval through canonical plans"
```

---

## Task 6: Make Text-Embedding Provider Selection Explicit and Add the Generic Self-Hosted Embedding Contract

**Files:**
- Modify: `src/hcmai/inference/config.py`
- Modify: `src/hcmai/inference/embeddings.py`
- Modify: `src/hcmai/orchestration/setup.py`
- Modify: `src/hcmai/orchestration/retrieval_setup.py`
- Modify: `src/hcmai/retrieval/embedding/adapters/remote.py`
- Modify: `llm/contracts.py`
- Modify: `llm/config.py` only if the text-encoder configuration needs a clearer alias; reuse `resolved_evidence_embedding` rather than defining a second model identity.
- Modify: `llm/local/adapter.py`
- Modify: `llm/local/readiness.py`
- Modify: `llm/pipeline.py`
- Modify: `llm/server/routers/embeddings.py`
- Modify: `llm/scripts/deploy_cloudflared_private.sh`
- Modify: `llm/scripts/run_local.sh`
- Test: extend `tests/hcmai/inference/test_clients.py`
- Test: `tests/llm/test_embedding_router.py`
- Test: `tests/hcmai/orchestration/test_readiness.py`

**Interfaces:**
- Produces: explicit `EmbeddingMode = Literal["remote", "local"]`; one `EmbeddingClient | None` constructed at setup; generic self-hosted `POST /v1/embeddings`; local text-embedding capability backed by `config.resolved_evidence_embedding`.
- Consumers: retrieval setup and readiness.

**Legacy impact:**
- Deletes `embedding_client: Any` type-detection and `_query_encoder()`'s implicit `load_embedding_endpoint()` + silent local fallback.
- Replaces the ambiguous deployment flag `HCMAI_ENABLE_CAPTION_EMBEDDING` with `HCMAI_ENABLE_TEXT_EMBEDDING` in this repository after tracing its callers.
- Keeps the deployed `/v1/embeddings/text` route only if another confirmed consumer needs it; HCMAI itself uses the generic `/v1/embeddings` contract.

- [ ] **Step 1: Write failing application-side tests for explicit mode selection**

Create `tests/hcmai/orchestration/test_readiness.py` (this file is first created here; Task 7 will extend it with KIS-specific readiness tests). Extend `tests/hcmai/inference/test_clients.py` with:

```python
def test_embedding_mode_requires_remote_endpoint_in_remote_mode(): ...
def test_remote_embedding_config_failure_does_not_create_local_encoder(): ...
def test_local_embedding_mode_never_constructs_remote_client(): ...
def test_query_encoder_accepts_typed_remote_client_or_explicit_local_mode_only(): ...
def test_embedding_client_uses_generic_v1_embeddings_contract(): ...
```

The remote client URL assertion is:

```python
assert transport.calls[0].url == "https://api.example/v1/embeddings"
```

- [ ] **Step 2: Write the failing self-hosted generic embedding-router tests**

Create `tests/llm/test_embedding_router.py` with an injectable runtime:

```python
from fastapi.testclient import TestClient

from llm.server.app import create_llm_app


class FakeRuntime:
    def load(self):
        pass

    def close(self):
        pass

    def embed_text(self, texts):
        assert texts == ["first", "second"]
        return [[1.0, 2.0], [3.0, 4.0]]

    def readiness(self):
        model = type(
            "Model",
            (),
            {"loaded": True, "checkpoint": "BAAI/bge-m3", "revision": "rev"},
        )()
        return type("Ready", (), {"models": {"text_embedding": model}})()


def test_generic_embeddings_returns_openai_shape():
    client = TestClient(create_llm_app(FakeRuntime()))
    response = client.post(
        "/v1/embeddings",
        json={"model": "BAAI/bge-m3", "input": ["first", "second"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "BAAI/bge-m3"
    assert [row["index"] for row in body["data"]] == [0, 1]
    assert body["data"][0]["embedding"] == [1.0, 2.0]
```

Also test model mismatch and disabled/unready text embedding.

- [ ] **Step 3: Run the focused tests and verify failure**

```bash
PYTHONPATH=src:. python -m pytest \
  tests/hcmai/inference/test_clients.py \
  tests/llm/test_embedding_router.py \
  tests/hcmai/orchestration/test_readiness.py -v
```

Expected: FAIL because explicit mode selection and the generic self-hosted text-embedding route do not exist.

- [ ] **Step 4: Add explicit embedding mode configuration**

In `src/hcmai/inference/config.py`, add a strict loader:

```python
EmbeddingMode = Literal["remote", "local"]


def load_embedding_mode() -> EmbeddingMode:
    raw = os.getenv("HCMAI_EMBEDDING_MODE", "remote").strip().lower()
    if raw not in {"remote", "local"}:
        raise ValueError("HCMAI_EMBEDDING_MODE must be 'remote' or 'local'")
    return cast(EmbeddingMode, raw)
```

Contract:

- `remote`: `HCMAI_EMBEDDING_BASE_URL`, `HCMAI_EMBEDDING_MODEL`, optional API key, and timeout are required; configuration failure makes remote retrieval unavailable.
- `local`: no remote endpoint is constructed; the existing local `EncoderConfig` for each index is used.

Do not infer mode from whether a URL happens to exist.

- [ ] **Step 5: Construct the remote client exactly once in orchestration setup**

In `src/hcmai/orchestration/setup.py`, load `embedding_mode` once. For `remote`, call `load_embedding_endpoint()` and build one `EmbeddingClient`. For `local`, keep `embedding_client=None`. Propagate both explicitly:

```python
load_retrieval(
    ...,
    embedding_mode=embedding_mode,
    embedding_client=embedding_client,
)
```

If mode is `remote` and endpoint/model configuration is invalid, record the startup error and mark the dependent retrieval capability unavailable. Do not instantiate a local encoder as recovery.

- [ ] **Step 6: Make `_query_encoder()` typed and deterministic**

Replace the current `Any`/runtime-type-detection signature with:

```python
def _query_encoder(
    config: EncoderConfig,
    index: Any,
    *,
    embedding_mode: EmbeddingMode,
    embedding_client: EmbeddingClient | None,
    source: str = "text",
) -> Any:
    if embedding_mode == "remote":
        if embedding_client is None:
            raise RuntimeError("remote embedding mode requires EmbeddingClient")
        return EmbeddingService.create_remote_adapter(
            embedding_client,
            config,
            index.metadata.embedding_dim,
            source,
        )
    return EmbeddingService.create_text_adapter(config)
```

Delete the internal call to `load_embedding_endpoint()` and its broad `except Exception` fallback. Update every `_query_encoder()` caller to pass the explicit mode/client.

- [ ] **Step 7: Make the inference service actually own a local text encoder**

In `llm/local/adapter.py`, add `enable_text_embedding: bool` and instantiate a text adapter from the existing canonical configuration. The text adapter is imported from `src/hcmai/retrieval/embedding/` which lives outside the `llm/` package; this is an intentional cross-package dependency because the inference server reuses the same encoder implementation rather than duplicating it:

```python
# Import from the HCMAI retrieval package — the inference server reuses
# the canonical text encoder rather than maintaining a separate implementation.
from hcmai.retrieval.embedding.adapters.text import TextEmbeddingAdapter

self.text_encoder = (
    TextEmbeddingAdapter(config.resolved_evidence_embedding)
    if enable_text_embedding
    else None
)
```

Load it via `_load_model()` inside `load()` and expose:

```python
def embed_text(self, texts: Sequence[str]) -> list[list[float]]:
    if self.text_encoder is None:
        raise RuntimeError("text embedding model is disabled")
    vectors = self.text_encoder.encode_text(list(texts))
    return np.asarray(vectors, dtype=np.float32).tolist()
```

Read `HCMAI_ENABLE_TEXT_EMBEDDING` in `from_environment()`. Update readiness with a distinct `text_embedding` model entry. Do not overload the visual-embedding readiness field.

- [ ] **Step 8: Add the generic OpenAI-compatible embedding route**

In `llm/contracts.py`, add strict request/response models for the subset HCMAI needs:

```python
class EmbeddingRequest(_HTTPContract):
    model: _NonEmptyString
    input: list[_NonEmptyString] = Field(min_length=1)
```

In `llm/server/routers/embeddings.py`, add `POST /v1/embeddings`. Validate the requested model against the loaded `text_embedding` checkpoint, call `runtime.embed_text(request.input)`, preserve request ordering, and return:

```json
{
  "object": "list",
  "model": "BAAI/bge-m3",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}
  ]
}
```

Keep `/v1/embeddings/images` and `/v1/embeddings/dino` unchanged. Reconcile the deployed `/v1/embeddings/text` implementation: if another consumer requires it, make it delegate to the same runtime; otherwise remove it after repository/access-log tracing. Do not special-case HCMAI's `EmbeddingClient` to call `/text`.

- [ ] **Step 9: Delegate text embedding through `LLMService` and deployment flags**

In `llm/pipeline.py`, add:

```python
def embed_text(self, texts):
    method = getattr(self.adapter, "embed_text", None)
    if method is None:
        raise RuntimeError("text embedding is not supported by this provider")
    return method(texts)
```

Migrate repository scripts from `HCMAI_ENABLE_CAPTION_EMBEDDING` / `--caption-embedding` to `HCMAI_ENABLE_TEXT_EMBEDDING` / `--text-embedding` after `rg` confirms there is no separate live caption-only consumer:

```bash
rg -n "ENABLE_CAPTION_EMBEDDING|caption-embedding" llm src
```

If a separate confirmed consumer exists, keep its flag and add the new text flag rather than silently changing its meaning.

- [ ] **Step 10: Run tests**

```bash
PYTHONPATH=src:. python -m pytest \
  tests/hcmai/inference/test_clients.py \
  tests/llm/test_embedding_router.py \
  tests/hcmai/orchestration/test_readiness.py -v
```

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add src/hcmai llm tests
git commit -m "refactor: make text embedding provider selection explicit"
```

---

## Task 7: Make KIS Readiness Reflect Mandatory End-to-End Dependencies

**Files:**
- Modify: `src/hcmai/orchestration/health.py`
- Modify: `src/hcmai/orchestration/setup.py`
- Modify: `src/hcmai/api/routers/system.py`
- Test: `tests/hcmai/orchestration/test_readiness.py`

**Interfaces:**
- Produces health capability fields: `kis`, `intent_resolution`, `event_translation`, `retrieval`.

**Legacy impact:**
- Replaces `kis = search_ready` and legacy `query_preparation` health naming.

- [ ] **Step 1: Extend readiness tests with KIS-specific assertions**

Extend `tests/hcmai/orchestration/test_readiness.py` (created in Task 6 Step 1) with KIS-level readiness tests. Task 6 tests cover embedding-mode selection; this task adds the composite KIS readiness gate:

```python
def test_kis_not_ready_without_intent_resolver(): ...
def test_kis_not_ready_when_vietnamese_dense_requires_missing_translator(): ...
def test_kis_ready_with_resolver_and_retrieval(): ...
def test_health_exposes_intent_resolution_event_translation_and_retrieval(): ...
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=src:. python -m pytest tests/hcmai/orchestration/test_readiness.py -v
```

- [ ] **Step 3: Fix capability projection**

In `build_health_report()` compute explicit booleans. `kis` must at minimum require canonical search data + temporal evidence + `intent_resolver`. `event_translation` reports translator availability independently; it may be optional for English-only requests but must be visible so Vietnamese Dense support is diagnosable.

Replace the legacy health key `query_preparation`.

- [ ] **Step 4: Keep readiness side-effect free**

Do not issue a fresh completion in `/ready`. Rely on constructed client/configuration plus inference-server readiness already sampled at startup. Runtime provider failures still surface as 502/503 on KIS search.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONPATH=src:. python -m pytest tests/hcmai/orchestration/test_readiness.py -v

git add src/hcmai/orchestration src/hcmai/api/routers/system.py tests/hcmai/orchestration/test_readiness.py
git commit -m "fix: report end-to-end KIS readiness"
```

---

## Task 8: Make Frontend KIS Revisions Transactional and Preserve Committed Exploration

**Files:**
- Modify: `frontend/src/features/kis/session.js`
- Modify: `frontend/src/features/kis/session.test.js`
- Modify: `frontend/src/features/search/components/SearchWorkspace.jsx`
- Modify: `frontend/src/features/search/components/SearchWorkspace.test.jsx`
- Modify: `frontend/src/api/kis.js`
- Modify: `frontend/src/api/kis.test.js`
- Modify: `frontend/src/features/alignment/hooks/useTemporalExploration.js`
- Modify: `frontend/src/features/alignment/hooks/useTemporalExploration.test.js`
- Modify: `frontend/src/App.jsx`
- Modify: `frontend/src/App.test.jsx`

**Interfaces:**
- Consumes: `KISRevisionSearchResponse.exploration_seed` from Task 5.
- Produces: committed revision state that survives pending/failing clue requests.

**Codebase state note:** The current `vbs` branch already has `commitSearchSuccess()` and `commitSearchFailure()` helpers in `frontend/src/features/kis/session.js` (introduced during the semantic-intent migration). This task must build on those existing helpers rather than creating new ones. Verify their current behavior matches the transactional invariant before extending.

**Legacy impact:**
- Removes duplicate `onFrameClick` call.
- Removes draft -> global `activeQuery` propagation.
- Removes construction of temporal exploration from top-level `dense_events` / `bm25_events`.

- [ ] **Step 1: Add regression tests for the reviewed bugs**

In `SearchWorkspace.test.jsx`, add tests that assert:

1. Q1 results remain in the DOM while Q2 API promise is unresolved.
2. Q2 failure keeps Q1 results/current intent and preserves Q2 draft.
3. Q2 success atomically replaces Q1 results and clears draft.
4. `onFrameClick` is called exactly once and carries the committed exploration seed.
5. typing Q2 does not call parent `onQueryChange` with draft text after Q1 is committed.
6. history-save failure shows a warning but does not keep `Searching…` active.

Use a deferred promise helper so pending state is observable before resolution.

- [ ] **Step 2: Add API tests for `exploration_seed`**

Update `frontend/src/api/kis.test.js` fixture to contain:

```js
exploration_seed: {
  query: 'canonical query',
  events: ['E1 text'],
  retrieval_events: ['dense E1 text'],
  caption_events: ['E1 text'],
  use_dense: true,
  use_bm25: true,
}
```

Remove expectations for top-level `dense_events` / `bm25_events`.

- [ ] **Step 3: Run focused frontend tests and verify failure**

```bash
cd frontend
CI=true npm test -- --runInBand \
  src/features/search/components/SearchWorkspace.test.jsx \
  src/features/kis/session.test.js \
  src/api/kis.test.js \
  src/features/alignment/hooks/useTemporalExploration.test.js
```

Expected: FAIL on current clearing/double-callback behavior.

- [ ] **Step 4: Stop clearing or hiding committed result state at KIS request start**

In `SearchWorkspace.submit()` remove the pre-request mutations that clear:

```text
liveKisSnapshotRef
frames
kisEvents
searchLatencyMs
resultType
activeQuerySession
```

for a normal clue revision. Keep them for explicit `New Search`, image-search mode switch, and replay mode switch.

Set only pending flags/errors at request start. In the render path, replace any `!isSearching && renderResults()` guard with an unconditional committed-results render plus the existing/loading overlay. On the first ever search there are simply no committed results to render; on Q2+ the old result grid remains interactive while the pending overlay is visible.

- [ ] **Step 5: Commit successful revision as one success branch**

After `searchKis()` returns:

```js
const eventTexts = response.intent.events.map((event) => event.text);
const explorationSnapshot = response.exploration_seed;

setKisSession((prev) => commitSearchSuccess(prev, response));
liveKisSnapshotRef.current = explorationSnapshot;
setFrames(response.results || []);
setKisEvents(eventTexts);
setSearchLatencyMs(response.latency);
setWarnings(response.warnings || []);
setResultType('retrieval');
onQueryChange?.(response.intent.query_text);
```

There must be no second path that updates global query state from the draft.

- [ ] **Step 6: Fix frame selection exactly once**

Replace `openCanonicalFrame()` with one callback:

```js
onFrameClick?.({
  frame,
  ...(liveKisSnapshotRef.current
    ? { explorationSnapshot: liveKisSnapshotRef.current }
    : {}),
});
```

Add `toHaveBeenCalledTimes(1)` regression assertion.

- [ ] **Step 7: Preserve committed results on failure**

On request failure, only call `commitSearchFailure()` and set the error. Do not replace `resultType`, frames, intent, latency, exploration seed, or activity state.

The failed draft remains in `kisSession.draft` for retry/editing.

- [ ] **Step 8: Move history persistence outside the live search lock**

After UI commit, release `isSearching`/request ownership before `createQueryHistory()` completes. Run history persistence as a follow-up promise keyed by the committed response; on failure append a warning only.

Do not pass the search request's abort signal to history persistence after the search transaction is committed; otherwise a subsequent search can cancel history for a committed revision.

- [ ] **Step 9: Migrate temporal exploration hook to the dedicated seed**

Change `useTemporalExploration.js` to expect:

```js
snapshot.query
snapshot.events
snapshot.retrieval_events
snapshot.caption_events
snapshot.use_dense
snapshot.use_bm25
```

directly. Remove all checks for `snapshot.dense_events` and `snapshot.bm25_caption_events`.

- [ ] **Step 10: Run focused tests**

```bash
cd frontend
CI=true npm test -- --runInBand \
  src/features/search/components/SearchWorkspace.test.jsx \
  src/features/kis/session.test.js \
  src/api/kis.test.js \
  src/features/alignment/hooks/useTemporalExploration.test.js \
  src/App.test.jsx
```

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add frontend/src
git commit -m "fix: make KIS revisions transactional"
```

---

## Task 9: Simplify KIS Panel Semantics and Separate Replay from Live Search

**Files:**
- Modify: `frontend/src/features/kis/components/KisPanel.jsx`
- Modify: `frontend/src/features/kis/components/KisPanel.test.jsx`
- Modify: `frontend/src/features/search/components/SearchWorkspace.jsx`
- Modify: `frontend/src/features/search/components/SearchWorkspace.test.jsx`
- Modify: `frontend/src/features/workspace/queryHistory.js`
- Modify: `frontend/src/features/workspace/queryHistory.test.js`
- Modify: KIS/chat styles in the existing stylesheet containing `.kis-chat-*` classes.

**Interfaces:**
- Produces: one intent-evolution panel showing clue history, current canonical intent, ordered event chain, collapsed semantic details.
- Replay becomes an explicit read-only result mode and never populates live draft state.

**Legacy impact:**
- Removes raw temporal-edge display from normal KIS panel.
- Removes replay behavior that calls `setDraft(..., item.query_text)`.

- [ ] **Step 1: Add panel/replay tests**

In `KisPanel.test.jsx` assert:

```text
revision 0 button = Search
revision >= 1 button = Add clue
Current intent label is visible
ordered E1/E2/E3 event rows are visible
Entities/bindings are hidden until Semantic details is expanded
raw temporal edge badges are absent in normal view
```

In `SearchWorkspace.test.jsx`, replay must assert the KIS textarea is hidden/disabled and does not contain the saved query as a new draft.

- [ ] **Step 2: Run and verify failure**

```bash
cd frontend
CI=true npm test -- --runInBand \
  src/features/kis/components/KisPanel.test.jsx \
  src/features/search/components/SearchWorkspace.test.jsx
```

- [ ] **Step 3: Simplify `KisPanel`**

Rename copy:

```text
KIS Semantic Search -> KIS
Resolved Query -> Current intent
Clues: -> Search clues
```

Render events as ordered vertical rows with `E1`, `E2`, ... and a visual down-arrow/separator between them.

Move entities and per-event bindings under native `<details><summary>Semantic details</summary>...</details>`. Do not render `temporal_edges` in normal product UI.

Set submit label inside the component or its caller from `revision`:

```js
const actionLabel = revision === 0 ? 'Search' : 'Add clue';
```

- [ ] **Step 4: Make replay read-only**

In replay effect, remove:

```js
setKisSession(setDraft(createInitialKisSessionState(), item.query_text || ''));
```

Instead reset live session and set explicit replay state only. While `resultType` starts with `replay-`, do not render the editable `KisPanel`; render a read-only summary from `result_snapshot.intent` plus a `New Search` action.

Do not add “continue from replay” in this batch.

- [ ] **Step 5: Keep history snapshots semantic**

`buildKisSnapshot()` must continue to store the canonical `intent`, latency, warnings, and result identities. It must not add embeddings, dense translation internals, or model-provider configuration.

- [ ] **Step 6: Run tests and commit**

```bash
cd frontend
CI=true npm test -- --runInBand \
  src/features/kis/components/KisPanel.test.jsx \
  src/features/search/components/SearchWorkspace.test.jsx \
  src/features/workspace/queryHistory.test.js

git add frontend/src
git commit -m "refactor: focus KIS panel on intent evolution"
```

---

## Task 10: Repository Cleanup, Full Regression Gate, and Manual Progressive-KIS Acceptance

**Files:**
- Create/Modify: root `.gitignore`
- Modify: stale docstrings/docs found by repository scans.
- Delete: generated caches/build artifacts from tracking/snapshot.
- Verify: all source and test files from Tasks 1–9.

**Interfaces:**
- Produces: no new runtime API; this is the completion gate before EventTrail design/implementation.

**Legacy impact:**
- Enforces zero production references to every superseded KIS/query-expansion/query-preparation abstraction.

- [ ] **Step 1: Add/repair ignore rules and remove generated artifacts**

Ensure root `.gitignore` contains at least:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
frontend/build/
frontend/node_modules/
.env
```

Remove generated artifacts from tracked/source snapshots:

```bash
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type f -name '*.pyc' -delete
rm -rf .pytest_cache frontend/build
```

Do not delete user-local `.env`; only ensure it is ignored.

- [ ] **Step 2: Run the hard legacy-reference scan**

Run:

```bash
rg -n \
  "KISIntentBuilder|plan_query_events|split_query_events|QueryPreparationService|generate_candidates|CandidateBundle|QueryCandidateSet|/api/v1/query-candidates|query_preparation|dense_events|bm25_events|HCMAI_ENABLE_QUERY_PREPARATION" \
  src llm frontend \
  --glob '!frontend/node_modules/**' \
  --glob '!frontend/build/**' \
  --glob '!**/__pycache__/**'
```

Expected production results:

- `KISIntentBuilder`, `plan_query_events`, `split_query_events` — already deleted in prior migration (commit `be790e0`); zero matches expected;
- no query-expansion/query-preparation service/API references;
- no top-level KIS `dense_events`/`bm25_events` consumer;
- **TRAKE exceptions:** references to `dense_events` inside `src/hcmai/api/contracts/trake.py` and `src/hcmai/orchestration/workflows/trake.py` are **allowed** — TRAKE is a separate task contract and is not migrated in this batch;
- no old inference `HCMAI_ENABLE_QUERY_PREPARATION` flag.

If a match is a stale docstring/README, update it in this task. If a match is production code, the responsible earlier task is incomplete and must be fixed before continuing.

- [ ] **Step 3: Compile Python sources**

```bash
PYTHONPATH=src:. python -m compileall -q src/hcmai llm tests
```

Expected: exit code 0.

- [ ] **Step 4: Run all backend stabilization tests**

```bash
PYTHONPATH=src:. python -m pytest tests -v
```

Expected: PASS.

- [ ] **Step 5: Install frontend dependencies reproducibly and run the full frontend suite**

```bash
cd frontend
npm ci
CI=true npm test -- --runInBand
```

Expected: PASS.

- [ ] **Step 6: Build the frontend**

```bash
cd frontend
npm run build
```

Expected: successful production build; generated `frontend/build/` remains ignored/untracked.

- [ ] **Step 7: Verify inference-server API shape**

Against the self-hosted API, verify:

```bash
curl -fsS https://api.iamphuckhang.dev/ready
curl -fsS -X POST https://api.iamphuckhang.dev/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${HCMAI_LLM_API_KEY}" \
  -d '{
    "model":"Qwen/Qwen3-4B",
    "messages":[{"role":"user","content":"Return JSON with answer equal to ok"}],
    "temperature":0,
    "max_tokens":64,
    "response_format":{
      "type":"json_schema",
      "json_schema":{
        "name":"Answer",
        "schema":{
          "type":"object",
          "properties":{"answer":{"type":"string"}},
          "required":["answer"],
          "additionalProperties":false
        }
      }
    }
  }'
```

Expected: 2xx, assistant content is a JSON object, and the model identity matches configured `HCMAI_LLM_MODEL`.

- [ ] **Step 8: Run the progressive KIS manual acceptance scenario**

Start the backend/frontend with configured LLM and retrieval dependencies. Execute:

```text
Q1: A woman is standing in a kitchen.
Q2: She is talking to a man.
Q3: Before taking a white plate, they move to the left.
```

Verify in sequence:

1. revision 1 results appear and remain inspectable;
2. while Q2 is pending, revision 1 results remain visible;
3. revision 2 canonical intent resolves “She” to the same woman;
4. while Q3 is pending, revision 2 remains usable;
5. revision 3 event order is `talk -> move left -> take plate`;
6. backend logs/test instrumentation show DP receives the same ordered event sequence;
7. opening a result invokes one selection and temporal exploration opens from the committed seed;
8. force the LLM endpoint to fail for Q4; revision 3 results/intent remain active and Q4 draft stays editable;
9. restore the provider and retry the preserved Q4 draft successfully;
10. history write failure, if simulated, shows only a warning and does not block the next clue.

- [ ] **Step 9: Verify readiness semantics**

Temporarily unset `HCMAI_LLM_BASE_URL`/`HCMAI_LLM_MODEL`, restart, and verify KIS readiness is false. Restore configuration and verify `intent_resolution`, `event_translation`, `retrieval`, and `kis` report expected states.

- [ ] **Step 10: Final codebase size/legacy audit**

Run:

```bash
git status --short
git ls-files | rg '__pycache__|\.pyc$|^frontend/build/' && exit 1 || true
```

Review the diff specifically for parallel old/new abstractions. No compatibility shim or unused package may remain solely “for later”.

- [ ] **Step 11: Commit the stabilization gate**

```bash
git add .
git commit -m "chore: complete KIS stabilization gate"
```

---

## Execution Order and Review Checkpoints

Do not parallelize Tasks 1–5 because each changes the contract consumed by the next task. Tasks 6 and 7 may be reviewed independently only after Task 5 is merged. Tasks 8 and 9 should run after the backend KIS response contract is stable. Task 10 is mandatory and must not be skipped.

### Codebase starting state (vbs branch)

The following work is already complete and should not be repeated:

- `KISIntentBuilder`, `plan_query_events`, `split_query_events`, legacy raw-query KIS path — deleted in commit `be790e0`.
- `commitSearchSuccess()` / `commitSearchFailure()` session helpers — exist in `frontend/src/features/kis/session.js`.
- `feat/ui-vbs` merge — chat panel layout, inline video player, direct DRES submission are in the current tree.
- Database tab and shared `AnswerWorkspace` — removed.

The following legacy is still alive and must be migrated by this plan:

- `QueryPreparationService` + `generate_candidates()` + `CandidateBundle` + `query_candidates` router — 40+ references.
- `dense_events`/`bm25_events` in KIS response and frontend exploration — active.
- `query_preparation` health key — active.
- `generate_structured(messages, KISIntent)` in resolver — must change to `KISResolution`.

### Rollback strategy

Each task ends with a commit. If Task N breaks integration that cannot be resolved promptly:

1. `git revert HEAD` (or `git reset --hard HEAD~1` if not yet pushed).
2. Investigate the failure cause using the task's own test suite.
3. Fix and re-attempt Task N before continuing to Task N+1.

Do not skip a failing task and proceed to later tasks that depend on its contract.

### Recommended checkpoints

1. **After Task 2:** verify generic provider/client boundary before domain refactors.
2. **After Task 3:** review semantic ownership (`KISResolution` vs `KISIntent`).
3. **After Task 5:** review end-to-end backend KIS contract and existing exploration compatibility.
4. **After Task 7:** verify startup/readiness behavior.
5. **After Task 9:** review KIS panel/replay UX before EventTrail work.
6. **After Task 10:** only then begin the next EventTrail brainstorming cycle.

## Self-Review Against the Approved Spec

- Decision A: covered by Tasks 8–9.
- Decision B: covered by Task 3.
- Decision C: covered by Task 5.
- Decision D: covered by Task 4.
- Decision E: covered by Tasks 2, 4, 6, 7.
- Decision F: covered by Task 9.
- Decision G: covered by Task 1 plus deployment reconciliation in Tasks 4/10.
- Decision H: covered by Task 10 and the test work embedded in every preceding task.
- Existing temporal exploration remains functional through `KISExplorationSeed`; no EventTrail feature is introduced.
- No placeholders, speculative query-expansion path, deterministic KIS fallback, or parallel query-preparation package remains in the target state.
