# Qwen Query Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stateless Vietnamese→English literal translation and exactly five event-aligned retrieval candidates through a locally hosted Qwen3-4B inference boundary.

**Architecture:** FastAPI/runtime never loads Qwen weights. Thundercompute hosts Qwen3-4B and exposes a narrow structured query-preparation operation; `hcmai.query_preparation` owns validation, cache, and orchestration. KIS raw text is split with the existing deterministic planner before generation, while TRAKE explicit event boundaries are preserved exactly.

**Tech Stack:** Python 3.11+, FastAPI/Pydantic, Thundercompute inference service, Qwen/Qwen3-4B, httpx/existing `LLMService` gateway, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-hybrid-dense-bm25-query-preparation-design.md`

## Global Constraints

- Qwen model: `Qwen/Qwen3-4B`; resolve and pin the immutable Hugging Face commit SHA before benchmark/freeze.
- Candidate count is exactly `5` in the public product contract.
- Candidate generation is explicit; normal Dense-only Original retrieval must not call Qwen.
- No server-side query/candidate session, candidate ID, or search state.
- For `N` input events, literal translation and every candidate bundle must contain exactly `N` events in the same order.
- Preserve entities, colors, numbers, quantities, actions, named tokens, and placeholders such as `X`; do not infer missing facts.
- Malformed structured output gets at most one generation retry and then an explicit failure.
- Cache key includes operation, normalized original events, model name, model revision, and prompt version.
- Runtime API workers must not import Transformers or allocate Qwen weights.
- Execution prerequisite: use the full repository checkout that contains `thundercompute/server/api.py`, `thundercompute/pipeline.py`, and `thundercompute/config.yaml`; the reviewed `src_hcmai_v6.zip` contains only `src/hcmai` and is insufficient for Task 3. If any of those three paths is absent, stop before Task 3 and restore the full Thundercompute source rather than inventing a second inference server.

---

## File Map

**Create in HCMAI runtime:**
- `src/hcmai/query_preparation/__init__.py` — public exports.
- `src/hcmai/query_preparation/models.py` — immutable internal candidate models and adapter protocol.
- `src/hcmai/query_preparation/cache.py` — bounded TTL process-local cache.
- `src/hcmai/query_preparation/service.py` — literal/candidate validation, retry, cache policy.
- `src/hcmai/query_preparation/adapters/qwen.py` — adapter over the Thundercompute client boundary.
- `src/hcmai/api/contracts/query_candidates.py` — Pydantic HTTP contracts.
- `src/hcmai/api/routers/query_candidates.py` — `/api/v1/query-candidates`.

**Modify in HCMAI runtime:**
- `src/hcmai/common/config.py` — `QueryPreparationConfig`.
- `src/hcmai/api/contracts/__init__.py` — exports.
- `src/hcmai/api/routers/__init__.py` — router export.
- `src/hcmai/app.py` — mount router.
- `src/hcmai/orchestration/pipeline.py` — service method and health.
- `src/hcmai/orchestration/setup.py` — construct adapter/service from the existing inference gateway.

**Modify/add in the full Thundercompute repository/service:**
- `thundercompute/config.yaml` — Qwen3-4B query-preparation model configuration.
- `thundercompute/server/api.py` — structured query-preparation endpoint/capability.
- `thundercompute/pipeline.py` — client/gateway method consumed by HCMAI.
- `thundercompute/query_preparation.py` — model loading, prompts, JSON generation.

**Tests:**
- `tests/query_preparation/test_cache.py`
- `tests/query_preparation/test_service.py`
- `tests/query_preparation/test_qwen_adapter.py`
- `tests/api/test_query_candidates.py`
- `tests/orchestration/test_query_preparation_health.py`
- `tests/thundercompute/test_query_preparation_api.py`

---

### Task 1: Add Query-Preparation Configuration and Internal Models

**Files:**
- Modify: `src/hcmai/common/config.py`
- Create: `src/hcmai/query_preparation/__init__.py`
- Create: `src/hcmai/query_preparation/models.py`
- Test: `tests/query_preparation/test_models.py`
- Test: `tests/common/test_query_preparation_config.py`

**Interfaces:**
- Produces: `QueryPreparationConfig`, `QueryCandidate`, `QueryCandidateSet`, and `QueryPreparationAdapter`.
- Later tasks consume `QueryPreparationAdapter.translate(events)` and `.generate_candidates(events, candidate_count)`.

- [ ] **Step 1: Write config validation tests**

```python
from pydantic import ValidationError
import pytest
from hcmai.common.config import QueryPreparationConfig


def test_query_preparation_defaults_are_frozen_product_defaults():
    cfg = QueryPreparationConfig()
    assert cfg.model_name == "Qwen/Qwen3-4B"
    assert cfg.prompt_version == "query-prep-v1"
    assert cfg.candidate_count == 5
    assert cfg.cache_ttl_seconds == 3600
    assert cfg.cache_max_entries == 2048


def test_candidate_count_cannot_drift_from_public_contract():
    with pytest.raises(ValidationError):
        QueryPreparationConfig(candidate_count=4)
```

- [ ] **Step 2: Run tests and verify they fail because the config does not exist**

Run: `PYTHONPATH=src pytest tests/common/test_query_preparation_config.py -v`

Expected: FAIL importing `QueryPreparationConfig`.

- [ ] **Step 3: Implement the config**

Add to `src/hcmai/common/config.py`:

```python
class QueryPreparationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = "Qwen/Qwen3-4B"
    model_revision: str = Field(min_length=40, max_length=40, pattern=r"^[0-9a-f]{40}$")
    prompt_version: str = Field(default="query-prep-v1", min_length=1)
    candidate_count: Literal[5] = 5
    cache_enabled: bool = True
    cache_ttl_seconds: float = Field(default=3600, gt=0)
    cache_max_entries: int = Field(default=2048, ge=1)
```

Add `query_preparation: QueryPreparationConfig` to `AppConfig`/the runtime settings object that currently owns `search` and `inference`.

Resolve the model SHA in the deployment environment with:

```bash
python - <<'PY2'
from huggingface_hub import HfApi
print(HfApi().model_info("Qwen/Qwen3-4B").sha)
PY2
```

Record that exact 40-character SHA in the deployment config; do not use `main` for the benchmark build.

- [ ] **Step 4: Add immutable internal models and adapter protocol**

```python
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True, slots=True)
class QueryCandidate:
    index: int
    events: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class QueryCandidateSet:
    original_events: tuple[str, ...]
    literal_en: tuple[str, ...]
    candidates: tuple[QueryCandidate, ...]

class QueryPreparationAdapter(Protocol):
    def translate(self, events_vi: Sequence[str]) -> tuple[str, ...]: ...
    def generate_candidates(
        self, events_vi: Sequence[str], candidate_count: int
    ) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]: ...
```

- [ ] **Step 5: Add model tests for immutability and positional event preservation**

```python
from dataclasses import FrozenInstanceError
import pytest
from hcmai.query_preparation.models import QueryCandidate


def test_query_candidate_is_immutable():
    candidate = QueryCandidate(index=1, events=("E1", "E2"))
    with pytest.raises(FrozenInstanceError):
        candidate.index = 2
```

- [ ] **Step 6: Run model/config tests**

Run: `PYTHONPATH=src pytest tests/common/test_query_preparation_config.py tests/query_preparation/test_models.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/hcmai/common/config.py src/hcmai/query_preparation tests/common/test_query_preparation_config.py tests/query_preparation/test_models.py
git commit -m "feat: define query preparation contracts"
```

---

### Task 2: Implement the Bounded TTL Cache

**Files:**
- Create: `src/hcmai/query_preparation/cache.py`
- Test: `tests/query_preparation/test_cache.py`

**Interfaces:**
- Consumes: operation name, normalized event tuple, model name/revision, prompt version.
- Produces: `QueryPreparationCache.get(key)` / `.put(key, value)` with TTL and LRU-style bounded eviction.

- [ ] **Step 1: Write cache tests using an injected clock**

```python
from hcmai.query_preparation.cache import QueryPreparationCache, cache_key


def test_cache_key_separates_translation_and_generation():
    common = dict(events=("xin chao",), model_name="qwen", model_revision="a" * 40, prompt_version="v1")
    assert cache_key(operation="translate", **common) != cache_key(operation="candidates", **common)


def test_cache_expires_and_evicts_oldest_entry():
    now = [100.0]
    cache = QueryPreparationCache(max_entries=1, ttl_seconds=10, clock=lambda: now[0])
    cache.put(("a",), "A")
    cache.put(("b",), "B")
    assert cache.get(("a",)) is None
    assert cache.get(("b",)) == "B"
    now[0] = 111.0
    assert cache.get(("b",)) is None
```

- [ ] **Step 2: Run and verify failure**

Run: `PYTHONPATH=src pytest tests/query_preparation/test_cache.py -v`

Expected: FAIL because cache module is absent.

- [ ] **Step 3: Implement deterministic normalization and bounded TTL cache**

Normalize each event with `" ".join(event.split())` only; do not lowercase or translate it because case-sensitive named tokens must remain visible to the model and cache identity.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src pytest tests/query_preparation/test_cache.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hcmai/query_preparation/cache.py tests/query_preparation/test_cache.py
git commit -m "feat: cache query preparation results"
```

---

### Task 3: Add the Thundercompute Qwen3-4B Query-Preparation Capability

**Files:**
- Create: `thundercompute/query_preparation.py`
- Modify: `thundercompute/config.yaml`
- Modify: `thundercompute/server/api.py`
- Modify: `thundercompute/pipeline.py`
- Test: `tests/thundercompute/test_query_preparation_api.py`

**Interfaces:**
- Produces two gateway operations used by HCMAI:
  - `translate_query_events(events: list[str]) -> list[str]`
  - `generate_query_candidates(events: list[str], candidate_count: int = 5) -> dict`
- The gateway response is structured data, never free-form text passed directly to DP.

- [ ] **Step 1: Write API contract tests with the Qwen engine replaced by a fake**

```python
def test_translate_query_events_returns_one_output_per_input(client, fake_query_preparer):
    fake_query_preparer.translation = ["a chef holds X", "the chef rolls X"]
    response = client.post("/query-preparation/translate", json={
        "events": ["dau bep cam X", "dau bep lan X"]
    })
    assert response.status_code == 200
    assert response.json()["events"] == fake_query_preparer.translation


def test_candidate_endpoint_returns_exactly_five_aligned_bundles(client, fake_query_preparer):
    response = client.post("/query-preparation/candidates", json={
        "events": ["E1", "E2"], "candidate_count": 5
    })
    body = response.json()
    assert len(body["literal_en"]) == 2
    assert len(body["candidates"]) == 5
    assert all(len(item) == 2 for item in body["candidates"])
```

- [ ] **Step 2: Run the contract tests and verify failure**

Run: `pytest tests/thundercompute/test_query_preparation_api.py -v`

Expected: FAIL with missing routes/capability.

- [ ] **Step 3: Add the pinned model configuration**

Add a `query_preparation` block to `thundercompute/config.yaml` using the immutable SHA resolved in Task 1:

Add this block using the 40-character SHA printed by Task 1 as the concrete `revision` value:

```yaml
query_preparation:
  model_checkpoint: "Qwen/Qwen3-4B"
  device: "cuda"
  dtype: "bfloat16"
  max_new_tokens: 768
  do_sample_translation: false
  candidate_temperature: 0.6
```

Set `query_preparation.revision` to the exact Task 1 stdout before the first test run. Add a config test requiring `revision` to match `^[0-9a-f]{40}$`; CI must fail if the immutable revision is absent.

- [ ] **Step 4: Implement one model owner in `thundercompute/query_preparation.py`**

Load tokenizer/model exactly once per Thundercompute process. Provide two prompts:

```text
TRANSLATE: Translate each Vietnamese retrieval event to concise literal English. Preserve names, numbers, colors, quantities, actions, acronyms and placeholders such as X. Do not add facts. Return JSON only and preserve input event count/order.

CANDIDATES: Return one literal English translation plus exactly five controlled English retrieval paraphrase bundles. Every bundle must preserve event count/order and all facts. Do not infer an unknown entity or replace X. Return JSON only.
```

Use non-thinking/greedy decoding for translation; use non-thinking generation with temperature `0.6` for candidate diversity. Validate the decoded JSON before returning from llm.

- [ ] **Step 5: Add the two FastAPI routes and gateway client methods**

Routes validate non-empty event arrays and return HTTP 422 on malformed request. Model output shape errors return 502/explicit inference failure rather than a partial candidate set.

- [ ] **Step 6: Run Thundercompute tests**

Run: `pytest tests/thundercompute/test_query_preparation_api.py -v`

Expected: PASS with the fake engine; no GPU is needed for this test.

- [ ] **Step 7: Run a one-query GPU smoke test on A6000/L40**

Run against the local service:

```bash
curl --fail --silent http://127.0.0.1:8100/query-preparation/candidates   -H 'Content-Type: application/json'   -d '{"events":["Một cô gái mặc tạp dề trắng cầm hai con X"],"candidate_count":5}'
```

Expected: JSON with `literal_en` length 1, `candidates` length 5, each candidate length 1, every output retaining `X` and quantity `two`/`2`.

- [ ] **Step 8: Commit**

```bash
git add thundercompute/config.yaml thundercompute/query_preparation.py thundercompute/server/api.py thundercompute/pipeline.py tests/thundercompute/test_query_preparation_api.py
git commit -m "feat: host qwen query preparation"
```

---

### Task 4: Implement Runtime Adapter, Validation, Retry, and Cache

**Files:**
- Create: `src/hcmai/query_preparation/adapters/__init__.py`
- Create: `src/hcmai/query_preparation/adapters/qwen.py`
- Create: `src/hcmai/query_preparation/service.py`
- Test: `tests/query_preparation/test_qwen_adapter.py`
- Test: `tests/query_preparation/test_service.py`

**Interfaces:**
- Produces:
  - `QueryPreparationService.translate_literal(events_vi) -> tuple[str, ...]`
  - `QueryPreparationService.generate_candidates(events_vi) -> QueryCandidateSet`
- Adapter wraps the Thundercompute gateway methods from Task 3.

- [ ] **Step 1: Write service tests with a scripted fake adapter**

Cover cache hit, event-count mismatch, exactly-five invariant, retry once, and preservation contract:

```python
def test_generate_candidates_retries_once_then_succeeds(fake_adapter, service):
    fake_adapter.outputs = [
        (("literal",), (("only one",),)),
        (("literal",), tuple((f"candidate {i}",) for i in range(5))),
    ]
    result = service.generate_candidates(("mot su kien",))
    assert len(result.candidates) == 5
    assert fake_adapter.generate_calls == 2


def test_event_count_mismatch_is_rejected(service, fake_adapter):
    fake_adapter.translation = ("one",)
    with pytest.raises(QueryPreparationError, match="event count"):
        service.translate_literal(("E1", "E2"))
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=src pytest tests/query_preparation/test_qwen_adapter.py tests/query_preparation/test_service.py -v`

Expected: FAIL because service/adapter are absent.

- [ ] **Step 3: Implement adapter as a thin conversion layer**

Do not parse arbitrary prose in HCMAI. Thundercompute returns structured arrays; the adapter converts them to tuples and raises `QueryPreparationError` for missing/wrong fields.

- [ ] **Step 4: Implement service validation and cache policy**

Translation performs one inference attempt. Candidate generation performs at most two total attempts. Validate all non-empty event strings and exact positional lengths before caching.

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=src pytest tests/query_preparation -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/hcmai/query_preparation tests/query_preparation
git commit -m "feat: validate and cache query preparation"
```

---

### Task 5: Add the Stateless Query-Candidate HTTP API

**Files:**
- Create: `src/hcmai/api/contracts/query_candidates.py`
- Create: `src/hcmai/api/routers/query_candidates.py`
- Modify: `src/hcmai/api/contracts/__init__.py`
- Modify: `src/hcmai/api/routers/__init__.py`
- Modify: `src/hcmai/app.py`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Test: `tests/api/test_query_candidates.py`

**Interfaces:**
- `POST /api/v1/query-candidates`
- Request uses exactly one of `query` or `events`.
- Response contains `original_events`, `literal_en`, exactly five candidate objects, and `query_preparation_ms`.

- [ ] **Step 1: Write Pydantic/request contract tests**

Test `query` only, `events` only, neither, and both. Both/neither must return 422.

- [ ] **Step 2: Write router tests proving KIS uses `split_query_events()` and TRAKE does not re-split**

For KIS input `"mot. hai."`, assert the fake service receives the exact deterministic planner output. For explicit `events=["a. b", "c"]`, assert those two strings remain two events.

- [ ] **Step 3: Run tests and verify failure**

Run: `PYTHONPATH=src pytest tests/api/test_query_candidates.py -v`

Expected: FAIL with missing route/contracts.

- [ ] **Step 4: Implement the contracts**

Use a Pydantic `model_validator(mode="after")` to enforce exactly one input form. Constrain candidate response indexes to 1..5 and verify candidate list length 5 before serialization.

- [ ] **Step 5: Add `SearchService.generate_query_candidates(...)`**

The orchestration method resolves original events then times `QueryPreparationService.generate_candidates`. It returns no persistent handle or ID.

- [ ] **Step 6: Mount the router under `/api/v1`**

Follow existing `search.py`/`trake.py` dependency injection style; map query-preparation unavailability to an explicit service-unavailable HTTP error.

- [ ] **Step 7: Run API tests**

Run: `PYTHONPATH=src pytest tests/api/test_query_candidates.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/hcmai/api src/hcmai/app.py src/hcmai/orchestration/pipeline.py tests/api/test_query_candidates.py
git commit -m "feat: expose query candidate generation"
```

---

### Task 6: Wire Query Preparation and Health Without Making Search Depend on It Globally

**Files:**
- Modify: `src/hcmai/orchestration/setup.py`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Test: `tests/orchestration/test_query_preparation_health.py`

**Interfaces:**
- `SearchService.query_preparation` may be unavailable while ordinary Dense-only search remains available.
- `health()["capabilities"]["query_preparation"]` reports readiness independently.

- [ ] **Step 1: Write health tests**

```python
def test_query_preparation_is_independent_from_dense_search(service_without_qwen):
    health = service_without_qwen.health()
    assert health["capabilities"]["search"] is True
    assert health["capabilities"]["query_preparation"] is False
```

- [ ] **Step 2: Run and verify failure**

Run: `PYTHONPATH=src pytest tests/orchestration/test_query_preparation_health.py -v`

Expected: FAIL because capability is absent.

- [ ] **Step 3: Construct `QueryPreparationService` only when the Thundercompute capability is ready**

Do not make failure to load Qwen disable the existing corpus or Dense retrieval. Append a startup diagnostic explaining only query preparation is unavailable.

- [ ] **Step 4: Close only resources actually owned by SearchService**

Keep the existing shared gateway lifecycle; `QueryPreparationService` itself owns no GPU/model resource.

- [ ] **Step 5: Run focused and orchestration tests**

Run: `PYTHONPATH=src pytest tests/query_preparation tests/api/test_query_candidates.py tests/orchestration/test_query_preparation_health.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/hcmai/orchestration tests/orchestration/test_query_preparation_health.py
git commit -m "feat: wire query preparation capability"
```

---

### Task 7: Query-Preparation Acceptance Gate

**Files:**
- No production changes unless verification exposes a defect.

- [ ] **Step 1: Compile runtime and Thundercompute code**

Run:

```bash
PYTHONPATH=.:src python -m compileall -q src/hcmai/query_preparation src/hcmai/api thundercompute
```

Expected: exit 0.

- [ ] **Step 2: Run the complete query-preparation test slice**

```bash
PYTHONPATH=.:src pytest   tests/query_preparation   tests/api/test_query_candidates.py   tests/orchestration/test_query_preparation_health.py   tests/thundercompute/test_query_preparation_api.py -v
```

Expected: all PASS.

- [ ] **Step 3: Verify no runtime Transformer/Qwen import**

Run:

```bash
rg -n "transformers|AutoModel|AutoTokenizer|Qwen3" src/hcmai
```

Expected: model name may appear in config only; no `transformers`/model loading import under `src/hcmai`.

- [ ] **Step 4: Verify stateless API**

Run:

```bash
rg -n "candidate_id|search_id|query_session|candidate_session" src/hcmai/query_preparation src/hcmai/api
```

Expected: no server-side candidate/session mechanism.

- [ ] **Step 5: Record GPU smoke-test latency and candidate shape in the implementation PR/benchmark notes**

Record actual `query_preparation_ms`, GPU model revision, and one sanitized input/output shape. Do not change product defaults based on a single query.
