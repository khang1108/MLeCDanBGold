# KIS Initial Resolver Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make initial natural-language KIS resolution fast and bounded by reducing the LLM task to concise ordered event decomposition only.

**Architecture:** The LLM returns only a bounded list of short event texts. `KISIntentResolver` owns all canonical graph fields: IDs, empty initial entity/binding state, canonical query text, revision, and adjacent temporal edges. Scoped patches and global rewrites keep their current richer contracts.

**Tech Stack:** Python 3, Pydantic v2, pytest, existing provider-agnostic `LLMClient`.

**Spec:** `docs/superpowers/specs/2026-09-17-kis-initial-resolver-simplification-design.md`

## Global Constraints

- Initial natural-language resolution must not extract entities or emit entity bindings.
- Initial model output contains only ordered event texts.
- Each accepted initial event text is non-blank and at most 240 characters.
- `KISIntentResolver` passes `temperature=0.0` and `max_tokens=512` explicitly.
- A continuous camera shot may contain multiple retrieval events; event boundary means distinct retrievable visual moment, not shot boundary.
- No regex splitter, deterministic semantic fallback, second LLM pass, retrieval/DP/EventTrail change, or model-provider change.
- Explicit `E#:` initial input, scoped patching, and `/llm-rewrite` behavior remain unchanged.
- The exported `src_v1.5` snapshot has no `.git` directory and no test tree; execute commit steps in the real checkout. In the snapshot, create the same test paths and skip only the commit command.

---

## File Structure

**Create**
- `tests/kis/test_initial_resolution_contract.py` — event-only output schema and hard length/cardinality bounds.
- `tests/kis/test_initial_prompt.py` — prompt semantics for visual-moment decomposition and continuous-shot separation.
- `tests/kis/test_initial_resolver.py` — resolver call contract and deterministic canonicalization.

**Modify**
- `src/hcmai/kis/models.py` — replace the old initial semantic resolution models with event-only initial resolution models.
- `src/hcmai/kis/prompts.py` — replace the initial entity-heavy prompt with a concise event-decomposition prompt and builder.
- `src/hcmai/kis/resolver.py` — call the event-only schema with a 512-token budget and canonicalize server-owned fields.
- `src/hcmai/kis/__init__.py` — export the new initial-resolution types and remove dead old initial-resolution exports.

**Do not modify unless a failing regression proves it is necessary**
- `src/hcmai/kis/scoped_resolver.py`
- `src/hcmai/kis/rewriter.py`
- `src/hcmai/orchestration/pipeline.py`
- retrieval, temporal DP, EventTrail, or frontend files.

---

### Task 1: Introduce the bounded event-only initial response contract

**Files:**
- Create: `tests/kis/test_initial_resolution_contract.py`
- Modify: `src/hcmai/kis/models.py:23-69`

**Interfaces:**
- Produces: `KISInitialResolutionEvent(text: InitialEventText)`.
- Produces: `KISInitialResolution(events: list[KISInitialResolutionEvent])` with `1 <= len(events) <= DEFAULT_MAX_TEMPORAL_EVENT_COUNT`.
- `InitialEventText` is stripped, non-blank, and `max_length=240`.
- Later tasks consume `KISInitialResolution` as the `LLMClient.generate_structured()` response model.

- [ ] **Step 1: Write failing contract tests**

```python
# tests/kis/test_initial_resolution_contract.py
import pytest
from pydantic import ValidationError

from hcmai.common.config import DEFAULT_MAX_TEMPORAL_EVENT_COUNT
from hcmai.kis.models import KISInitialResolution, KISInitialResolutionEvent


def test_initial_resolution_schema_contains_only_events() -> None:
    schema = KISInitialResolution.model_json_schema()
    assert set(schema["properties"]) == {"events"}


def test_initial_event_rejects_text_over_240_characters() -> None:
    with pytest.raises(ValidationError):
        KISInitialResolutionEvent(text="x" * 241)


def test_initial_resolution_rejects_too_many_events() -> None:
    events = [
        {"text": f"Distinct visual moment {index}."}
        for index in range(DEFAULT_MAX_TEMPORAL_EVENT_COUNT + 1)
    ]
    with pytest.raises(ValidationError):
        KISInitialResolution(events=events)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=src pytest -q tests/kis/test_initial_resolution_contract.py
```

Expected: collection/import failure because `KISInitialResolution` and `KISInitialResolutionEvent` do not exist yet.

- [ ] **Step 3: Add the minimal event-only models**

In `src/hcmai/kis/models.py`, add:

```python
InitialEventText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=240),
]


class KISInitialResolutionEvent(BaseModel):
    """One concise retrievable visual moment returned by the initial resolver."""

    model_config = ConfigDict(extra="forbid")
    text: InitialEventText = Field(
        description=(
            "One concise self-contained English description of exactly one distinct "
            "retrievable visual moment. Do not combine sequential moments or explain reasoning."
        )
    )


class KISInitialResolution(BaseModel):
    """Event-only output contract for initial natural-language KIS resolution."""

    model_config = ConfigDict(extra="forbid")
    events: list[KISInitialResolutionEvent] = Field(
        min_length=1,
        max_length=DEFAULT_MAX_TEMPORAL_EVENT_COUNT,
        description="Chronologically ordered distinct retrievable visual moments.",
    )
```

Do not delete `KISResolution*` yet; Task 4 removes them after the resolver has migrated.

- [ ] **Step 4: Run the contract tests and verify GREEN**

```bash
PYTHONPATH=src pytest -q tests/kis/test_initial_resolution_contract.py
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/hcmai/kis/models.py tests/kis/test_initial_resolution_contract.py
git commit -m "refactor: add bounded KIS initial event contract"
```

---

### Task 2: Replace the initial prompt with one narrow event-decomposition task

**Files:**
- Create: `tests/kis/test_initial_prompt.py`
- Modify: `src/hcmai/kis/prompts.py:1-71`

**Interfaces:**
- Produces: `KIS_INITIAL_RESOLVER_SYSTEM_PROMPT`.
- Produces: `build_kis_initial_messages(clues: Sequence[str]) -> list[dict[str, str]]`.
- The prompt requests only `{"events": [{"text": "..."}]}` semantics through the response schema; it must not ask for entities, `entity_indices`, query summarization, contradiction tables, IDs, bindings, or edges.

- [ ] **Step 1: Write failing prompt tests**

```python
# tests/kis/test_initial_prompt.py
from hcmai.kis.prompts import (
    KIS_INITIAL_RESOLVER_SYSTEM_PROMPT,
    build_kis_initial_messages,
)


def test_initial_prompt_defines_event_as_retrievable_visual_moment_not_shot() -> None:
    prompt = KIS_INITIAL_RESOLVER_SYSTEM_PROMPT.lower()
    assert "retrievable visual moment" in prompt
    assert "continuous camera" in prompt
    assert "separate events" in prompt


def test_initial_prompt_does_not_request_entity_graph_work() -> None:
    prompt = KIS_INITIAL_RESOLVER_SYSTEM_PROMPT.lower()
    forbidden = (
        "entity tracking",
        "entity_indices",
        "query_text",
        "contradictions & corrections",
    )
    assert all(term not in prompt for term in forbidden)


def test_initial_messages_keep_one_narrative_as_input_without_pre_splitting() -> None:
    messages = build_kis_initial_messages((
        "The camera pans from a framed photograph to an artisan drawing a portrait.",
    ))
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Clue 1:" in messages[1]["content"]
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
PYTHONPATH=src pytest -q tests/kis/test_initial_prompt.py
```

Expected: import failure because the new prompt constant/builder do not exist.

- [ ] **Step 3: Replace only the initial prompt/builder**

Use this compact prompt content in `src/hcmai/kis/prompts.py`:

```python
KIS_INITIAL_RESOLVER_SYSTEM_PROMPT = """You decompose a video-search description into chronologically ordered, visually retrievable moments.

Rules:
- One event is one distinct retrievable visual moment.
- Sequential views or actions are separate events, even when one continuous camera shot or pan connects them.
- Details that are genuinely simultaneous in the same visual moment stay in one event.
- Write each event as one concise, self-contained English sentence suitable for visual/text retrieval.
- Preserve uncertainty; do not invent a specific tool, material, person, place, text, or action that the input does not establish.
- Do not explain reasoning, summarize the whole query, create entity tables, or put multiple stages inside one event text.

Example:
Input: The camera starts on a framed certificate, then pans right to a craftsperson engraving a metal plate with an unusual tool.
Events:
1. A framed certificate is visible on a work table.
2. A craftsperson engraves a metal plate with an unusual tool.
"""


def build_kis_initial_messages(clues: Sequence[str]) -> list[dict[str, str]]:
    """Build messages for bounded initial event decomposition."""
    formatted = "\n".join(
        f"Clue {index + 1}: {clue}" for index, clue in enumerate(clues)
    )
    return [
        {"role": "system", "content": KIS_INITIAL_RESOLVER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Return the chronological retrievable moments for this input. "
                "A single clue may contain multiple events.\n\n"
                f"{formatted}"
            ),
        },
    ]
```

Leave scoped/global prompt functions unchanged.

- [ ] **Step 4: Run prompt tests and verify GREEN**

```bash
PYTHONPATH=src pytest -q tests/kis/test_initial_prompt.py
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/hcmai/kis/prompts.py tests/kis/test_initial_prompt.py
git commit -m "refactor: narrow KIS initial resolver prompt"
```

---

### Task 3: Make `KISIntentResolver` use the bounded contract and own canonical graph construction

**Files:**
- Create: `tests/kis/test_initial_resolver.py`
- Modify: `src/hcmai/kis/resolver.py:13-122`

**Interfaces:**
- Consumes: `build_kis_initial_messages()` and `KISInitialResolution`.
- Calls: `LLMClient.generate_structured(messages, KISInitialResolution, temperature=0.0, max_tokens=512)`.
- Produces: `KISIntent` with `entities=[]`, all event `bindings=[]`, canonical `E1..En`, adjacent edges, and `query_text=" ".join(event.text ...)`.

- [ ] **Step 1: Write a fake LLM and failing resolver tests**

```python
# tests/kis/test_initial_resolver.py
from hcmai.kis.models import KISInitialResolution, KISInitialResolutionEvent
from hcmai.kis.resolver import KISIntentResolver


class CapturingLLM:
    def __init__(self, result: KISInitialResolution) -> None:
        self.result = result
        self.calls = []

    def generate_structured(
        self,
        messages,
        response_model,
        *,
        temperature=0.0,
        max_tokens=None,
    ):
        self.calls.append(
            {
                "messages": messages,
                "response_model": response_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self.result


def test_initial_resolver_uses_event_only_schema_and_512_token_budget() -> None:
    llm = CapturingLLM(
        KISInitialResolution(
            events=[KISInitialResolutionEvent(text="A framed photograph is visible.")]
        )
    )
    KISIntentResolver(llm).resolve_initial("Một bức ảnh trên bàn.", revision=1)

    call = llm.calls[0]
    assert call["response_model"] is KISInitialResolution
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 512


def test_initial_resolver_canonicalizes_two_events_without_entities() -> None:
    llm = CapturingLLM(
        KISInitialResolution(
            events=[
                KISInitialResolutionEvent(
                    text="A photograph shows a person shaking hands with Ho Chi Minh."
                ),
                KISInitialResolutionEvent(
                    text="An artisan draws that person's portrait with an unusual pen."
                ),
            ]
        )
    )

    intent = KISIntentResolver(llm).resolve_initial("ignored by fake", revision=4)

    assert intent.revision == 4
    assert intent.entities == []
    assert [event.id for event in intent.events] == ["E1", "E2"]
    assert [event.bindings for event in intent.events] == [[], []]
    assert intent.query_text == (
        "A photograph shows a person shaking hands with Ho Chi Minh. "
        "An artisan draws that person's portrait with an unusual pen."
    )
    assert [(edge.source, edge.target) for edge in intent.temporal_edges] == [
        ("E1", "E2")
    ]
```

- [ ] **Step 2: Run resolver tests and verify RED**

```bash
PYTHONPATH=src pytest -q tests/kis/test_initial_resolver.py
```

Expected: failures because the current resolver requests `KISResolution`, omits `max_tokens`, and constructs entities/bindings from model output.

- [ ] **Step 3: Implement the minimal resolver migration**

Change imports to use:

```python
from hcmai.kis.models import (
    KISEvent,
    KISInitialResolution,
    KISIntent,
    KISTemporalEdge,
)
from hcmai.kis.prompts import build_kis_initial_messages
```

Replace the generation/canonicalization body with:

```python
messages = build_kis_initial_messages(normalized)
resolution = self._llm.generate_structured(
    messages,
    KISInitialResolution,
    temperature=0.0,
    max_tokens=512,
)

events = [
    KISEvent(id=f"E{index + 1}", text=event.text, bindings=[])
    for index, event in enumerate(resolution.events)
]
query_text = " ".join(event.text for event in events if event.text)
edges = [
    KISTemporalEdge(source=f"E{i}", target=f"E{i + 1}")
    for i in range(1, len(events))
]

try:
    return KISIntent(
        revision=revision,
        query_text=query_text,
        entities=[],
        events=events,
        temporal_edges=edges,
    )
except ValueError as exc:
    raise KISResolutionError(
        f"Canonical intent validation failed: {exc}"
    ) from exc
```

Retain input normalization and revision validation. Remove entity-index validation because the initial model no longer emits entity references.

- [ ] **Step 4: Run resolver + contract + prompt tests**

```bash
PYTHONPATH=src pytest -q \
  tests/kis/test_initial_resolution_contract.py \
  tests/kis/test_initial_prompt.py \
  tests/kis/test_initial_resolver.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/hcmai/kis/resolver.py tests/kis/test_initial_resolver.py
git commit -m "refactor: simplify KIS initial resolution"
```

---

### Task 4: Delete the superseded initial semantic-resolution types and exports

**Files:**
- Modify: `src/hcmai/kis/models.py:28-69,201-211`
- Modify: `src/hcmai/kis/__init__.py`
- Modify: `src/hcmai/kis/prompts.py` — delete `KIS_RESOLVER_SYSTEM_PROMPT` and `build_kis_intent_messages` if still present after Task 2.

**Interfaces:**
- Deletes: `KISResolutionEntity`, `KISResolutionEvent`, `KISResolution`.
- Keeps: `KISResolutionError` because scoped/global flows use this error type.
- Keeps scoped/global rewrite models untouched.

- [ ] **Step 1: Add a dead-surface regression check**

Append to `tests/kis/test_initial_resolution_contract.py`:

```python
def test_legacy_initial_semantic_models_are_not_exported() -> None:
    import hcmai.kis as kis

    assert not hasattr(kis, "KISResolution")
    assert not hasattr(kis, "KISResolutionEntity")
    assert not hasattr(kis, "KISResolutionEvent")
```

- [ ] **Step 2: Run the check and verify RED**

```bash
PYTHONPATH=src pytest -q tests/kis/test_initial_resolution_contract.py::test_legacy_initial_semantic_models_are_not_exported
```

Expected: FAIL because the legacy models are still exported.

- [ ] **Step 3: Remove the dead models/prompts/exports**

Delete the three old initial semantic response classes from `models.py`; remove them from `__all__` and `src/hcmai/kis/__init__.py`. Remove the superseded initial prompt constant/builder if Task 2 left compatibility aliases.

- [ ] **Step 4: Prove no production caller still references the superseded surface**

Run:

```bash
rg -n "KISResolutionEntity|KISResolutionEvent|KISResolution\b|build_kis_intent_messages|KIS_RESOLVER_SYSTEM_PROMPT" \
  src/hcmai \
  --glob '!kis/resolver.py'
```

Expected: no matches for deleted types/functions/constants. `KISResolutionError` matches are allowed and must remain.

Then run:

```bash
PYTHONPATH=src pytest -q tests/kis
```

Expected: all KIS tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/hcmai/kis tests/kis
git commit -m "refactor: remove legacy KIS initial semantic graph output"
```

---

### Task 5: Verify the exact regression cases and repository health

**Files:**
- No production files unless verification exposes a defect in the approved scope.

**Interfaces:**
- Verifies the bounded initial-resolver contract end-to-end at the resolver boundary.
- Does not turn live-model behavior into a flaky CI assertion.

- [ ] **Step 1: Run all focused automated tests**

```bash
PYTHONPATH=src pytest -q tests/kis
```

Expected: all pass.

- [ ] **Step 2: Compile the KIS package**

```bash
PYTHONPATH=src python -m compileall -q src/hcmai/kis
```

Expected: exit 0.

- [ ] **Step 3: Run static reference scans**

```bash
rg -n "generate_structured\(" src/hcmai/kis
rg -n "max_tokens=512" src/hcmai/kis/resolver.py
rg -n "entity_indices|ENTITY TRACKING|query_text" src/hcmai/kis/prompts.py
```

Expected:
- initial resolver has exactly one explicit `max_tokens=512` call;
- initial prompt has no entity-graph/query-summary instructions;
- scoped/global generation calls remain present and unchanged.

- [ ] **Step 4: Run two live acceptance probes against the configured model**

Run from the real environment where `HCMAI_LLM_*` is configured:

```bash
PYTHONPATH=src python - <<'PY'
from time import perf_counter
from hcmai.common.environment import load_repository_environment
load_repository_environment()
from hcmai.inference.config import load_llm_endpoint
from hcmai.inference.llm import LLMClient
from hcmai.kis.resolver import KISIntentResolver

queries = [
    "Đoạn clip bắt đầu khi có 1 người đang phất cờ về đích đến trong chặng đua xe đạp quanh hồ. Ngay sau đó, một đoàn đông vận động viên xe đạp đang vào cua giữa khung cảnh đường phố rợp bóng cây. Một người đứng bên đường, tay trái cầm điện thoại để ghi hình, còn tay phải cầm một cây gậy có gắn thiết bị quay ở đầu hướng về phía đoàn đua.",
    "Camera lia chậm trên mặt bàn làm việc, bắt đầu từ một bức ảnh/tranh một người đang bắt tay với Bác Hồ, sau đó chuyển sang phần bên phải là tay nghệ nhân đang vẽ lại chân dung người bắt tay Bác. Người nghệ nhân này vẽ bằng một cây bút đặc biệt trên một chất liệu đặc biệt.",
]

resolver = KISIntentResolver(LLMClient(load_llm_endpoint()))
for query in queries:
    started = perf_counter()
    intent = resolver.resolve_initial(query, revision=1)
    elapsed = perf_counter() - started
    print(f"latency={elapsed:.2f}s events={len(intent.events)}")
    for event in intent.events:
        print(event.id, len(event.text or ""), event.text)
    print([(e.source, e.target) for e in intent.temporal_edges])
    assert all(len(event.text or "") <= 240 for event in intent.events)
PY
```

Acceptance expectations:
- bicycle narrative: multiple concise ordered events, not one multi-stage essay;
- camera-pan narrative: at least two concise events, separating the reference image view from the artisan drawing moment;
- every event is <=240 characters;
- no repeated paragraphs or explanatory reasoning;
- record latency before/after, but do not hardcode a CI wall-clock threshold because hardware/provider throughput varies.

- [ ] **Step 5: Commit any test-only acceptance documentation if added in the real checkout**

If no files changed during verification, do not create an empty commit.

---

## Self-Review Result

- **Spec coverage:** all approved requirements are mapped to Tasks 1-5.
- **Type consistency:** `KISInitialResolution` and `KISInitialResolutionEvent` are introduced in Task 1 and consumed under the same names in later tasks.
- **Scope:** no change to scoped/global rewrite, retrieval, temporal DP, EventTrail, frontend, or generic inference server.
- **Failure bounding:** hard event-text length plus explicit 512-token generation budget prevents accepted initial intents from carrying essay-length event text even if the model degenerates.
