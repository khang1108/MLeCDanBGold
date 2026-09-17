# KIS Temporal Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the initial KIS resolver reliably decompose one natural-language narrative into multiple chronological `KISResolutionEvent` objects when the narrative contains distinct sequential moments, without adding deterministic splitters or changing downstream temporal retrieval.

**Architecture:** Keep semantic decomposition entirely inside the existing LLM resolver boundary. Strengthen the contract twice—first in `KIS_RESOLVER_SYSTEM_PROMPT` / `build_kis_intent_messages()`, then in the Pydantic JSON schema sent to the model—while leaving `KISIntentResolver` responsible only for canonical IDs, bindings, revision, and adjacent temporal edges.

**Tech Stack:** Python, Pydantic v2, pytest, existing provider-agnostic `LLMClient`, OpenAI-compatible structured-output JSON schema.

**Spec:** `docs/superpowers/specs/2026-09-17-kis-temporal-decomposition-design.md`

## Global Constraints

- One clue is not one event: one natural-language clue may resolve to one or many chronological events.
- Sequential moments must become separate `KISResolutionEvent` list items; simultaneous actions from the same moment may remain together.
- Do not add regex/keyword event splitting, sentence splitting, a deterministic fallback builder, or a second LLM decomposition pass.
- Do not change retrieval, fusion, temporal DP, EventTrail, scoped `E#:` updates, or `/llm-rewrite` behavior.
- Do not move semantic splitting into `KISIntentResolver`; the resolver continues to canonicalize validated model output only.
- Keep the cycling query out of the few-shot example so the acceptance case remains independent of the prompt demonstration.
- Deterministic CI tests must not call a live LLM endpoint. The live cycling probe is a manual acceptance gate only.
- Do not assert exact generated English wording in the live probe; assert only event cardinality/order and adjacent temporal-edge structure.
- `src/hcmai/kis/resolver.py` is not a planned production-code edit. Change it only if the focused regression tests expose a genuine canonicalization defect.

---

## File Structure

The implementation is intentionally small and keeps each responsibility in its current module.

```text
src/hcmai/kis/
├── prompts.py      # initial resolver semantic instructions + one-clue→N-events few-shot
├── models.py       # structured-output schema descriptions shown to the LLM
└── resolver.py     # unchanged canonicalization boundary

tests/hcmai/kis/
└── test_temporal_decomposition.py
    # focused contract + resolver regression tests for this bug
```

If `tests/hcmai/kis/` already exists in the real repository, add the focused file there. If the source snapshot being inspected has no `tests/` tree, create this file only in the real implementation repository; do not invent a second test root elsewhere.

---

### Task 1: Strengthen the Initial Resolver Prompt Contract

**Files:**
- Create: `tests/hcmai/kis/test_temporal_decomposition.py`
- Modify: `src/hcmai/kis/prompts.py`

**Interfaces:**
- Consumes: `build_kis_intent_messages(clues: Sequence[str]) -> list[dict[str, str]]` and `KIS_RESOLVER_SYSTEM_PROMPT`.
- Produces: prompt messages that explicitly distinguish clues from temporal events and include one concise one-clue→three-event JSON example.
- Preserves: `build_kis_scoped_messages()` and `build_kis_global_rewrite_messages()` unchanged.

- [ ] **Step 1: Create the focused test directory and write failing prompt-contract tests**

Create `tests/hcmai/kis/test_temporal_decomposition.py` with:

```python
from hcmai.kis.prompts import (
    KIS_RESOLVER_SYSTEM_PROMPT,
    build_kis_intent_messages,
)


def test_initial_prompt_states_one_clue_can_produce_multiple_events() -> None:
    messages = build_kis_intent_messages(
        ["A person enters a room. Then the person sits down."]
    )

    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "A clue is not an event" in system
    assert "one clue may resolve to one or many chronological events" in system
    assert "single narrative containing multiple chronological moments" in user
    assert "One clue may produce multiple events" in user


def test_initial_prompt_requires_sequential_moments_to_be_separate_events() -> None:
    system = KIS_RESOLVER_SYSTEM_PROMPT

    assert "TEMPORAL DECOMPOSITION" in system
    assert "Each distinct sequential moment MUST be a separate object" in system
    assert "Do NOT combine several sequential moments into a single event text" in system
    assert "Merge actions only when they occur as part of the same temporal moment" in system


def test_initial_prompt_contains_non_cycling_one_clue_multi_event_example() -> None:
    system = KIS_RESOLVER_SYSTEM_PROMPT

    assert "A man first enters a room carrying a box" in system
    assert "Then he puts the box on a table" in system
    assert "Finally a woman opens the box" in system
    assert '"events": [' in system
    assert system.count('"text":') >= 3
```

These tests deliberately check the semantic contract rather than the entire prompt string so harmless wording changes do not require snapshot rewrites.

- [ ] **Step 2: Run the prompt tests and verify RED**

Run:

```bash
PYTHONPATH=src pytest -q \
  tests/hcmai/kis/test_temporal_decomposition.py \
  -k 'prompt'
```

Expected: FAIL against the pre-fix `src_v1.4` prompt because it does not explicitly state the clue/event distinction, does not strongly require one event object per sequential moment, and does not contain the one-clue→multi-event few-shot.

Do not modify production code until the failure is observed and attributable to these missing prompt semantics.

- [ ] **Step 3: Replace the initial resolver prompt with the explicit decomposition contract**

In `src/hcmai/kis/prompts.py`, keep the module structure and scoped/global prompts intact. Replace only `KIS_RESOLVER_SYSTEM_PROMPT` with:

```python
KIS_RESOLVER_SYSTEM_PROMPT = """You are an expert video retrieval query intent resolver for multimodal search competitions.
Your task is to analyze either a single narrative containing multiple chronological moments or an ordered history of clues revealed over time for a video segment, then resolve the input into a clean semantic resolution.
A clue is not an event: one clue may resolve to one or many chronological events.

Required JSON Structure:
You MUST output a JSON object containing ALL of the following fields:
- "query_text": concise string summarizing the full video query (required)
- "entities": list of tracked entities [{"kind": "person"|"object"|"place"|"text"|"other", "description": "..."}, ...], or [] if none
- "events": list of chronological events [{"text": "...", "entity_indices": [...]}, ...] (at least 1 event required)

Guidelines:
1. ENTITY TRACKING: Identify key entities (persons, objects, places, texts) appearing across the clue history or narrative.
2. COREFERENCE & PRONOUNS: Resolve ambiguous pronouns ("he", "she", "they", "it", "that thing") to their canonical entity descriptions.
3. CONTRADICTIONS & CORRECTIONS: If a subsequent clue explicitly corrects an earlier clue (e.g., "actually orange, not red"), the canonical query_text, entities, and events MUST reflect the correction instead of concatenating contradictions.
4. TEMPORAL ORDERING: Output events in strictly chronological order even when clues reveal them out of order.
5. TEMPORAL DECOMPOSITION:
   - Identify distinct moments or actions that occur sequentially in the described video.
   - Each distinct sequential moment MUST be a separate object in the "events" array.
   - Do NOT combine several sequential moments into a single event text.
   - Merge actions only when they occur as part of the same temporal moment or simultaneous scene.
   - Temporal transition phrases such as "initially", "then", "afterward", "later", "bắt đầu khi", "sau đó", "ngay sau đó", and "tiếp theo" are examples of evidence for temporal change, not an exhaustive keyword list.
6. EVENT TEXT: Write a concise, self-contained English description of exactly one temporal moment suitable for dense and lexical visual retrieval. Resolve pronouns to explicit entities where useful. Never write an Event 1/Event 2/Event 3 list or a multi-stage narrative inside a single event text.
7. ENTITY INDICES: For each event, entity_indices must be a list of 0-based indices into the entities array. If entities is empty [], entity_indices MUST be empty [] for all events. The same entity index may appear in multiple events when that entity persists across time.
8. RESTRICTIONS: Do not emit timestamps, candidate IDs, retrieval translations, event IDs, entity IDs, clue history copies, or edges.

Example — one clue can contain multiple events:
Input:
A man first enters a room carrying a box. Then he puts the box on a table. Finally a woman opens the box.

Output:
{
  "query_text": "A man enters a room carrying a box, places it on a table, and a woman later opens the box.",
  "entities": [
    {"kind": "person", "description": "a man carrying and placing a box"},
    {"kind": "object", "description": "a box"},
    {"kind": "person", "description": "a woman who opens the box"}
  ],
  "events": [
    {"text": "A man enters a room carrying a box.", "entity_indices": [0, 1]},
    {"text": "The man places the box on a table.", "entity_indices": [0, 1]},
    {"text": "A woman opens the box on the table.", "entity_indices": [2, 1]}
  ]
}
"""
```

Do not add the cycling query to this prompt.

- [ ] **Step 4: Update only the initial user-message framing**

Replace the `user_prompt` body in `build_kis_intent_messages()` with:

```python
user_prompt = (
    "The input may be either a single narrative containing multiple chronological "
    "moments or an ordered history of progressively revealed clues. One clue may "
    "produce multiple events when it describes sequential moments.\n\n"
    f"Analyze the following {len(clues)} clue(s) for a video segment and resolve "
    f"them into a semantic KISResolution:\n\n{formatted_clues}"
)
```

Keep the `Clue 1`, `Clue 2`, ... formatting for compatibility with progressive clue history.

- [ ] **Step 5: Run prompt tests and verify GREEN**

Run:

```bash
PYTHONPATH=src pytest -q \
  tests/hcmai/kis/test_temporal_decomposition.py \
  -k 'prompt'
```

Expected: PASS.

- [ ] **Step 6: Run a narrow syntax/import check**

Run:

```bash
PYTHONPATH=src python -m compileall -q src/hcmai/kis/prompts.py
PYTHONPATH=src python - <<'PY'
from hcmai.kis.prompts import build_kis_intent_messages
messages = build_kis_intent_messages(["A person enters. Then another person leaves."])
assert len(messages) == 2
assert messages[0]["role"] == "system"
assert messages[1]["role"] == "user"
PY
```

Expected: both commands exit 0.

- [ ] **Step 7: Commit Task 1**

```bash
git add src/hcmai/kis/prompts.py tests/hcmai/kis/test_temporal_decomposition.py
git commit -m "fix: strengthen KIS temporal decomposition prompt"
```

---

### Task 2: Reinforce the Same Contract in the Structured-Output Schema

**Files:**
- Modify: `tests/hcmai/kis/test_temporal_decomposition.py`
- Modify: `src/hcmai/kis/models.py`

**Interfaces:**
- Consumes: `KISResolution.model_json_schema()` as the schema passed by `LLMClient.generate_structured()`.
- Produces: JSON-schema descriptions for `KISResolution.events`, `KISResolutionEvent.text`, and `KISResolutionEvent.entity_indices` that reinforce the same temporal semantics as the prompt.
- Preserves: all validation types, `min_length=1`, event-count guard ownership, and server-owned canonical IDs/edges.

- [ ] **Step 1: Add failing schema-description tests**

Append to `tests/hcmai/kis/test_temporal_decomposition.py`:

```python
from hcmai.kis.models import KISResolution


def test_resolution_schema_describes_events_as_distinct_temporal_moments() -> None:
    schema = KISResolution.model_json_schema()
    events = schema["properties"]["events"]

    description = events["description"]
    assert "One input clue may produce multiple events" in description
    assert "sequential moments must be separate list items" in description


def test_resolution_event_schema_limits_text_to_one_temporal_moment() -> None:
    schema = KISResolution.model_json_schema()
    event_schema = schema["$defs"]["KISResolutionEvent"]

    text_description = event_schema["properties"]["text"]["description"]
    indices_description = event_schema["properties"]["entity_indices"]["description"]

    assert "exactly one temporal moment" in text_description
    assert "Never combine distinct sequential moments" in text_description
    assert "Zero-based indices into the entities array" in indices_description
    assert "same entity index may appear in multiple events" in indices_description
```

- [ ] **Step 2: Run schema tests and verify RED**

Run:

```bash
PYTHONPATH=src pytest -q \
  tests/hcmai/kis/test_temporal_decomposition.py \
  -k 'schema'
```

Expected: FAIL against the pre-fix models because the relevant Pydantic fields do not yet expose these semantic descriptions.

- [ ] **Step 3: Add semantic `Field(description=...)` metadata to `KISResolutionEvent`**

In `src/hcmai/kis/models.py`, set the two fields exactly as follows:

```python
class KISResolutionEvent(BaseModel):
    """Semantic event returned by LLM with zero-based entity index references."""

    model_config = ConfigDict(extra="forbid")
    text: NonBlank = Field(
        description=(
            "Self-contained English retrieval description of exactly one temporal "
            "moment or simultaneous scene. Never combine distinct sequential moments "
            "inside one event text."
        )
    )
    entity_indices: list[int] = Field(
        default_factory=list,
        description=(
            "Zero-based indices into the entities array for entities participating in "
            "this event. The same entity index may appear in multiple events when the "
            "entity persists across time."
        ),
    )
```

Do not add a validator that infers event count from sentences or transition words.

- [ ] **Step 4: Add the event-list semantic description to `KISResolution`**

Change only the `events` field to:

```python
events: list[KISResolutionEvent] = Field(
    min_length=1,
    description=(
        "Chronologically ordered distinct temporal moments. One input clue may "
        "produce multiple events; sequential moments must be separate list items."
    ),
)
```

Do not move `DEFAULT_MAX_TEMPORAL_EVENT_COUNT` into this schema. The existing resolver/domain guard remains the authority for maximum cardinality.

- [ ] **Step 5: Run schema tests and all focused contract tests**

Run:

```bash
PYTHONPATH=src pytest -q tests/hcmai/kis/test_temporal_decomposition.py
```

Expected: all prompt and schema tests added so far PASS.

- [ ] **Step 6: Verify the schema actually emitted by `LLMClient` contains the descriptions**

Run:

```bash
PYTHONPATH=src python - <<'PY'
from hcmai.kis.models import KISResolution

schema = KISResolution.model_json_schema()
assert "multiple events" in schema["properties"]["events"]["description"]
assert "exactly one temporal moment" in (
    schema["$defs"]["KISResolutionEvent"]["properties"]["text"]["description"]
)
print("schema contract ok")
PY
```

Expected output:

```text
schema contract ok
```

- [ ] **Step 7: Commit Task 2**

```bash
git add src/hcmai/kis/models.py tests/hcmai/kis/test_temporal_decomposition.py
git commit -m "fix: describe KIS event decomposition in schema"
```

---

### Task 3: Lock Existing Resolver Canonicalization and Validate the Reported Failure End-to-End

**Files:**
- Modify: `tests/hcmai/kis/test_temporal_decomposition.py`
- Verify only: `src/hcmai/kis/resolver.py`

**Interfaces:**
- Consumes: `KISIntentResolver.resolve_initial(text: str, revision: int) -> KISIntent` and semantic `KISResolution` objects returned by a fake LLM.
- Produces: regression coverage proving that three model events become `E1`, `E2`, `E3` with adjacent edges and shared canonical entity bindings, while one simultaneous event remains one event with no edge.
- Preserves: `KISIntentResolver` source unchanged unless these tests expose a real defect.

- [ ] **Step 1: Add a minimal fake LLM test double and multi-event canonicalization regression**

Append to `tests/hcmai/kis/test_temporal_decomposition.py`:

```python
from typing import Any

from hcmai.kis.models import KISResolution
from hcmai.kis.resolver import KISIntentResolver


class FakeStructuredLLM:
    def __init__(self, resolution: KISResolution) -> None:
        self._resolution = resolution

    def generate_structured(
        self,
        messages: Any,
        response_model: type[KISResolution],
        **kwargs: Any,
    ) -> KISResolution:
        assert response_model is KISResolution
        return self._resolution


def test_resolver_canonicalizes_three_model_events_into_adjacent_temporal_chain() -> None:
    resolution = KISResolution.model_validate(
        {
            "query_text": "A man enters, places a box, then a woman opens it.",
            "entities": [
                {"kind": "person", "description": "a man"},
                {"kind": "object", "description": "a box"},
                {"kind": "person", "description": "a woman"},
            ],
            "events": [
                {"text": "A man enters a room carrying a box.", "entity_indices": [0, 1]},
                {"text": "The man places the box on a table.", "entity_indices": [0, 1]},
                {"text": "A woman opens the box on the table.", "entity_indices": [2, 1]},
            ],
        }
    )

    intent = KISIntentResolver(FakeStructuredLLM(resolution)).resolve_initial(
        "narrative",
        revision=1,
    )

    assert [event.id for event in intent.events] == ["E1", "E2", "E3"]
    assert [(edge.source, edge.target) for edge in intent.temporal_edges] == [
        ("E1", "E2"),
        ("E2", "E3"),
    ]
    assert [binding.entity_id for binding in intent.events[0].bindings] == ["X1", "X2"]
    assert [binding.entity_id for binding in intent.events[1].bindings] == ["X1", "X2"]
    assert [binding.entity_id for binding in intent.events[2].bindings] == ["X3", "X2"]
```

This is a characterization/regression test for existing server-owned behavior; no resolver production-code change is expected.

- [ ] **Step 2: Add the simultaneous single-event regression**

Append:

```python
def test_resolver_keeps_one_simultaneous_scene_as_one_event() -> None:
    resolution = KISResolution.model_validate(
        {
            "query_text": "A woman talks while holding a cup.",
            "entities": [
                {"kind": "person", "description": "a woman"},
                {"kind": "object", "description": "a cup"},
            ],
            "events": [
                {
                    "text": "A woman talks while holding a cup.",
                    "entity_indices": [0, 1],
                }
            ],
        }
    )

    intent = KISIntentResolver(FakeStructuredLLM(resolution)).resolve_initial(
        "one simultaneous scene",
        revision=1,
    )

    assert [event.id for event in intent.events] == ["E1"]
    assert intent.temporal_edges == []
```

- [ ] **Step 3: Run the resolver regressions**

Run:

```bash
PYTHONPATH=src pytest -q tests/hcmai/kis/test_temporal_decomposition.py
```

Expected: PASS without modifying `src/hcmai/kis/resolver.py`.

If either resolver regression fails, stop and diagnose the canonicalization defect before changing `resolver.py`; do not compensate by weakening the tests or adding semantic splitting to the resolver.

- [ ] **Step 4: Run focused existing KIS resolver/model tests from the real repository**

Run whichever focused files exist in the repository, preferring the known resolver path:

```bash
PYTHONPATH=src pytest -q \
  tests/hcmai/kis/test_temporal_decomposition.py \
  tests/hcmai/kis/test_resolver.py
```

If additional existing focused KIS model/prompt tests are present, include them in the same run rather than replacing this command with a repository-wide suite.

Expected: PASS. Existing correction/coreference and server-owned ID/edge behavior must remain green.

- [ ] **Step 5: Run static/syntax verification for the KIS package**

Run:

```bash
PYTHONPATH=src python -m compileall -q src/hcmai/kis
```

Expected: exit 0.

- [ ] **Step 6: Run the manual live-model acceptance probe with the reported cycling narrative**

This step is intentionally outside deterministic CI. Run it only in the real environment where `HCMAI_LLM_*` variables and the configured model endpoint are available:

```bash
PYTHONPATH=src python - <<'PY'
from hcmai.common.environment import load_repository_environment
from hcmai.inference.config import load_llm_endpoint
from hcmai.inference.llm import LLMClient
from hcmai.kis.resolver import KISIntentResolver

load_repository_environment()

query = (
    "Đoạn clip bắt đầu khi có 1 người đang phất cờ về đích đến trong chặng đua xe đạp quanh hồ. "
    "Ngay sau đó, một đoàn đông vận động viên xe đạp đang vào cua giữa khung cảnh đường phố rợp bóng cây. "
    "Một người đứng bên đường, tay trái cầm điện thoại để ghi hình, còn tay phải cầm một cây gậy có gắn "
    "thiết bị quay ở đầu hướng về phía đoàn đua."
)

intent = KISIntentResolver(LLMClient(load_llm_endpoint())).resolve_initial(
    query,
    revision=1,
)

print("events:", len(intent.events))
for event in intent.events:
    print(event.id, event.text)
print("edges:", [(edge.source, edge.target) for edge in intent.temporal_edges])

assert len(intent.events) == 3
assert [(edge.source, edge.target) for edge in intent.temporal_edges] == [
    ("E1", "E2"),
    ("E2", "E3"),
]
PY
```

Expected semantic decomposition, without asserting exact English wording:

```text
E1: finish-line flag / race-finishing moment
E2: group of cyclists entering a turn
E3: roadside spectator filming the cyclists
```

Expected assertions: PASS.

If this probe still returns one event, record the structured `KISResolution` output and model identifier. Do not add a deterministic splitter; treat the failure as model/prompt quality evidence for the next design iteration.

- [ ] **Step 7: Verify no forbidden fallback or unrelated subsystem change was introduced**

Run:

```bash
git diff -- src/hcmai/kis/prompts.py src/hcmai/kis/models.py src/hcmai/kis/resolver.py
git diff --name-only
```

Expected production-code diff:

```text
src/hcmai/kis/prompts.py
src/hcmai/kis/models.py
```

`src/hcmai/kis/resolver.py`, retrieval, temporal DP, EventTrail, scoped resolver, and global rewriter should not appear unless a separately diagnosed defect required an explicit follow-up change.

- [ ] **Step 8: Commit Task 3**

```bash
git add tests/hcmai/kis/test_temporal_decomposition.py
git commit -m "test: lock KIS temporal decomposition regressions"
```

---

## Final Verification Gate

Before declaring the fix complete, run all of the following from the real repository:

```bash
PYTHONPATH=src pytest -q \
  tests/hcmai/kis/test_temporal_decomposition.py \
  tests/hcmai/kis/test_resolver.py

PYTHONPATH=src python -m compileall -q src/hcmai/kis

git diff --check
```

Then run the live cycling probe from Task 3 Step 6 in the configured model environment.

Completion requires:

1. deterministic focused tests pass;
2. existing resolver regressions pass;
3. package compilation passes;
4. `git diff --check` is clean;
5. live cycling narrative resolves to three chronological events;
6. temporal edges are exactly `E1→E2` and `E2→E3`;
7. no deterministic splitter/fallback/two-pass decomposition exists;
8. no retrieval, DP, EventTrail, scoped-update, or global-rewrite code changed as part of this fix.

