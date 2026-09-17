# KIS Initial Temporal Decomposition Design

**Date:** 2026-09-17  
**Scope:** bounded KIS resolver correction in `src_v1.4`  
**Status:** design approved in chat; implementation not started  

## 1. Problem

The initial KIS resolver can receive one natural-language clue that describes several chronological moments, but the current prompt can still collapse the complete narrative into a single `KISResolutionEvent`.

For an input such as:

```text
A person waves a finish flag. Immediately afterward a large group of cyclists
enters a turn. A roadside spectator records them with a phone and camera pole.
```

the desired semantic resolution is three chronological events, not one event whose `text` contains the entire story.

When the resolver returns only one event, the downstream server behaves correctly but can only execute single-event retrieval. The server already canonicalizes multiple returned events into `E1..En` and creates the adjacent temporal chain `E1 -> E2 -> ... -> En`; therefore the defect is at the semantic decomposition boundary, not in temporal DP/alignment.

## 2. Current Source State

Relevant files in `src_v1.4`:

```text
src/hcmai/kis/prompts.py
src/hcmai/kis/models.py
src/hcmai/kis/resolver.py
```

Current behavior:

- `build_kis_intent_messages()` frames input as an ordered history of `Clue N` items.
- `KIS_RESOLVER_SYSTEM_PROMPT` says to split sequential actions, but does not strongly state that one clue may contain many temporal events.
- `KISResolutionEvent.text` and `KISResolution.events` carry little semantic guidance in their JSON schema descriptions.
- `KISIntentResolver` already converts the returned event list into canonical `E1..En` IDs and a complete adjacent temporal chain.

The implementation must preserve the resolver's server-owned responsibilities. The model decides semantic decomposition; the server continues to own IDs, revision, bindings, and edges.

## 3. Goals

The change must make the initial resolver reliably understand that:

1. **A clue is not an event.** One clue may describe one or many chronological moments.
2. Distinct sequential moments become distinct event objects.
3. Simultaneous actions belonging to the same moment remain in one event.
4. Each event description is self-contained and retrieval-oriented.
5. Shared entities may appear in multiple events through `entity_indices`.
6. The existing downstream canonicalization and temporal search pipeline remain unchanged.

## 4. Non-Goals

This change does **not** introduce:

- deterministic sentence or keyword splitting;
- regex rules that create events from words such as `then`, `sau đó`, or `ngay sau đó`;
- a fallback deterministic intent builder;
- a second LLM decomposition pass;
- a new resolver service or API;
- changes to event IDs, entity IDs, revisions, bindings, or temporal-edge ownership;
- changes to scoped `E#:` updates or `/llm-rewrite` semantics;
- changes to temporal DP, retrieval fusion, EventTrail, or ranking;
- model-specific hacks for the cycling example.

## 5. Design Principle

Temporal decomposition remains an **LLM-owned semantic decision under an explicit output contract**.

The implementation strengthens the contract in two places:

```text
prompt instruction
      +
structured-output JSON schema descriptions
```

The server does not attempt to repair a one-event model response by splitting it afterward.

## 6. Prompt Contract

### 6.1 Input framing

The system and user messages must stop implying that one `Clue N` corresponds to one event.

The prompt must explicitly state that the input can be either:

- a single narrative containing several chronological moments; or
- an ordered history of progressively revealed clues.

Recommended wording:

```text
The input may be either a single narrative containing multiple chronological
moments or an ordered history of clues. A clue is not an event: one clue may
resolve to one or many chronological events.
```

`build_kis_intent_messages()` may continue formatting inputs as `Clue 1`, `Clue 2`, etc. for compatibility, but the surrounding instruction must make the distinction above explicit.

### 6.2 Temporal decomposition rule

The prompt must promote decomposition from a weak guideline to a hard semantic rule:

```text
TEMPORAL DECOMPOSITION:
- Identify distinct moments/actions that occur sequentially in the described video.
- Each distinct sequential moment MUST be a separate object in `events`.
- Do not encode several sequential moments inside one event's `text`.
- Merge actions only when they occur as part of the same temporal moment/scene.
- Temporal transition words such as "initially", "then", "afterward", "later",
  "bắt đầu khi", "sau đó", "ngay sau đó", and "tiếp theo" are examples of
  evidence for temporal change, not an exhaustive keyword list.
```

This is semantic guidance rather than deterministic parsing. A transition can exist without one of the example words, and an example word does not automatically imply that every clause must be split.

### 6.3 Event text rule

Each `events[i].text` must describe exactly one temporal moment or simultaneous scene:

```text
EVENT TEXT:
Write a concise, self-contained English description of exactly one temporal
moment, suitable for dense/lexical visual retrieval. Resolve pronouns to
explicit entities where useful. Never write an Event 1/Event 2/Event 3 list or
multi-stage narrative inside a single event text.
```

### 6.4 Few-shot example

The system prompt must include one short example demonstrating **one natural-language clue -> multiple event objects**.

The example must not reuse the cycling query used for acceptance testing.

Recommended example shape:

```text
Input:
A man first enters a room carrying a box. Then he puts the box on a table.
Finally a woman opens the box.

Output semantics:
- Event 1: man enters room carrying box
- Event 2: man places box on table
- Event 3: woman opens box
```

The actual example should use the same JSON shape as `KISResolution`, including `query_text`, `entities`, `events`, and valid zero-based `entity_indices`.

The example stays short to avoid dominating the model context or teaching a domain-specific pattern.

## 7. Structured-Output Schema Contract

`KISResolutionEvent` and `KISResolution.events` must carry semantic descriptions through Pydantic `Field(description=...)` so the JSON schema sent to the LLM reinforces the same contract.

### 7.1 `KISResolutionEvent.text`

Target semantics:

```python
text: NonBlank = Field(
    description=(
        "Self-contained English retrieval description of exactly one temporal "
        "moment or simultaneous scene. Never combine distinct sequential moments "
        "inside one event text."
    )
)
```

### 7.2 `KISResolutionEvent.entity_indices`

The existing list remains zero-based. Its schema description should state that the indices reference only entities participating in this event and may repeat across multiple events when the same entity persists through time.

### 7.3 `KISResolution.events`

Target semantics:

```python
events: list[KISResolutionEvent] = Field(
    min_length=1,
    description=(
        "Chronologically ordered distinct temporal moments. One input clue may "
        "produce multiple events; sequential moments must be separate list items."
    ),
)
```

No maximum is added here beyond the existing server-side `DEFAULT_MAX_TEMPORAL_EVENT_COUNT` guard.

## 8. Resolver Behavior

`KISIntentResolver` should remain structurally unchanged.

Given model output:

```text
events = [e1, e2, e3]
```

the server continues to create:

```text
E1
E2
E3

E1 -> E2
E2 -> E3
```

No post-hoc splitting is allowed.

The following existing validations remain authoritative:

- non-empty natural-language input;
- server-owned revision >= 1;
- maximum event cardinality;
- valid and non-duplicated entity indices;
- entity-index referential integrity;
- canonical sequential event IDs;
- complete adjacent temporal edge chain.

## 9. Failure Semantics

A syntactically valid one-event response is still a valid model response when the input genuinely describes one temporal moment. The server must not reject it merely because the source contains multiple sentences.

The system must therefore avoid hard validators such as:

```text
number_of_sentences > 1 => number_of_events > 1
```

or:

```text
contains("then") => must split
```

If the configured model still collapses an obviously sequential narrative after the strengthened contract, that is an inference-quality failure to observe and benchmark, not a reason to silently run a deterministic fallback.

## 10. Test Strategy

### 10.1 Deterministic unit tests

Unit tests must not depend on a live LLM endpoint.

They should verify:

1. `build_kis_intent_messages()` contains the explicit `one clue may produce multiple events` framing.
2. The system prompt contains the temporal decomposition invariant and one-clue/multi-event example.
3. `KISResolution.model_json_schema()` exposes the event-list and event-text semantic descriptions.
4. A fake LLM returning three events is canonicalized by `KISIntentResolver` into `E1`, `E2`, `E3` with edges `E1->E2`, `E2->E3`.
5. A fake LLM returning one simultaneous event remains one event with no temporal edge.
6. Shared entity indices across sequential events produce bindings to the same canonical server-owned entity IDs.

### 10.2 Live-model acceptance probe

A separate manual/integration probe may call the configured LLM endpoint with the known cycling narrative that exposed the defect.

Expected semantic decomposition:

```text
E1: finish-line flag / race-finishing moment
E2: group of cyclists entering a turn
E3: roadside spectator filming the cyclists
```

Acceptance checks:

```text
len(intent.events) == 3
intent.temporal_edges == [E1->E2, E2->E3]
```

The exact English wording of event text is not asserted.

This probe must not become a mandatory deterministic CI test because model/provider changes can make exact live inference behavior nondeterministic.

## 11. Regression Cases

The change is considered safe only if the following semantics remain possible:

| Input semantics | Expected decomposition |
|---|---:|
| One simple visual moment | 1 event |
| Multiple sentences describing the same simultaneous scene | 1 event |
| One clue with three sequential moments | 3 events |
| Sequential narrative sharing the same person/object | multiple events sharing entity references |
| Progressive clue history correcting a prior attribute | correction behavior remains unchanged |

The implementation must not improve multi-event decomposition by forcing every sentence into a separate event.

## 12. Files in Scope

Expected implementation surface:

```text
Modify:
  src/hcmai/kis/prompts.py
  src/hcmai/kis/models.py

Test/modify existing focused KIS resolver/prompt tests if present in the full repo.
If the submitted source snapshot has no tests directory, create tests only in the real repository rather than inventing a parallel test layout inside this archive.
```

`src/hcmai/kis/resolver.py` should require no semantic logic change unless a focused test exposes a genuine canonicalization defect.

## 13. Observability

No new telemetry subsystem is required.

For debugging and benchmark traces, existing query/intention logging should be sufficient to inspect:

```text
input query
resolved event count
resolved event texts
resolved temporal edges
```

Do not log hidden chain-of-thought or model reasoning. Only structured resolver output is relevant.

## 14. Acceptance Criteria

The change is complete only when all of the following are true:

1. The prompt explicitly states that one clue can contain multiple chronological events.
2. Sequential moments are required to become separate event objects.
3. Simultaneous descriptions are still allowed to remain one event.
4. The prompt includes one concise non-cycling one-clue -> multi-event few-shot example.
5. Event text is constrained to one temporal moment and self-contained retrieval wording.
6. Pydantic JSON-schema descriptions reinforce the same decomposition contract.
7. No regex/deterministic event splitter or fallback builder is introduced.
8. The resolver still owns only canonicalization, not semantic splitting.
9. Fake-LLM tests cover one-event, multi-event, and shared-entity cases.
10. The cycling acceptance probe resolves to three chronological events with the adjacent temporal chain under the configured model before the fix is considered validated for the reported failure.
11. Existing correction/coreference and server-owned ID/edge behavior remain unchanged.
12. No changes are made to retrieval, DP, EventTrail, or scoped/global progressive KIS operations as part of this fix.

## 15. Rationale

This is intentionally a small semantic-contract correction rather than a new planning subsystem.

The downstream architecture already supports multi-event temporal retrieval once the resolver returns a proper event list. Strengthening both the prompt and the structured-output schema addresses the observed failure at its actual boundary while preserving the existing server-owned graph invariants and avoiding a brittle deterministic fallback.
