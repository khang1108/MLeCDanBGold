# KIS Initial Resolver Simplification Design

**Status:** Approved from the 2026-09-17 v1.5 debugging discussion.

## Problem

`KISIntentResolver.resolve_initial()` currently asks one LLM call to summarize the query, extract entities, resolve coreference and contradictions, decompose temporal events, translate them to English, and bind entities to events. The downstream retrieval plan primarily consumes `event.text`, while the extra semantic work increases output size and gives the model room to degenerate into long repetitive prose. The initial resolver also inherits a generic token budget because it does not pass `max_tokens` explicitly.

## Decision

The initial natural-language resolver will do exactly one semantic task: convert the input narrative into a short, chronologically ordered list of visually retrievable event descriptions.

The model output contract becomes event-only:

```python
KISInitialResolution:
    events: list[KISInitialResolutionEvent]

KISInitialResolutionEvent:
    text: concise English event description
```

The server, not the LLM, owns canonical IDs, query text, empty initial entity state, bindings, revision, and adjacent temporal edges.

## Event semantics

An event is a distinct retrievable visual moment, not a shot boundary. Two moments must remain separate even when a continuous camera pan connects them. Actions/details that are genuinely simultaneous in the same visual moment stay together.

Each event text must be self-contained English, at most 240 characters, and should normally fit in one short sentence. The model must preserve uncertainty instead of inventing specifics (for example, "an unusual pen" rather than guessing the tool type).

## Generation budget

`KISIntentResolver` will call the generic `LLMClient.generate_structured()` with an explicit `max_tokens=512` and `temperature=0.0`. This task-level budget must not depend on `HCMAI_LLM_MAX_TOKENS`.

## Canonicalization

For model events `[e1, e2, ..., en]`, the server constructs:

- `E1..En` in returned order;
- `bindings=[]` for every initial event;
- `entities=[]` for the initial intent;
- `query_text` by joining canonical event texts in order;
- adjacent `before` edges `E1->E2`, ..., `E(n-1)->En`;
- the caller-owned revision unchanged.

Explicit `E#:` initial input, scoped patches, and `/llm-rewrite` remain separate flows. `/llm-rewrite` may still perform richer entity/global semantic reasoning later.

## Non-goals

This change does not add regex sentence splitting, deterministic semantic fallback, a second LLM pass, entity extraction in initial resolution, hard grammar-constrained decoding, changes to retrieval/DP/EventTrail, or a new model provider.

## Acceptance cases

1. A single simultaneous scene resolves to one short event.
2. A narrative with three distinct temporal moments resolves to three short events.
3. A continuous camera pan from one retrievable subject to a later retrievable action resolves to separate events despite being one shot.
4. No accepted initial event can exceed 240 characters.
5. The initial resolver passes `max_tokens=512` explicitly.
6. The canonical intent contains empty entities/bindings and an exact adjacent temporal chain.
7. Existing explicit `E#:`/scoped/global rewrite flows remain unchanged.
