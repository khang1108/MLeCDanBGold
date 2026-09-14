# KIS Semantic Intent + Provider-Agnostic Inference — Design

## Scope

This design supersedes the deterministic KIS intent-builder portion of the previous Unified KIS plan. The goal is to finish the Unified KIS foundation before EventTrail by replacing punctuation-based event splitting with SLM-resolved structured intent, while making LLM/text-embedding providers switchable through environment configuration.

The production KIS path becomes:

`ordered KIS clues -> KISIntentResolver -> KISIntent semantic graph -> retrieval query preparation -> temporal evidence scoring -> DP alignment -> KIS results`

EventTrail actions (`anchor`, `reject`, `replace`, `verify`) remain out of scope, but the intent schema must already expose stable event/entity identities so EventTrail can consume it later without another schema rewrite.

## Global architectural constraints

1. **One source of truth per concept.** A new abstraction must replace the old abstraction, migrate all production callers, and delete unreachable legacy code in the same implementation plan.
2. **No deterministic KIS fallback.** `KISIntentResolver` is the only production intent resolver. Provider/schema failures surface explicitly.
3. **Provider details stay below domain services.** KIS code depends on `LLMClient`, never Qwen, OpenAI, Bedrock, or a concrete base URL.
4. **Prompts are version-controlled behavior.** `.env` selects provider/model/endpoint; system prompts stay in source code.
5. **DP does not understand language.** The resolver owns coreference, correction, merge/split, and temporal ordering. DP only aligns the resulting ordered event sequence.
6. **No speculative server-side KIS session store.** The frontend still sends the full ordered clue list for each revision.

## 1. Provider-agnostic inference clients

Introduce capability-level inference clients under `hcmai.inference`:

- `LLMClient`: structured text generation.
- `EmbeddingClient`: text embedding.
- shared endpoint/HTTP configuration.

The domain service receives a client instance. It does not know whether inference is provided by `api.iamphuckhang.dev` or a third-party API.

Environment configuration:

```env
HCMAI_LLM_BASE_URL=https://api.iamphuckhang.dev/v1
HCMAI_LLM_API_KEY=
HCMAI_LLM_MODEL=Qwen/Qwen3-4B
HCMAI_LLM_TIMEOUT_SECONDS=30

HCMAI_EMBEDDING_BASE_URL=https://api.iamphuckhang.dev/v1
HCMAI_EMBEDDING_API_KEY=
HCMAI_EMBEDDING_MODEL=google/siglip2-base-patch16-224
HCMAI_EMBEDDING_TIMEOUT_SECONDS=30
```

`LLMClient` is the default implementation. Provider-specific adapters may be added only when the provider cannot satisfy the capability contract.

The system prompt is *not* stored in these client configs. `KISIntentResolver` owns its prompt/profile; Query Preparation owns a different prompt/profile.

### Legacy impact

- Remove the KIS/query-preparation dependency on `QwenQueryPreparationAdapter`.
- Remove duplicated LLM HTTP/auth handling from domain-specific adapters after all callers move to `LLMClient`.
- `llm.remote.InferenceClient` may remain only for capabilities not replaced by the new client layer (OCR/caption/ASR/image-only private endpoints). Overlapping text-generation/text-embedding methods must be removed once migrated.
- `RemoteEmbeddingAdapter` continues to own index/model validation, but imports the shared `EmbeddingClient` contract instead of declaring another client protocol.

## 2. Full semantic KIS intent graph (Decision C)

Move the semantic intent out of the HTTP-contract module into a dedicated KIS domain package.

### Entity node

```python
class KISEntity(BaseModel):
    id: EntityId
    kind: Literal["person", "object", "place", "text", "other"]
    description: NonBlank
```

Examples:

```text
X1: person — woman wearing a white apron
X2: person — man talking with X1
X3: object — white plate
X4: object — steak
```

### Event-to-entity binding

```python
class KISEntityBinding(BaseModel):
    entity_id: EntityId
    role: NonBlank
```

Stable entity IDs encode continuity. If `X1` appears in E1, E2 and E3, downstream code can explicitly know that the resolver believes it is the same participant; the DP does not yet enforce that identity visually.

### Event node

```python
class KISEvent(BaseModel):
    id: EventId
    text: NonBlank
    bindings: list[KISEntityBinding]
```

`text` must be self-contained for retrieval. It must not depend on unresolved pronouns such as `she`, `they`, `it`, or `that plate` when a canonical entity description can be stated instead.

### Temporal edge

Current DP implements a strict chronological chain. Therefore the initial semantic graph intentionally supports only explicit `BEFORE` edges:

```python
class KISTemporalEdge(BaseModel):
    source: EventId
    relation: Literal["before"] = "before"
    target: EventId
```

Simultaneous/overlapping facts are merged into one event by the resolver instead of emitting a relation the decoder cannot honor. This keeps the schema truthful while still creating the entity/event graph needed by later EventTrail work.

### Intent

```python
class KISIntent(BaseModel):
    revision: int
    inputs: list[NonBlank]
    language: Literal["vi", "en"]
    query_text: NonBlank
    entities: list[KISEntity]
    events: list[KISEvent]
    temporal_edges: list[KISTemporalEdge]
```

Validation requires:

- unique entity IDs and event IDs;
- event IDs ordered `E1..En`;
- all binding references resolve;
- all temporal-edge references resolve;
- no self edge or cycle;
- every explicit `BEFORE` edge agrees with the canonical `events` list order;
- event count is within `DEFAULT_MAX_TEMPORAL_EVENT_COUNT`.

The `events` list is the canonical topological order that current DP consumes. The edge list is explicit semantic evidence for UI/debug/EventTrail rather than a second competing order.

## 3. KISIntentResolver

`KISIntentResolver` is a domain service under `hcmai.kis`.

Input: ordered raw clue strings.

Output: validated `KISIntent`.

It calls `LLMClient.generate_structured(...)` once with the full clue history. There is no deterministic fallback and no second punctuation-based planner.

Prompt invariants:

1. Preserve every supported fact from the clue history unless a later clue explicitly corrects it.
2. Resolve pronouns/coreference into stable entities.
3. Corrections replace contradicted attributes in `query_text`, entities, and event text; raw `inputs` remain immutable history.
4. Merge descriptions that belong to one simultaneous state/event.
5. Split genuinely sequential actions into separate events.
6. Reorder events into actual temporal order even when clues are revealed out of order (`before that`, `earlier`, `afterwards`).
7. Emit self-contained retrieval event text.
8. Keep canonical text in the dominant input language (`vi` or `en`). Query preparation handles retrieval-language transformation after resolution.
9. For the current decoder, normalize temporal semantics into a strict event chain. Do not emit unsupported overlap/branch relations.

Example:

```text
Q1: A man enters a room.
Q2: Before that, he talks to a woman.
```

must resolve to roughly:

```text
X1 = man
X2 = woman
E1 = X1 talks to X2
E2 = X1 enters the room
E1 BEFORE E2
```

## 4. Query preparation occurs after intent resolution

Intent resolution and retrieval rewriting remain separate concerns.

`KISIntentResolver` answers: **what does the evolving clue history mean?**

`QueryPreparationService` answers: **how should each already-resolved event be expressed for retrieval?**

The order is:

`raw clues -> semantic graph -> aligned event texts -> translation/paraphrase -> retrieval`

Never run query preparation before resolver coreference/temporal resolution.

For the default KIS search:

- BM25 caption/OCR/ASR queries use the resolved canonical event text so the corpus language is preserved.
- Dense retrieval uses a literal English bundle from Query Preparation when configured.
- Query-candidate generation may still produce five aligned alternative bundles, but it must receive explicit resolved event texts; it no longer accepts one raw query and splits it itself.

Query Preparation must preserve event count/order/identity. It may rewrite text but must not modify the semantic graph.

## 5. DP projection

Current DP stays numerical and unchanged.

A tiny projection extracts:

```python
original_events = tuple(event.text for event in intent.events)
```

and passes those strings plus aligned retrieval rewrites into `TemporalSearchService`.

Stable event/entity IDs remain attached to `KISIntent`; current `AlignedPath.frame_ids[i]` corresponds to `intent.events[i]`.

This creates the future EventTrail mapping without teaching the DP about NLP.

## 6. Unified revisioned KIS API

The canonical text KIS endpoint is:

`POST /api/v1/kis/search`

Request remains client-owned revision history:

```json
{
  "inputs": [
    {"text": "A woman is in a kitchen."},
    {"text": "She is talking to a man."}
  ],
  "expected_revision": 1,
  "use_dense": true,
  "use_bm25": true,
  "top_k": 20
}
```

Response has no duplicated query/event source of truth:

```json
{
  "intent": { ...full KISIntent graph... },
  "dense_events": ["..."],
  "bm25_events": ["..."],
  "results": [...],
  "latency": {...}
}
```

`intent.revision`, `intent.inputs`, `intent.query_text`, and `intent.events` are canonical; do not duplicate them at the response top level.

### Legacy endpoint rule

After the frontend migrates, the old text `POST /api/v1/search` KIS route is removed. Image search/filter routes in the same router file remain because they are different capabilities. The old `SearchRequest` and old raw-query `KISPipeline.execute()` are deleted after no callers remain.

## 7. Unified KIS Panel

The frontend owns:

```text
draft
committedInputs[]
revision
currentIntent
```

Submit behavior:

- revision 0 + first input -> normal KIS-T-like one-shot search;
- later submissions append a clue and keep the same logical KIS session;
- backend returns the rebuilt semantic graph each revision;
- failed requests do not commit the new draft;
- New Search aborts the request and resets revision/input/intent state.

The KIS panel shows:

- ordered clue timeline;
- canonical current query;
- compact resolved events (`E1`, `E2`, ...);
- optional entity chips (`X1`, `X2`, ...) for inspection/debug.

It is not a chatbot. EventTrail manipulation remains the next phase.

## 8. History

History snapshots store the full resolved `KISIntent` graph with result IDs/timestamps/scores. History does not become the live session source of truth.

No new database tables are required in this phase if the existing history JSON snapshot supports arbitrary JSON; intent revisions can be persisted inside the snapshot. A future stateful EventTrail session may justify normalized revision/action tables.

## 9. Legacy removal map

The implementation is incomplete until this map is satisfied.

| New owner | Replaces | Final action |
|---|---|---|
| `hcmai.kis.KISIntentResolver` | `KISIntentBuilder` | delete `orchestration/workflows/kis_intent.py` |
| semantic `KISIntent` models | `KISIntent` inside API contracts | move domain model; API imports it |
| resolver event structure | `plan_query_events` in KIS path | remove KIS calls; delete planner splitting once last raw-query caller migrates |
| `temporal.events.normalize_event_texts` | normalization living beside planner | move normalizer; delete retired planner module |
| `LLMClient` | `QwenQueryPreparationAdapter` and domain-specific LLM transport | migrate Query Preparation; delete adapter |
| shared `EmbeddingClient` protocol | duplicate remote embedding client protocol | import shared contract; delete duplicate protocol declarations |
| `/api/v1/kis/search` | text KIS `/api/v1/search` | migrate frontend then remove old text route/request |
| `KisSession` frontend state | `eventDescription` as entire KIS domain state | migrate KIS state; keep unrelated image/filter state |

`git grep`/`rg` must verify zero production references before each legacy deletion.

## 10. Acceptance criteria

1. Two or more KIS clues resolve into one validated semantic graph with explicit entities, events, bindings, and temporal edges.
2. Coreference and temporal reordering are performed by the LLM resolver before retrieval.
3. No production KIS path invokes `plan_query_events` or the deterministic builder.
4. DP receives only resolved ordered event strings and returns one aligned frame per event.
5. Query Preparation consumes resolved events and cannot alter semantic event cardinality/order.
6. LLM endpoint/model/API key can be changed from `.env` without modifying KIS/query-preparation code.
7. Text embedding endpoint/model/API key can be changed from `.env` without modifying retrieval domain code.
8. The frontend can perform revision 1, append clue revision 2+, and display the returned semantic intent.
9. Failed revisions preserve previously committed inputs/intent.
10. Old text `/api/v1/search`, `KISIntentBuilder`, KIS `plan_query_events` usage, and `QwenQueryPreparationAdapter` are removed after migration.
11. Existing image search, filter, TRAKE, temporal DP, and result inspection remain functional.
