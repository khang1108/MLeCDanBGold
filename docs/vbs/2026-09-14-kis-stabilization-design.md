# KIS Stabilization Design

**Date:** 2026-09-14  
**Status:** Approved through Design Decisions A-H  
**Scope:** Stabilize the unified revisioned KIS flow before EventTrail / interactive multi-event evidence search.

## 1. Goal

Build one reliable KIS foundation in which progressive clues are resolved into one semantic intent, retrieval is derived from that intent without mutating it, temporal alignment is delegated to the existing DP path, and frontend revisions are transactional. The stabilization batch must also remove superseded query-planning, query-expansion, and query-preparation abstractions rather than retaining parallel legacy paths.

The target online flow is:

```text
KIS clues[]
    -> KISIntentResolver
    -> KISResolution
    -> server canonicalization
    -> KISIntent
    -> KISRetrievalPlan
        -> canonical/BM25 text
        -> EventTranslator -> dense English text when required
    -> TemporalSearchService
    -> DP alignment
    -> aligned KIS results
    -> transactional KIS panel / result workspace
```

This design deliberately does **not** add EventTrail, event anchoring/rejection, VLM temporal verification, arbitrary temporal graphs, or entity-consistency scoring. Those depend on a stable KIS foundation and belong to the next design cycle.

## 2. Architectural Principles

### 2.1 One semantic source of truth

`KISIntent` is the canonical application-level meaning of the current KIS session revision. Once created, retrieval preparation must not mutate its entities, event cardinality, event order, bindings, IDs, or temporal meaning.

### 2.2 Server-owned metadata, model-owned semantics

The language model reasons only about semantic content. The server owns deterministic metadata such as revision, raw clue history, entity IDs, event IDs, and the adjacent `before` chain required by the current DP decoder.

### 2.3 No hidden fallback path

`KISIntentResolver` is the only production intent-construction path. There is no deterministic production fallback. Resolver/provider failures surface explicitly.

### 2.4 Replace, migrate, delete

Every new abstraction must identify what it replaces. All production callers are migrated in the same stabilization batch, then obsolete code is deleted. Compatibility shims are forbidden unless an explicit external consumer is identified.

### 2.5 Retrieval implementation details are not product semantics

Dense translations, BM25 texts, provider model names, and cache keys are internal retrieval concerns. Product state centers on the committed `KISIntent`, results, and user clue timeline.

---

## 3. Decision A — Transactional KIS Frontend State

The current frontend mutates committed result state before a new revision succeeds. This causes failed clue additions to erase still-useful results and creates inconsistent inspector/query state.

### 3.1 State invariant

Submitting a new clue creates a **pending revision** while the previous committed revision remains fully usable:

```text
Committed rev N
  intent N
  results N
  exploration snapshot N

+ draft clue

-> pending rev N+1
   previous committed state remains visible

success -> atomically replace committed state with rev N+1
failure -> discard pending state, keep rev N, preserve draft + error
```

A pending revision must never clear committed frames, event list, latency, result type, committed exploration snapshot, or committed canonical query.

### 3.2 `activeQuery` semantics

The global query context used by inspectors/modals means **committed canonical intent**, not textarea draft text.

- Before first successful search: draft may be used as temporary context.
- After a successful revision: source is `kisSession.currentIntent.query_text`.
- Typing a later clue must not change inspector query context until that revision succeeds.
- There must be exactly one propagation path for committed query changes.

### 3.3 Frame selection

`openCanonicalFrame()` must invoke `onFrameClick` exactly once with a selection object built from the committed revision:

```text
{
  frame,
  explorationSnapshot: committedExplorationSnapshot
}
```

No second callback may overwrite the selection without the snapshot.

### 3.4 History persistence is off the critical path

Search transaction completion is defined by retrieval success and UI commit, not by SQLite/history persistence.

```text
search response
   -> commit intent/results
   -> release search lock
   -> persist history asynchronously/best-effort
```

History failure displays a warning but does not roll back the live session or block the next clue.

### 3.5 Replay is a separate mode

Historical replay is read-only and must not inject the saved canonical query into the live draft field.

```text
LIVE   = clue evolution + draft + search
REPLAY = stored intent/results/activity, no editable draft
```

`New Search` returns to a fresh live session. Branching from replay is not part of this batch.

---

## 4. Decision B — `KISResolution` vs `KISIntent`

The current resolver asks the LLM to regenerate `revision`, raw `inputs`, canonical IDs, and temporal edges even though the server already owns those values. This creates unnecessary failure modes and duplicates temporal truth.

### 4.1 LLM output schema

Introduce a model-only semantic output, conceptually:

```text
KISResolution
  language
  query_text
  entities[]
    kind
    description
  events[]
    text
    entity_indices[]
```

The model does **not** emit:

- revision,
- raw clue history,
- `X1` / `E1` identifiers,
- temporal edges,
- retrieval translations,
- timestamps or candidate decisions.

Entity references use zero-based indices into the emitted `entities[]` list so the server can assign canonical IDs deterministically.

### 4.2 Server canonicalization

`KISIntentResolver`:

1. normalizes and validates raw clues;
2. requests `KISResolution` from `LLMClient`;
3. validates entity indices and event cardinality;
4. assigns `X1..Xm` entity IDs;
5. assigns `E1..En` event IDs;
6. converts entity indices to bindings;
7. derives adjacent temporal edges `E1 before E2`, `E2 before E3`, ...;
8. injects exact normalized `inputs` and `revision=len(inputs)`;
9. returns validated `KISIntent`.

### 4.3 Current temporal expressiveness

The existing DP supports an ordered chain, not an arbitrary temporal graph. Therefore the model must emit events in canonical chronological order.

Simultaneous/overlapping actions that cannot be represented by the chain are expressed as one event for this phase rather than inventing unsupported relation types.

For multi-event intent, the canonical graph must contain the complete adjacent chain. Duplicate or non-adjacent redundant temporal edges are not model-generated and do not need to be stored.

### 4.4 Error semantics

- malformed client request / blank clues -> HTTP 422;
- provider unavailable -> HTTP 503;
- valid provider response that fails semantic/schema contract -> HTTP 502 through a domain-specific `KISResolutionError`;
- no conversion of model/schema failures into user-input 422 errors.

---

## 5. Decision C — Immutable Intent and `KISRetrievalPlan`

The current orchestration passes parallel `list[str]` collections (`dense_events`, `bm25_events`, `retrieval_events`) whose alignment depends on position. This becomes fragile as the semantic graph grows.

### 5.1 Retrieval projection

Introduce an internal `KISRetrievalPlan` keyed/aligned by stable event ID. Each retrieval event contains the text views required by active retrieval sources, conceptually:

```text
RetrievalEvent
  event_id
  canonical_text
  dense_text | null
  bm25_text | null

KISRetrievalPlan
  events[]
```

The builder validates that retrieval event IDs exactly match the canonical `KISIntent.events` IDs and order.

### 5.2 Query preparation invariant

Retrieval preparation may rewrite event text **one-to-one** for a modality, but may never:

- merge/split events,
- reorder events,
- change event IDs,
- change entity bindings,
- change entity IDs,
- change graph semantics.

### 5.3 Source behavior

- BM25 uses canonical event text.
- Dense uses canonical event text when language/model compatibility allows it.
- For Vietnamese-to-English dense retrieval, `EventTranslator` produces an aligned English translation for each event.
- DP and materialization operate on the same canonical event order regardless of retrieval modality.

### 5.4 Public API

Dense/BM25 prepared event strings are retrieval implementation details. They should not remain in `KISRevisionSearchResponse` unless a confirmed frontend/debug consumer requires them. The product response centers on canonical intent, results, enabled sources, warnings, and latency.

---

## 6. Decision D — Remove Query Expansion

Query expansion/candidate generation is not part of the production KIS path, has no frontend consumer in the reviewed snapshot, adds an additional LLM call, and lacks an explicit fusion policy. It is therefore removed rather than retained as speculative code.

### 6.1 Delete the backend chain

Remove the backend-only chain:

```text
/api/v1/query-candidates
  -> SearchService.generate_query_candidates()
  -> QueryPreparationService.generate_candidates()
  -> candidate bundle/query candidate models
```

Delete associated contracts, router registration, configuration, prompts, cache operation, exports, and dead tests/docs.

### 6.2 Reduce query preparation to translation

After candidate generation is deleted, `QueryPreparationService` is renamed/reduced to `EventTranslator` with one responsibility:

> Translate an ordered event list one-to-one into retriever-compatible English without changing event count, order, entities, actions, attributes, numbers, required tokens, or meaning.

The package `hcmai.query_preparation` must not survive solely as a legacy name. Move the remaining translation code under a retrieval-owned package, target path:

```text
src/hcmai/retrieval/translation/
```

### 6.3 No automatic reintroduction

If expansion is revisited later, it is a separate retrieval experiment that must define candidate fusion and demonstrate value via benchmark/ablation before entering production architecture.

---

## 7. Decision E — Provider-Agnostic Inference Cleanup

Deployment/model identity belongs exclusively to capability clients. Domain services own prompts and domain invariants only.

### 7.1 Capability clients

Keep provider-agnostic clients:

```text
LLMClient
EmbeddingClient
```

Each is configured from `.env` / endpoint config with:

```text
base_url
api_key
model
timeout
```

Self-hosted and third-party providers use the same application-facing interface.

### 7.2 Domain configuration

`EventTranslator` configuration contains only translation behavior such as cache settings and prompt version. It does not store a hard-coded model name/revision or query-expansion candidate count.

Cache identity uses the **actual model configured on the capability client**, plus operation/prompt/input identity. Changing `.env` model/provider must never reuse a cache entry from a different model identity.

### 7.3 Embedding configuration

Remote vs local embedding is explicit. If remote embedding is configured and its endpoint/auth/model fails, the application reports failure; it must not silently execute a different local model.

Remove `Any`-typed "maybe client" seams. Retrieval setup receives an explicit `EmbeddingClient | None` or an explicit local encoder selected by configuration.

### 7.4 Error taxonomy

Use a small inference error taxonomy sufficient to distinguish:

- provider unavailable/timeout,
- auth/configuration failure,
- malformed/invalid provider response.

Domain services wrap semantic contract failures into domain errors (`KISResolutionError`, `EventTranslationError`) rather than exposing provider/Pydantic internals directly to HTTP routing.

### 7.5 Latency ownership

Orchestration records the user-visible KIS stages:

```text
intent_ms
translation_ms
retrieval_ms
alignment_ms
materialization_ms
total_ms
```

`total_ms` covers the complete online request path represented by those stages. If compatibility requires retaining `query_ms`, it is derived as `intent_ms + translation_ms` rather than omitting translation latency.

---

## 8. Decision F — KIS Panel as Intent-Evolution Surface

The KIS panel answers "what am I searching for?"; the results/EventTrail workspace answers "where is the evidence?". The panel is not a chatbot and not a full semantic-graph inspector.

### 8.1 Always visible

The panel shows:

1. ordered clue timeline (`Q1`, `Q2`, ...);
2. committed canonical intent (`Current intent`);
3. ordered event sequence (`E1`, `E2`, ...).

### 8.2 Collapsed/debug detail

Entities and bindings are available under a collapsed `Semantic details` section or debug mode. Raw adjacent temporal edges are not shown in normal product UI because event ordering already communicates the current chain.

### 8.3 Pending revision UX

While a new clue is resolving:

- previous committed results remain visible and inspectable;
- previous committed intent remains the active query context;
- the draft stays visible;
- one revision is allowed in flight;
- the input may remain disabled while pending, but result browsing must remain enabled.

On failure, the draft and committed state remain intact with an explicit error.

### 8.4 Copy

- revision 0 submit button: `Search`;
- revision >= 1 submit button: `Add clue`;
- placeholder: `Search or add another clue...`.

No moderator/user/assistant message schema is added until the exact VBS KIS-C protocol requires it.

---

## 9. Decision G — Generic LLM Endpoint on the Existing Inference Server

The deployed model API currently exposes task-specific query-preparation operations, while the new `LLMClient` expects a generic OpenAI-compatible generation endpoint. The long-term provider-switching goal requires the generic contract, not a new HCMAI-specific client adapter.

### 9.1 Add generic text generation

The existing inference service must be extended to expose:

```text
POST /v1/chat/completions
```

with the subset of OpenAI-compatible fields required by `LLMClient`:

- `model`,
- `messages`,
- `temperature`,
- `max_tokens`,
- structured JSON / JSON-schema response format.

The inference server executes the model only. It does not own KIS, translation, EventTrail, or other domain prompts.

### 9.2 Domain prompts remain in HCMAI backend

- `KISIntentResolver` owns KIS semantic-resolution prompts/schema.
- `EventTranslator` owns translation prompt/schema.
- future domain services own their own prompts.

The same `LLMClient` can point through `.env` to the self-hosted API or a compatible third-party provider without domain-code changes.

### 9.3 Remove task-specific query-preparation API

After caller migration, delete/supersede task-specific inference endpoints for:

```text
/query-preparation/translate
/query-preparation/candidates
```

and the associated query-preparation inference router/contracts if no independent consumer remains.

The reviewed source archive does not contain the source router shown by the deployed Swagger instance; implementation must reconcile the deployed/runtime source before deletion rather than assuming the archive is authoritative for that endpoint.

### 9.4 Base URL invariant

`HCMAI_LLM_BASE_URL` is the API root including `/v1`, e.g.:

```text
https://api.iamphuckhang.dev/v1
```

The client appends `/chat/completions`.

### 9.5 Readiness

KIS readiness requires all mandatory KIS capabilities, including configured intent resolution. Health must not report KIS ready when every text KIS request would fail for missing LLM configuration.

The backend readiness projection reports at least:

```text
kis
intent_resolution
event_translation
retrieval
```

A readiness request does not need to run a fresh LLM completion every time; configuration/client/model-server readiness is sufficient, while runtime provider failures still surface on the request path.

---

## 10. Decision H — Stabilization Completion Gate

Stabilization is complete only when the new KIS flow is verified across inference, semantic resolution, orchestration, frontend transaction behavior, readiness, history persistence, and legacy deletion.

### 10.1 Inference contract tests

Use a fake OpenAI-compatible HTTP server/transport boundary so tests verify the actual URL/payload/response contract instead of mocking away `LLMClient`.

Cover:

- `/v1/chat/completions` structured success,
- malformed structured content,
- timeout/unavailability,
- auth error,
- provider 5xx,
- text embedding request/response ordering.

### 10.2 Resolver tests

Use a fake `LLMClient` semantic response and verify canonicalization for:

- pronoun/coreference resolution,
- chronological reordering (`before that`),
- same-event merge,
- sequential-action split,
- clue correction/replacement,
- entity-index validation,
- deterministic `Xn`/`En` ID assignment,
- complete adjacent `before` chain,
- exact raw clue/revision ownership by the server.

### 10.3 Backend KIS orchestration test

Exercise:

```text
POST /api/v1/kis/search
 -> resolver
 -> retrieval plan
 -> optional translation
 -> temporal search
 -> DP
 -> materialization
 -> response
```

Scorers/corpus may use deterministic test doubles, but orchestration boundaries must remain real. Verify event IDs/order and latency stage consistency.

### 10.4 Frontend transaction tests

Required regression cases:

- old results remain visible while next revision is pending;
- success atomically commits new clue/intent/results;
- failure keeps previous revision/results/intent and preserves draft;
- frame click callback fires exactly once with committed exploration snapshot;
- typing draft does not change committed inspector query;
- replay is read-only and does not become live draft state;
- history-save failure does not fail/block live KIS search.

### 10.5 Manual acceptance scenario

Run one progressive KIS session:

```text
Q1: A woman is standing in a kitchen.
Q2: She is talking to a man.
Q3: Before taking a white plate, they move to the left.
```

Verify:

- revision progression 1 -> 2 -> 3;
- Q2 resolves `She` consistently;
- Q3 canonical event order is `talk -> move left -> take plate`;
- DP receives that same order;
- result inspection receives one selection callback;
- forcing LLM failure on a later clue leaves revision 3/result state usable and preserves the failed draft.

---

## 11. Legacy Removal Map

The implementation plan must trace every caller before deletion and end with zero production references to superseded symbols/routes.

### 11.1 KIS semantic/planning legacy

Delete if still present/unused after migration:

```text
KISIntentBuilder
old kis_intent.py deterministic builder
plan_query_events
split_query_events
legacy raw-query text KIS path
```

### 11.2 Query expansion/preparation legacy

Delete/migrate:

```text
hcmai.query_preparation package
QueryPreparationService
generate_candidates
CandidateBundle
QueryCandidate
QueryCandidateSet
/api/v1/query-candidates
SearchService.generate_query_candidates
candidate_count / candidate prompt config
query-preparation router/contracts on inference server
```

The one-to-one translation behavior moves to `hcmai.retrieval.translation.EventTranslator`.

### 11.3 Provider/config legacy

Delete/migrate:

- query-preparation hard-coded model name/revision as cache identity;
- provider-specific query-preparation adapters;
- `Any`-typed embedding-client detection;
- silent remote-to-local embedding fallback;
- stale exports/imports tied to removed packages.

### 11.4 API response legacy

Remove `dense_events` and `bm25_events` from product KIS response if repository-wide tracing confirms no required consumer after `KISRetrievalPlan` migration. Exploration must consume a canonical revision snapshot, not reconstruct its source data from public retrieval-debug fields.

### 11.5 Generated/stale artifacts

Remove generated artifacts from source snapshots/tracking and establish ignore rules for at least:

```text
__pycache__/
*.py[cod]
.pytest_cache/
frontend/build/
frontend/node_modules/
.env
```

Update stale docstrings/docs that still claim ownership of deleted query splitting, query expansion, or query-preparation behavior.

---

## 12. File/Module Target State

The exact implementation plan may adjust individual filenames after reference tracing, but the target ownership is:

```text
src/hcmai/kis/
  models.py              # KISResolution + canonical KISIntent graph types
  prompts.py             # semantic-resolution prompt
  resolver.py            # KISResolution -> canonical KISIntent

src/hcmai/inference/
  config.py
  http.py
  llm.py                 # generic OpenAI-compatible generation client
  embeddings.py          # provider-agnostic text embedding client

src/hcmai/retrieval/translation/
  __init__.py
  models.py              # only if a translation response schema is needed
  prompts.py
  service.py             # EventTranslator

src/hcmai/orchestration/
  pipeline.py            # revision orchestration + latency ownership
  setup.py               # explicit capability wiring
  health.py              # end-to-end capability readiness

src/hcmai/orchestration/workflows/
  kis.py                 # consumes KISIntent + KISRetrievalPlan
  temporal_search.py     # unchanged semantic responsibility

src/hcmai/api/contracts/kis.py
src/hcmai/api/routers/kis.py

llm/server/routers/
  generation.py          # /v1/chat/completions
  ... existing capability routers

frontend/src/features/kis/
  session.js
  components/KisPanel.jsx

frontend/src/features/search/components/SearchWorkspace.jsx
frontend/src/App.jsx
```

No EventTrail-specific module is introduced in this stabilization batch.

---

## 13. Non-Goals

This batch does not implement:

- interactive event anchor/reject/replace actions;
- EventTrail rendering;
- graph-aware temporal decoding beyond an ordered chain;
- VLM temporal verification;
- entity re-identification or identity scoring;
- arbitrary `overlap`, `during`, causal, or dependency relations;
- query expansion/fusion;
- official moderator/user message metadata for KIS-C;
- history branching/continue-from-replay.

These remain future work after the stabilization completion gate passes.

---

## 14. Success Criteria

The stabilization design is satisfied when all of the following are true:

1. Clicking KIS `Search` successfully reaches a compatible generic LLM generation API and produces a canonical `KISIntent`.
2. Semantic intent is resolved once; retrieval preparation cannot alter graph semantics.
3. Dense/BM25 text is represented through one aligned `KISRetrievalPlan`, not parallel positional lists.
4. Query expansion and obsolete query-preparation infrastructure are removed.
5. Provider/model switching is controlled by `.env` capability-client configuration.
6. Missing/failed required inference capabilities are visible through readiness and correct HTTP error classes.
7. KIS clue revisions are transactional in the frontend.
8. History persistence cannot block or roll back a successful search.
9. Replay is explicitly separate from live KIS state.
10. Required backend/frontend/inference regression tests pass.
11. Repository-wide legacy/dead-code scans show no production references to superseded abstractions.
12. Generated build/cache artifacts and stale documentation are cleaned before EventTrail work begins.
