# Unified Multimodal Progressive KIS Design

**Date:** 2026-09-15  
**Status:** Approved through S1.5  
**Scope:** Finish the incomplete KIS stabilization work, then replace the current clue-list / separate-image-search model with one revisioned multimodal event composer that becomes the substrate for EventTrail.

## 1. Goal

The next implementation cycle has two ordered stages:

1. **S0 — Finish stabilization:** land the remaining pieces of the previously approved KIS stabilization design that are still absent or incorrect in the reviewed source snapshot.
2. **S1 — Unified Multimodal Progressive KIS:** make one KIS flow support initial natural-language search, explicit multi-event authoring, scoped progressive event refinement, image-only events, text+image events, global semantic rewrite, and retrieval-only re-search without inventing a second search architecture.

EventTrail is **not implemented in this S0/S1 batch**. S1 must deliberately produce the event representation, retrieval snapshots, and UI state that EventTrail can consume in the next design cycle.

The target flow is:

```text
Query Composer
    -> InitialResolve | PatchEvents | GlobalRewrite | SearchOnly
    -> canonical KISIntent revision
    -> KISRetrievalPlan
        -> textual evidence
        -> image evidence
    -> existing temporal fusion
    -> DP temporal alignment
    -> ranked results
```

## 2. Current Source Ground Truth

The reviewed archive already contains several parts of the previous stabilization work:

- `KISResolution` is separated from `KISIntent`.
- `KISIntentResolver` canonicalizes entity IDs, event IDs, and adjacent temporal edges server-side.
- a provider-agnostic `LLMClient` exists.
- a generic chat-completions inference boundary has been introduced.

However, the reviewed snapshot still contains or exhibits important incomplete stabilization behavior that must be finished before S1 depends on it:

- `KISIntent` still stores `inputs` and derives `revision` from clue count.
- query-preparation / query-candidate infrastructure still exists in production references.
- public KIS responses still expose retrieval-preparation details such as dense/BM25 event strings.
- `KISRetrievalPlan` and the final `EventTranslator` ownership are not fully landed.
- image search remains a separate request path rather than an event modality.
- frontend KIS revision state still has transactional bugs identified in the previous review, including committed-state clearing before success and duplicate frame-selection propagation.
- replay/history semantics still overlap with live draft semantics.

S0 must reconcile these against the previous approved stabilization specification before S1 changes the API contracts further.

## 3. Design Principles

### 3.1 One event model for all KIS input modes

Textual KIS, visual KIS, multi-event KIS, and progressive KIS-C must converge on one canonical event sequence. Separate product modes may exist for convenience, but there must not be separate semantic or temporal retrieval pipelines.

### 3.2 Explicit scope beats implicit mutation

Once an intent exists, the system does not guess which event a free-form follow-up should edit. Local progressive edits name their target using `E#:` syntax. Global semantic reconsideration is opt-in through `/llm-rewrite`.

### 3.3 Local model reasoning, server-owned structure

The SLM/LLM resolves semantics within the scope granted by the operation. Event identity, revision, image attachment ownership, event ordering, and operation legality are deterministic server responsibilities.

### 3.4 Images are canonical user evidence, not generated text

Image attachments are stored and referenced as image assets. The system does not automatically turn image captions into canonical event text or BM25 evidence. Derived captions may be explored later, but they are not semantic truth in this design.

### 3.5 DP remains modality-agnostic

Each event ultimately produces one temporal score row over frames. The DP decoder must not need to know whether that evidence came from text, image, or both.

### 3.6 No speculative fusion contribution in S1

S1 does not introduce a new embedding-fusion research method. Images add visual evidence into the existing calibrated/reliability-aware fusion path. The research contribution cycle remains focused on EventTrail.

## 4. S0 — Stabilization Completion Before S1

S0 is not a second refactor project. It is the completion gate for the already-approved stabilization work.

The implementation must first trace the current source and finish at least these unresolved items before introducing the new S1 contracts:

1. Remove obsolete query-expansion/query-candidate production paths after caller tracing.
2. Complete the move from generic query-preparation ownership to retrieval-owned one-to-one translation (`EventTranslator`).
3. Introduce/use `KISRetrievalPlan` rather than loose parallel dense/BM25/retrieval string lists.
4. Remove top-level retrieval-preparation debug strings from the product KIS response only after temporal exploration is migrated to a dedicated `KISExplorationSeed`/plan snapshot contract. Exploration must no longer depend on `dense_events` / `bm25_events` aliases reconstructed by the frontend.
5. Migrate `useTemporalExploration`, `ExplorationOpenRequest`, `QueryBinding`, and the selected-video rescoring path to consume the dedicated exploration seed. The seed must preserve event order and retriever-facing text for S0, then extend to image refs in S1 so image-only events remain explorable.
6. Make frontend KIS revision updates transactional: old committed intent/results/exploration remain usable while a new semantic operation is pending; failure preserves them and preserves the draft.
7. Fix duplicate frame-selection propagation and make the selected frame carry the committed exploration snapshot exactly once while still recording the viewed frame exactly once.
8. Move history persistence outside the live search critical path using per-query ordered writes: activity writes for a query must wait for that query row creation, and late history responses must never replace the active live session.
9. Keep replay read-only and separate from live composer state.
10. Ensure readiness and inference error semantics reflect mandatory KIS dependencies.
11. Complete legacy/dead-code/generated-artifact cleanup from the earlier stabilization spec.

S1 implementation must not preserve an obsolete S0 abstraction merely to avoid migration work.

## 5. S1.1 — Explicit Event Routing With Scoped Semantic Resolution

### 5.1 `E#:` is routing syntax, not raw replacement

Input such as:

```text
E2: actually the man is a chef wearing black
```

means:

> use the full current intent as read-only semantic context, but allow the SLM/LLM to resolve and return only the semantic update for `E2`.

The event is not replaced with the literal user string. The scoped resolver may resolve pronouns, make the text self-contained, and select bindings permitted by the scoped contract.

### 5.2 Existing and appended events

If the current intent contains `E1..En`:

- `Ej` with `j <= n` updates exactly that existing event.
- `E(n+1)` creates exactly one new event through scoped semantic resolution.
- a contiguous suffix `E(n+1)..E(n+k)` may be added in one batch.
- gaps are invalid and rejected before model inference.
- duplicate event IDs in one command are invalid.

### 5.3 Batch patches

One composer submission may contain several explicit event instructions:

```text
E1: woman enters the kitchen
E2: she talks to a chef
E4: then she takes a white plate
```

The parser deterministically extracts the event IDs and instructions. The backend validates legality against the base intent. A single scoped model request may resolve all explicitly named events using the full intent as read-only context, but the output is allowed to contain only those target event IDs.

One successful batch is one semantic revision.

### 5.4 Global rewrite is explicit

The only command that grants the model permission to reconsider the semantic representation globally is:

```text
/llm-rewrite
<instruction>
```

The global rewriter may rewrite all event text, resolve coreference globally, rebuild entities/bindings, and regenerate `query_text`.

It must preserve:

- event IDs,
- event count,
- event order,
- attached image ownership,
- the sequential temporal topology.

It may not split, merge, insert, delete, or reorder events in this phase. Structural edits remain explicit `E#:` operations.

## 6. S1.2 — Stateless Canonical Intent Snapshot

### 6.1 `KISIntent` is semantic state, not conversation history

The target canonical intent is conceptually:

```text
KISIntent
  revision
  language: vi | en | null
  query_text: string | null
  entities[]
  events[]
  temporal_edges[]
```

Raw user inputs/clue strings are not stored as semantic truth inside `KISIntent`. For an image-only intent with no textual semantics, `language` and `query_text` are `null`; the server must not invent a language or textual description merely to satisfy the schema. Text-bearing intents retain explicit `vi`/`en` language metadata.

A scoped operation may perform the first textual mutation of an image-only intent. In that case `base.language == null` is valid, the scoped model returns the detected textual language (`vi` or `en`), and the successful new intent adopts that language. Once a base intent already has `vi`/`en`, later scoped outputs must match it unless the user explicitly performs a global rewrite that changes language. This transition must be covered by an image-only `E1` -> text+image `E1` test.

Interaction history and operation logging are separate concerns used for replay, analysis, and UI history.

### 6.2 Revision ownership

Revision no longer equals number of clues or messages.

A successful semantic operation increments once:

```text
Rev N -> Rev N+1
```

regardless of the number of event patches in that operation.

A retrieval-only re-search does not increment semantic revision.

### 6.3 Stateless request model

The backend remains stateless for semantic session state. A request supplies the current canonical snapshot plus one operation:

```text
KISSearchRequest
  base_intent: KISIntent | null
  expected_revision: int
  operation: InitialResolve | PatchEvents | GlobalRewrite | SearchOnly
  retrieval options
```

Validation:

- `base_intent == null` requires `expected_revision == 0`.
- otherwise `expected_revision` must equal `base_intent.revision`.
- semantic success returns a new snapshot with `revision + 1`.
- `SearchOnly` returns the same semantic revision.

A revision mismatch is HTTP 409.

## 7. S1.3 — Multimodal Event Model

### 7.1 Event container

An event becomes a multimodal container:

```text
KISEvent
  id
  text: string | null
  images: KISImageRef[]
  bindings[]
```

Invariant:

```text
text is non-empty OR images is non-empty
```

Valid events include:

- text only,
- image only,
- text + image.

### 7.2 Image asset reference

Images are uploaded once and referenced by opaque asset identity:

```text
KISImageRef
  asset_id
  content_type
```

A bounded query-asset endpoint validates supported formats, computes a content identity, stores/deduplicates the image, and returns the reference. Progressive revisions reuse refs rather than resending historical image bytes.

The same asset identity must also be retrievable for display. A read endpoint such as `GET /api/v1/kis/assets/images/{asset_id}` returns the validated original bytes with the stored content type and immutable caching headers. Canonical `KISImageRef` stays deployment-neutral (asset ID + content type); the frontend derives the display URL from the API route. This is required so event-card thumbnails survive replay/reload without retaining browser object URLs.

The exact storage implementation may be local/content-addressed initially; storage semantics must remain behind the asset reference contract.

### 7.3 Image ownership is deterministic

The LLM does not decide which event owns an image.

Image attachment/removal is an explicit deterministic operation associated with an event ID. The model may rewrite the target event text but cannot invent, remove, duplicate, move, or replace image attachments unless the user performed the corresponding image operation.

### 7.4 Initial multimodal behavior

To avoid ambiguous image ownership:

- initial text-only natural input may use the full resolver and may decompose into `E1..En`;
- initial image-only input creates one image-only `E1`;
- initial unscoped text + image creates one multimodal `E1`;
- initial multi-event multimodal input must use explicit `E#:` routing so image ownership is known.

## 8. S1.4 — Query Composer UX

### 8.1 One composer, structured event state

The KIS panel becomes a keyboard-first multimodal Query Composer.

Event cards represent the current committed semantic state. One shared composer/textarea is the textual command surface.

Conceptually:

```text
Current intent
...

E1  Woman enters kitchen
    [image thumbnails]

E2  Talks to chef
    [image thumbnails]

E3  Takes white plate

--------------------------------
E2: actually he wears black
[attach] [+ Event] [Update]
```

Event cards are not permanent editable form fields.

### 8.2 Input routes

The composer recognizes:

1. initial natural-language query;
2. explicit `E#:` patch batch;
3. `/llm-rewrite` global rewrite;
4. image attachment/removal associated with an event;
5. retrieval-only re-search when only search options change.

There is no persistent mode selector for these routes; syntax/state determines routing.

### 8.3 `+ Event`

`+ Event` prefills the next legal event prefix (`E(n+1):`) and focuses the composer. It does not create an empty event until the operation succeeds.

### 8.4 Image interaction

Supported fast paths:

- drag/drop or paste directly onto an event card;
- `Add image` from an event card;
- attach/paste while the composer clearly targets `Ej:`.

If an existing multi-event session has no explicit target and the user attaches an image, the UI must request a target event (`E1..En` or next event) rather than asking the model to infer ownership.

### 8.5 Composer preview and validation

The frontend parser may preview deterministic command meaning before submit, for example:

```text
Will update: E2, E4
Global rewrite
Missing E3
Duplicate E2
```

The backend remains authoritative and repeats structural validation.

### 8.6 Dynamic action copy

Recommended submit labels:

- `Search` for the first semantic search,
- `Update` for scoped/image mutations,
- `Rewrite` for `/llm-rewrite`.

The exact label styling is product polish; command semantics must not depend on the label.

### 8.7 Current-state-first history

The panel primarily shows the current canonical intent and event sequence. Raw chat history is not the main semantic UI.

A collapsed history may summarize operations such as:

```text
Rev 4 · Updated E2
Rev 5 · Added E4
Rev 6 · Global rewrite
```

Entities/bindings remain collapsed/debug detail rather than primary competition UI.

### 8.8 Transactional behavior

While a semantic update is pending:

- the previous committed intent remains active;
- previous results remain visible and inspectable;
- previous exploration snapshot remains valid;
- the draft and staged attachments remain visible;
- result browsing is not disabled.

On success, the new intent/results commit atomically. On failure, committed state is untouched and draft/attachments remain available for correction or retry.

History persistence is ordered but non-blocking. The frontend may expose the live result immediately, but each `query_id` owns a small write queue whose first operation is query-row creation. Viewed-frame/activity writes for that query are enqueued behind creation and therefore cannot race a missing history row. A late history completion may update status only if its `query_id` still matches the same live session; it must never reactivate or overwrite a newer live/replay session.

## 9. S1.5 — Progressive Operation State Machine

### 9.1 Operations

Conceptual operation types:

```text
InitialResolve
  natural text or initial explicit event batch

PatchEvents
  EventPatch[]

GlobalRewrite
  instruction

SearchOnly
```

An `EventPatch` may contain:

```text
event_id
instruction?        # scoped SLM/LLM semantic update
add_image_ids[]     # deterministic
remove_image_ids[]  # deterministic
```

An image-only patch does not call the LLM.

### 9.2 Scoped resolver contract

The scoped resolver receives:

- full base intent as read-only context;
- exact target event IDs;
- instructions for those targets.

It may return only resolutions for the named target IDs. It may not return mutations to other events, event ordering, revision, image ownership, or global topology.

Scoped output returns complete canonical bindings, not bare entity IDs. Each returned binding contains both `entity_id` and a non-blank semantic `role`, matching `KISEntityBinding`. When a base intent exists, every returned `entity_id` must already exist in `base.entities`; the server never invents a role after the model response. For an initial explicit batch with no canonical entity table yet, scoped bindings are empty.

Local event refinement should not silently mutate descriptions of global entities used by other events. It may reference existing entity IDs when confident. Bindings for a target event are replaced by the bindings explicitly resolved for that target; stale bindings are not preserved blindly.

If the local update introduces concepts that are not safely representable in the existing global entity table, the event text may carry that retrieval semantics without forcing speculative global entity mutation. `/llm-rewrite` is the explicit mechanism for global semantic normalization.

### 9.3 Global rewriter contract

`GlobalRewrite` may regenerate:

- event text for all existing event IDs,
- canonical `query_text`,
- entities,
- bindings,
- language metadata when appropriate.

It must preserve event topology and image ownership exactly.

### 9.4 Search-only contract

Changing retrieval settings (`top_k`, enabled retrieval sources, or equivalent search parameters) does not represent a semantic mutation.

`SearchOnly` reruns retrieval using the same canonical `KISIntent` revision.

This distinction is required for correct experiment logs and future EventTrail actions.

## 10. Multimodal Retrieval Semantics

### 10.1 Retrieval projection

`KISRetrievalPlan` is the modality contract between semantic intent and retrieval.

Each event projection contains aligned views such as:

```text
KISRetrievalEvent
  event_id
  canonical_text?
  dense_text?
  bm25_text?
  image_refs[]
```

The plan preserves event IDs and order exactly. Plan construction is modality-aware: only text-bearing rows are sent through translation, translated outputs are restored to their original event positions, and every event copies its canonical image refs unchanged. Image-only rows therefore remain present in the plan with `canonical_text/dense_text/bm25_text = null` and non-empty `image_refs`.

### 10.2 Text evidence

Text events continue to use the existing retrieval components and one-to-one translation behavior where required. Translation may alter retriever-facing text but not canonical semantic text.

### 10.3 Image evidence

Image refs are encoded with the image tower compatible with the canonical visual frame index and produce an event-by-frame image evidence component (conceptually `visual_image`).

`visual_image` must have positive configured mass in every production/baseline configuration that enables this component; adding a Python default alone is insufficient because YAML-provided `base_component_weights` replace the default dictionary. Baseline-loaded tests must prove an image-only event receives non-zero ranking signal.

Image-only events therefore remain valid temporal events without synthesized text.

### 10.4 Text + image fusion

S1 does not introduce a novel embedding-fusion method. Text and image remain independent evidence components and are combined by the existing calibrated/reliability-aware temporal fusion mechanism.

Missing modalities contribute no effective weight.

For multiple images attached to one event, images are treated as alternative visual exemplars for that event; the initial retrieval implementation should use a simple deterministic score-level aggregation (MAX is the preferred default) rather than averaging incompatible semantics into one query embedding.

### 10.5 No automatic image caption retrieval path

The system does not automatically caption query images and feed generated captions into BM25/text retrieval as part of this design.

### 10.6 Temporal exploration migration

Temporal exploration is a consumer of the exact scoring projection used by the committed KIS result. Removing top-level prepared text must therefore be paired with a dedicated exploration snapshot rather than dropping the data. `KISSearchResponse` carries an `exploration_seed` generated from the committed `KISRetrievalPlan`. The seed preserves event IDs/order plus per-event canonical text, dense text, BM25 text, and image refs, together with retrieval-source flags and semantic revision.

`useTemporalExploration` stores this seed unchanged and sends it to `ExplorationOpenRequest`; it must not rebuild prepared rows from public response aliases. The backend converts the seed back into the plan/query binding used for selected-video scoring. During S0 this supports textual events; after image scoring lands it also accepts image-only events and uses the same `score_plan()` path as the original KIS search. Thus exploration remains available after `dense_events`/`bm25_events` are removed and remains valid for future EventTrail work.

## 11. API and Error Semantics

### 11.1 Product response

The target KIS search response centers on:

```text
intent
operation_summary
results
retrieval settings
exploration_seed
latency
warnings
```

`operation_summary` includes at least operation kind and affected event IDs so frontend history can show concise revision summaries.

Retriever-internal prepared strings are not exposed as loose top-level product state. The dedicated `exploration_seed` is the only response-owned scoring snapshot and exists solely so selected-video exploration can faithfully reproduce the committed retrieval projection.

### 11.2 Structural client errors — HTTP 422

Examples:

- invalid `E#:` grammar,
- duplicate target IDs,
- illegal event gaps,
- invalid image/event target,
- unscoped free-form progressive mutation after an intent exists,
- invalid operation shape.

### 11.3 Revision conflict — HTTP 409

Returned when the supplied `expected_revision` does not match the base intent snapshot.

### 11.4 Inference errors

- provider/configuration unavailable -> HTTP 503;
- model output violates scoped/global semantic contract -> HTTP 502.

Inference failure must not partially commit a semantic revision.

## 12. Logging for Research and EventTrail

S1 should establish operation-level logging now so the EventTrail study does not need retrofitted semantics later.

At minimum log:

```text
semantic_revision
operation_kind
affected_event_ids
image_attach/remove
search_only
result_open
video/frame inspection
submission
latency stages
```

EventTrail-specific feedback (`confirm`, `reject`, `undo`, etc.) is added in the next design cycle, not invented in S1.

## 13. Target Ownership / Module Direction

Exact filenames are finalized during implementation planning after repository-wide reference tracing, but responsibilities should converge toward:

```text
src/hcmai/kis/
  models.py               # canonical multimodal KIS intent + resolution schemas
  parser.py               # deterministic E#: / command grammar
  resolver.py             # initial semantic resolution
  scoped_resolver.py      # PatchEvents semantic resolution
  rewriter.py             # /llm-rewrite semantics

src/hcmai/retrieval/
  plan.py                 # KISRetrievalPlan
  translation/            # EventTranslator
  ... existing scorers

src/hcmai/api/contracts/kis.py
src/hcmai/api/routers/kis.py

frontend/src/features/kis/
  session.js
  parser.js
  components/
    KisPanel.jsx
    IntentSummary.jsx
    EventList.jsx
    EventCard.jsx
    QueryComposer.jsx
```

Do not create these files mechanically if current module boundaries provide a cleaner equivalent. Ownership is more important than filename count.

## 14. Legacy Removal / Migration Map

After callers are migrated, the S0/S1 plan must remove or supersede production use of:

```text
KISIntent.inputs as canonical state
revision == len(inputs)
committedInputs as semantic source of truth
separate product-level image-search bypass for KIS
QueryPreparationService / query-candidate generation
/api/v1/query-candidates
parallel dense_events / bm25_events product response state after exploration migrates to `exploration_seed`
raw clue replay into live draft
implicit progressive free-form mutation of all events
```

The existing standalone low-level image encoder/retrieval utilities may remain where they have independent uses, but the KIS product flow must no longer branch around `KISIntent` when an image is supplied.

## 15. Non-Goals

S0/S1 does **not** implement:

- EventTrail UI;
- event confirm/reject/anchor interaction redesign;
- arbitrary temporal graphs beyond the ordered event chain;
- model-learned image-text fusion;
- embedding-fusion contribution experiments;
- automatic VLM captioning of query images as canonical semantics;
- automatic image-to-event ownership inference;
- free-form LLM structural edits that add/delete/reorder events;
- entity ReID/entity-consistency research;
- AVS/VQA redesign beyond what is necessary to preserve compatibility.

## 16. Completion Criteria

S0/S1 is complete only when:

1. the unfinished stabilization defects in the reviewed source are resolved before the new contracts become the only path;
2. one canonical `KISIntent` can represent text-only, image-only, and text+image events;
3. initial natural text can still resolve into multiple events;
4. explicit `E#:` batches mutate only the named events;
5. contiguous new events can be appended without global rewrite;
6. `/llm-rewrite` can globally normalize semantics while preserving topology and image ownership;
7. image-only patches do not invoke the LLM;
8. semantic operations increment revision exactly once; `SearchOnly` does not;
9. invalid event grammar/gaps are rejected deterministically before inference;
10. image assets are uploaded once, reused by opaque refs across revisions, and remain retrievable for event-card thumbnails after replay/reload;
11. text/image evidence is projected into the same temporal score/DP pipeline, and the baseline-loaded fusion configuration assigns positive mass to `visual_image`;
12. `KISRetrievalPlan` construction translates only textual rows, restores original event positions, and preserves image refs;
13. temporal exploration no longer depends on loose `dense_events`/`bm25_events` fields and can open a branch from the committed exploration seed, including image-only events after multimodal scoring lands;
14. separate KIS image-search bypass is removed from product orchestration;
15. frontend updates are transactional and preserve committed results on failure;
16. history writes are per-query ordered so activity cannot race query creation and late responses cannot replace the active session;
17. replay/history remain separate from live semantic state;
18. operation-level logs are sufficient to support the later EventTrail evaluation;
19. scoped resolver outputs preserve full binding roles and support the `language=null` -> first textual language transition;
20. repository-wide dead-code/reference scans show no production caller depending on superseded clue-list/query-expansion/separate-image-KIS abstractions.

## 17. Follow-On Design Cycle

After S0/S1 passes its completion gate, the next architectural cycle is **EventTrail**. It will consume the stable event IDs, multimodal event semantics, retrieval snapshots, and operation logging defined here. EventTrail research/design should then decide how candidate temporal evidence is visualized and how confirm/reject/anchor/explore feedback updates DP hypotheses without weakening the fast ranked-grid workflow.
