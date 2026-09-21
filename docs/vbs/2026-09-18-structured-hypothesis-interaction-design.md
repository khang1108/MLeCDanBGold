# Structured Hypothesis Interaction Design

**Date:** 2026-09-18  
**Target source:** `src_v2.0`  
**Status:** Approved brainstorming design, pending written-spec review  
**Scope:** KIS query interpretation and selected-video temporal interaction. AVS and transition-aware scoring are explicitly out of scope.

## 1. Summary

This design replaces two opaque and unstable interaction patterns in the current KIS workflow with one explicit hypothesis-oriented interaction model.

The current system has two observed problems:

1. The initial LLM resolver is the semantic authority for event count, event boundaries, and English event wording. The same natural-language clue can therefore produce materially different event structures and retrieval semantics across runs.
2. EventTrail `Decline` rejects only the current candidate's local DP cell. When several nearby frames represent the same temporal occurrence, repeated declines often surface another nearly identical candidate rather than a meaningfully different explanation.

The proposed system treats both query interpretation and temporal retrieval as **inspectable hypotheses**:

- the **Query Hypothesis** is a revisioned, source-grounded event topology proposed by the LLM and explicitly editable by the user;
- the **Result Hypothesis** is a complete temporal path through one selected video, accompanied by event-conditioned alternative hypotheses and explicit constraints.

Both sides follow the same interaction grammar:

`Proposal -> Inspect -> Preview -> Commit`

A preview never mutates canonical state. A commit always becomes an explicit deterministic state change and bumps a monotonic revision.

The core result-side interaction is **Branch-and-Commit**. When the user focuses event `Ei`, the system exposes temporally distinct complete-path alternatives conditioned on candidate occurrences for `Ei`. The user may preview a branch, anchor an occurrence, or reject an entire temporal mode rather than repeatedly declining adjacent frames.

The design deliberately does **not** require transition-aware scoring, a new VLM, a new online reranker, or a persistent search tree. It reuses the existing KIS intent, temporal DP, EventTrail constraint algebra, session snapshots, and undo infrastructure in `src_v2.0`.

---

## 2. Research framing

The system is framed as **Structured Hypothesis Interaction for Interactive Video Retrieval**.

The research question is not whether an LLM can generate a better free-form rewrite, nor whether users can provide generic relevance feedback. Instead, the system asks how uncertainty in both query interpretation and temporal retrieval can be surfaced as structured, inspectable hypotheses that users can constrain directly.

The two hypothesis spaces are:

- `HQ`: query interpretation, represented as an ordered event topology grounded in the original user query;
- `HR`: temporal retrieval, represented as complete ordered event paths within a selected video.

The shared interaction principle is:

`HypothesisState --user constraint--> HypothesisState'`

The code does not need a generic `Hypothesis` superclass. Query and result hypotheses have different semantics and remain separate domain models. What is unified is the interaction grammar, revision semantics, and provenance model.

### 2.1 Proposed paper research questions

**RQ1.** How consistently can an LLM form retrievable event structures from complex KIS descriptions, and how effectively can source-grounded hypothesis editing mitigate decomposition errors?

**RQ2.** Can event-conditioned temporal hypotheses expose meaningfully distinct alternatives and reduce repeated local rejection during temporal result correction?

**RQ3.** Does structured hypothesis interaction reduce search effort and time-to-target compared with opaque LLM reformulation and sequential candidate rejection?

### 2.2 Intended contributions

If supported by experiments, the system may contribute:

1. a source-grounded, inspectable query hypothesis representation that separates user semantics from retrieval projection;
2. event-conditioned temporal hypothesis exploration over complete DP paths;
3. a constraint-based preview-and-commit interaction model using positive anchors and negative temporal-mode exclusions;
4. an empirical analysis of query-decomposition variability and interactive search efficiency in VBS-style multi-event retrieval.

The paper must not claim novelty as “dynamic programming” itself. DP is an implementation substrate. The research contribution is the explicit hypothesis representation, event-conditioned alternatives, and structured user interaction over the inference space.

---

## 3. Current `src_v2.0` facts that motivate the design

### 3.1 Initial query resolution is currently free-form semantic generation

`src/hcmai/kis/models.py` defines `KISInitialResolutionEvent.text` as one concise English retrievable moment. `src/hcmai/kis/resolution/prompts.py` instructs the model to separate sequential views/actions but also to author each event as a self-contained English sentence. `KISIntentResolver` then constructs `KISEvent.text` directly from that model output and rebuilds `KISIntent.query_text` by joining the generated event texts.

As a result, the initial resolver currently controls all of the following in one model call:

- event count;
- event boundaries;
- event granularity;
- event semantic wording;
- translation/paraphrase into English.

`temperature=0.0` does not remove ambiguity in this task definition.

### 3.2 Generic feedback can also mutate event structure

`src/hcmai/kis/feedback/models.py` exposes `EditIntentAction` and `RestructureAction`. The latter can replace a contiguous event block with newly authored event descriptions. This gives generic chat a second path for mutating canonical query topology and semantics.

The new design removes canonical topology ownership from generic chat. Chat may propose a structured edit, but only the Query Hypothesis mutation service commits it.

### 3.3 EventTrail rejects one local cell at a time

`src/hcmai/event_trail/decoding/decoder.py` stores rejection constraints as per-event tuples of `Interval`. `rejection_cell()` derives a midpoint-bounded temporal cell around one frame. `DeclineCandidate` appends that interval to `rejected_cells[event_idx]` and re-decodes.

This algebra is useful and should be retained. The problem is not the existence of interval exclusions; the problem is the current policy for deriving an exclusion interval from one frame.

### 3.4 Selected-video decoding currently returns one path

`decode_video_scores()` in `src/hcmai/orchestration/workflows/search/temporal.py` calls `align_video(..., paths=1, ...)`. `EventTrailSession` therefore owns one `current_path` plus one `last_valid_path`.

The underlying DP implementation already supports multiple paths, but current `align_video()` separates multiple returned paths only by the final event timestamp. Generic `paths > 1` is therefore insufficient for a focused-event interaction: alternatives for `E2` may still be nearly identical at `E2` while differing only at `En`.

### 3.5 Existing infrastructure worth preserving

The design intentionally reuses:

- `KISIntent` and sequential `KISEvent` IDs;
- monotonic KIS revisions;
- `AlignedPath` as the core complete temporal path representation;
- `EvidenceSnapshot` and selected-video evidence ownership;
- `ConstraintSnapshot(anchors, rejected_cells, window)`;
- `Interval` rejection and `Conditions/build_mask`;
- EventTrail session locking, revision guards, bounded storage, and checkpoints;
- existing diff-aware path updates;
- existing retrieval-plan separation between canonical and retrieval-specific text.

---

## 4. Design principles

### 4.1 LLMs propose; they do not silently own canonical state

Initial decomposition, scoped rewrites, and full rewrites may use an LLM, but any semantic or structural proposal that changes canonical query state must be visible before it is committed.

### 4.2 Source semantics and retrieval projection are separate

The system must preserve what the user actually said. Translation, dense-query wording, BM25 wording, or future retrieval-specific expansion belongs to a downstream projection layer and must not overwrite source provenance.

### 4.3 Structural edits are deterministic

Split, merge, reorder, add, anchor, reject, and undo are explicit operations over structured state. They do not require an LLM unless the user explicitly asks the LLM to propose wording.

### 4.4 Preview is non-mutating

A preview does not change a revision, constraints, history, anchors, rejected regions, or search results. Only an explicit commit mutates canonical state.

### 4.5 User actions constrain inference rather than replace inference outputs

Selecting a temporal alternative does not directly assign `session.current_path = alternative`. It commits the constraint that identifies the branch, then the decoder solves again under the new constraints.

### 4.6 No persistent branch tree

A result session owns one active hypothesis, one bounded current alternative set, and monotonic history/checkpoints. It does not persist an unbounded tree of every explored branch.

### 4.7 Advanced interaction remains optional

If the initial query hypothesis looks correct, the user may search immediately. If a result path looks correct, the user may submit immediately. The system must not force event-by-event approval or alternative exploration.

---

## 5. Overall lifecycle

The canonical KIS lifecycle becomes:

1. user enters a raw natural-language query;
2. the initial resolver proposes a source-grounded Query Hypothesis;
3. the user optionally inspects/edits topology;
4. explicit `Search` creates a search snapshot bound to the current query revision;
5. the user selects a promising video/result;
6. the Result Hypothesis Explorer opens from `snapshot_id + result_id` and the source query revision;
7. the user optionally previews alternative hypotheses and commits positive/negative constraints;
8. the user submits a selected frame/result.

There are two independent monotonic revisions:

- `query_revision`: canonical query-hypothesis revision;
- `result_revision`: selected-video hypothesis/constraint revision.

A result session is bound to the query revision and search snapshot that created it.

---

## 6. Query Hypothesis domain model

### 6.1 Reuse `KISIntent` as the canonical query-hypothesis core

Do not create a parallel `QueryHypothesis` object duplicating `KISIntent`. Evolve the existing intent/event models so they can preserve source provenance and explicit user overrides.

Conceptually:

```text
KISIntent
  revision
  query_text        # canonical original user query associated with this hypothesis
  language          # server-owned: vi | en | mixed
  events[]
  temporal_edges[]

KISEvent
  id
  text              # current canonical semantic text for the event
  source_provenance # optional original-query provenance
  origin            # source | user_override | user_added
  images[]
  bindings[]
```

`KISIntent.query_text` must represent the original user query associated with the hypothesis. It must no longer be reconstructed by joining LLM-authored event descriptions.

### 6.2 Source provenance

Each source-grounded text event may carry:

```text
SourceProvenance
  source_text
  start_char
  end_char
```

`start_char` and `end_char` refer to the canonical original query string. For this design, the canonical original query is the participant input after the same deterministic whitespace normalization already used by the resolver (`" ".join(text.split())`). Provenance indexes that canonical string, which is stored unchanged for the lifetime of the hypothesis. The server, not the LLM, computes and validates these spans.

`origin` has the following semantics:

- `source`: canonical event text is grounded directly in an original-query span;
- `user_override`: user explicitly changed the canonical wording while provenance still records the original source segment;
- `user_added`: user explicitly added semantics not represented by an original-query span; provenance is absent.

### 6.3 Initial resolver contract

Replace free-form English event authoring with a source-grounded segmentation proposal.

The initial structured output should contain ordered event source fragments copied from the original query. The model must not be asked to generate character indices. Indices are fragile under Unicode normalization and tokenization.

Conceptual output:

```json
{
  "events": [
    {"source_text": "Người đàn ông bước vào phòng"},
    {"source_text": "sau đó lấy chiếc cốc"},
    {"source_text": "rồi ngồi xuống bàn"}
  ]
}
```

The server aligns these fragments sequentially to the canonical original query, validates that each fragment is grounded, assigns `E1..En`, and creates adjacent temporal edges. Alignment is left-to-right so repeated source phrases resolve to the next valid occurrence after the previous event. If a fragment cannot be verified against the canonical source string, the proposal is rejected rather than paraphrase-matched.

The resolver may decide event count and boundaries. It must not freely translate or paraphrase event semantics in the initial canonical hypothesis.

### 6.4 Granularity contract

The initial prompt should instruct the model to create the smallest chronological units that could reasonably be retrieved at different timestamps, while keeping attributes that are expected to co-occur in the same visual moment together.

Examples should include difficult multi-event cases, especially:

- several actions in sequence;
- a continuous camera move containing distinct retrievable moments;
- cooking/ingredient sequences similar to homogeneous L26-style material;
- simultaneous subject attributes that should remain one event;
- ambiguous descriptions where uncertainty must be preserved.

Few-shot examples improve proposal quality but are not a correctness boundary. The editor and grounding validation provide that boundary.

### 6.5 Grounding validation and safe failure

If initial output is malformed, empty, not sequentially alignable to the source query, or otherwise fails validation, do not retry multiple semantic-generation loops in the critical path.

Fallback canonical hypothesis:

- one event;
- event text equals the untouched original query;
- provenance spans the full query;
- adjacent temporal edges are empty because there is one event.

The UI should explain that automatic decomposition could not be verified and that the full query was retained as one editable event.

This is a safety fallback, not a deterministic linguistic splitter. The server must not introduce rule-based splitting on punctuation or words such as “then”, “after”, or “sau đó”.

### 6.6 Retrieval projection

Canonical query semantics and retrieval-specific text are distinct.

The existing `KISRetrievalEvent(canonical_text, dense_text, bm25_text, image_refs)` abstraction remains the right boundary. `canonical_text` comes from the committed Query Hypothesis. Translation or retrieval-specific rewriting belongs downstream and must not overwrite provenance.

Because the existing `EventTranslator.translate(events, language)` requires language metadata while `KISIntent` currently discards legacy `language`, the evolved Query Hypothesis restores a server-owned `language` field with values `vi`, `en`, or `mixed`. This metadata is used only to choose literal retrieval projection/translation behavior; it is not LLM-authored semantic content. English bypasses translation, while non-English/mixed text may use the existing literal one-to-one `EventTranslator` contract.

This design does not require a new online reranker or a new query object duplicating `KISRetrievalEvent`.

---

## 7. Query Hypothesis Editor

### 7.1 Canonical owner

The Query Hypothesis Editor and its backend mutation service are the only canonical owners of query topology after initial resolution.

Generic chat may produce proposals, but it does not directly commit topology changes.

### 7.2 Supported operations

The initial version supports:

- `Split(event_id, boundary)`;
- `Merge(event_id_a, event_id_b)` for adjacent events only;
- `Reorder(event_order)`;
- `Edit(event_id, text)`;
- `Add(position, text/images)`;
- `Undo`.

Every committed operation creates a new `KISIntent` revision. Event IDs are re-canonicalized to `E1..En` after structural changes, and adjacent `before` edges are rebuilt to match the committed order.

Undo restores an equivalent earlier semantic state but still creates a **new** monotonically increasing revision. Revision numbers never move backward.

### 7.3 Split

Split is a deterministic structural operation over one event. The user explicitly selects a valid boundary in the event/source text. The operation previews the two child events before apply.

If the event has attached images, every image must be explicitly assigned to at least one child before the split can be committed. The system does not silently duplicate or discard image evidence.

Image-only events cannot be text-split. The user may add another event explicitly instead.

### 7.4 Merge

Merge operates only on adjacent events. The merged event preserves source provenance where possible and combines image evidence as a deduplicated union.

Non-adjacent merge is not supported in the first version because the canonical domain is an ordered adjacent chain and arbitrary graph surgery would complicate semantics without serving the primary KIS use case.

### 7.5 Reorder

Reorder changes chronological topology explicitly. Source provenance stays attached to each event and therefore continues to record its original location in the natural-language query even if the committed event chronology differs from source order.

### 7.6 Edit and Add

Direct edit is an explicit user semantic override. The source provenance remains available for comparison and `origin` becomes `user_override`.

Add creates a user-authored event. Added semantics have `origin=user_added` and no source span unless the user later grounds them through a separate explicit operation.

### 7.7 Preview-before-commit

Structural changes and LLM-generated semantic rewrites must display the proposed change before apply. Preview does not bump revision.

For an LLM-generated full rewrite, show a side-by-side topology/semantic diff and require explicit `Replace hypothesis` to commit it.

### 7.8 Search boundary

Editing the Query Hypothesis does not automatically re-run retrieval. `Search` is an explicit commit boundary from query state to retrieval state.

Search results carry the `query_revision` they were produced from. If the query changes later, the old result grid is marked stale and must not be presented as results for the new revision.

---

## 8. Chat and feedback authority

### 8.1 Chat remains useful but is de-authorized

Chat remains a supporting interaction for explanation, semantic suggestions, and convenience. It no longer owns canonical query topology.

A chat request such as “split E2 into two events” should produce a structured proposal or open the Split affordance. It must not silently execute a `RestructureAction` against canonical state.

### 8.2 Existing feedback action migration

The current action space should evolve as follows:

- `RestructureAction`: absorbed into Query Hypothesis structural operations;
- `EditIntentAction`: retained only as an LLM/user proposal that requires preview and explicit apply;
- full LLM rewrite: retained as an explicit power tool with full diff and explicit replace;
- `RefineRetrievalAction`: may remain as a supporting retrieval-specific override so long as it does not change canonical topology;
- `AnchorAction` / candidate rejection semantics: owned by the Result Hypothesis Explorer;
- `ClarifyAction`: remains conversational;
- generic feedback undo: should not ambiguously undo query and result states together.

### 8.3 Scoped undo

Query edits and result constraints have separate undo histories.

- Query Undo creates a new Query Hypothesis revision equivalent to the previous semantic state.
- Result Undo creates a new result revision restoring previous constraints.
- Chat transcript itself is not a canonical undo stack.

---

## 9. Result Hypothesis domain model

### 9.1 Complete path as hypothesis

An `AlignedPath` is the core complete temporal hypothesis:

`E1@t1 -> E2@t2 -> ... -> En@tn`

Do not duplicate its frame/timestamp payload in a second domain object unless needed for an API presentation view.

The interaction layer needs stable opaque IDs so the client can refer to hypotheses and alternatives without submitting arbitrary frame arrays or trusting client-authored timestamps.

### 9.2 Result session evolves from EventTrailSession

The existing `EventTrailSession` becomes the implementation foundation for the Result Hypothesis Explorer. It retains:

- selected video evidence;
- decoder configuration snapshot;
- monotonic result/trail revision;
- constraints;
- current active path;
- last valid path;
- history/checkpoints;
- selected submission frame;
- active/exhausted status.

It additionally exposes a bounded alternative set for the currently focused event and revision.

Conceptually:

```text
ResultHypothesisSession
  active_path
  focused_event_id
  alternatives[]
  constraints
  result_revision
  query_revision
  search_snapshot_id
  result_id
```

The implementation may retain the existing `trail_revision` field name initially for compatibility, but the product/research semantics are “result hypothesis revision”.

### 9.3 Alternative identity

Each current alternative receives an opaque `alternative_id` bound to:

- session ID;
- result revision;
- focused event;
- induced complete path/mode.

The client sends `alternative_id` for preview or commit. The backend rejects stale or foreign IDs.

Alternatives are invalidated by every committed result mutation. Preview does not invalidate them because it does not bump revision.

### 9.4 No persistent tree

The session caches only the bounded alternatives for the current focused event and revision. Changing focus replaces that cache. Historical semantic states remain recoverable through checkpoints/undo, not through a permanently materialized branch graph.

---

## 10. Event-conditioned temporal hypothesis generation

### 10.1 Why generic top-k paths are insufficient

The current DP can return multiple paths, but separation is applied at the final event timestamp. When the user focuses an interior event such as `E2`, generic top-k paths may all contain nearly the same `E2` occurrence.

The new decoder must generate alternatives conditioned on the **focused event**.

### 10.2 Conditional complete-path score

For event `Ei` at timestamp/frame position `t`, define the best complete-path score conditioned on using that occurrence:

`h_i(t) = max Score(t1, ..., t_{i-1}, t, t_{i+1}, ..., tn)`

subject to ordering and all active constraints.

Efficiently, the decoder computes:

- a forward best-prefix score ending at each possible occurrence of `Ei`;
- a backward best-suffix score starting at each possible occurrence of `Ei`;
- the conditional complete-path score by combining prefix and suffix while subtracting the focused event's unary score once to avoid double counting.

The decoder also retains predecessor/successor information needed to reconstruct the induced complete path for a selected occurrence.

This formulation remains compatible with future pairwise/transition-aware scoring, but transition-aware scoring is not required for this design.

### 10.3 Alternative peaks

The system identifies temporally separated high-quality local maxima of `h_i(t)` for the focused event and reconstructs one complete induced path for each selected peak.

Alternatives are therefore complete explanations, not isolated frame candidates.

An event alternative is meaningful only through the complete path it induces.

---

## 11. Temporal modes and rejection

### 11.1 Mode abstraction

Nearby frames belonging to the same temporal occurrence must not appear as independent alternatives. The system groups a focused event's nearby candidate occurrences into **temporal modes**.

Each mode contains:

- event ID/index;
- representative frame/timestamp;
- bounded temporal interval;
- conditional complete-path score;
- induced complete path.

### 11.2 First-version mode-region rule

The first version uses a deterministic separated-peak rule rather than a learned clusterer or noisy score-basin segmentation.

For a peak at time `t_j`, define its rejection radius from the nearest competing peak distance, capped by a server-side maximum radius:

`r_j = min(r_max, 0.5 * nearest_competing_peak_distance)`

If there is no competing peak, use `r_j = r_max`. The mode interval is clipped to the video's valid time domain. Neighbor-derived half-distances prevent adjacent mode regions from overlapping before clipping.

The exact numeric `r_max` and minimum peak separation are server configuration and evaluation parameters, not frontend knobs.

### 11.3 Reject mode

Replace the user-facing `Decline` concept with `Reject occurrence` (paper term: temporal-mode rejection).

Rejecting a mode appends the mode interval to the existing per-event `rejected_cells` interval tuple, then re-decodes under the updated constraints.

The downstream constraint algebra does not need to change: `Interval`, `Conditions`, and `build_mask` already support interval exclusions.

This is the key fix for repeated-decline behavior: one action excludes the temporal occurrence family rather than one midpoint cell.

---

## 12. Branch-and-Commit interaction

### 12.1 Focused-event alternatives

The first version exposes alternatives only for the event the user has selected in the Result Hypothesis Explorer. It does not present a global carousel of unrelated full paths.

The UI shows a compact alternative strip for the focused event, for example:

```text
CURRENT       ALTERNATIVES
18s           41s     70s     102s
```

Each alternative card represents a temporal mode and can expose the representative keyframe and score/rank metadata needed for interaction.

### 12.2 Preview

Previewing an alternative shows the complete induced path and a deterministic diff from the active path.

For each event, the UI may show:

- unchanged;
- selected branch event moved;
- indirectly adjusted to maintain the optimal ordered path.

Preview never:

- changes anchors;
- appends rejections;
- writes undo history;
- changes current path;
- bumps result revision.

Hover may show a lightweight ghost preview. Video seeking should require an explicit click, not mere hover.

### 12.3 Commit branch / Use

Committing an alternative must not directly assign the previewed path as canonical state.

If the focused event alternative represents `Ei@t`, commit creates the positive constraint identifying that occurrence, using the same canonical frame/anchor semantics already supported by the current decoder. The decoder then solves again under the new constraint.

This guarantees that the committed result remains explainable as the optimum under the user's constraints rather than as a client-selected arbitrary path snapshot.

### 12.4 Keep, Use, Reject vocabulary

The result-side mutation vocabulary should converge to three core actions:

- **Keep**: anchor the current active occurrence for the focused event;
- **Use**: anchor a chosen alternative/manual occurrence for the focused event;
- **Reject**: exclude the focused temporal mode.

Explore, seek, focus-event selection, and preview are non-mutating interactions.

This yields a simple constraint algebra:

- positive constraints: anchors;
- negative constraints: temporal-mode exclusions.

---

## 13. Unified interaction and state ownership

### 13.1 Canonical owners

`QueryHypothesisStore/service` owns:

- original query;
- event topology;
- event canonical semantics/provenance;
- query revision.

`ResultHypothesisSession` owns:

- selected video;
- source query revision/search snapshot;
- active temporal constraints;
- active complete path;
- focused-event alternatives;
- result revision.

Chat owns:

- conversational history;
- proposed structured actions.

Frontend-only UI state owns:

- hovered/previewed alternative;
- expanded panels;
- current visual focus;
- local draft edits before apply.

### 13.2 Query change after search

If a user edits the Query Hypothesis after a search, existing results are marked as based on an older query revision. The system does not silently reinterpret them as current.

A Result Hypothesis session remains bound to its source query revision. If the user changes the query while the session is open, the session becomes stale relative to the current query but may remain inspectable as the old semantic world. It cannot merge result constraints into the new query revision.

### 13.3 Product migration from EventTrail

The new product-facing selected-video interaction is **Hypothesis Explorer**, not a second feature beside EventTrail.

Existing EventTrail infrastructure is absorbed and evolved. Users should not be presented with both “Open EventTrail” and “Open Hypothesis Explorer” as parallel workflows.

---

## 14. Failure and conflict semantics

### 14.1 General rule

Failure never silently mutates canonical hypothesis state.

A committed operation has one of two outcomes:

- success: new monotonic revision;
- failure: canonical revision and state unchanged.

No half-committed state is allowed.

### 14.2 Query resolver failure

Malformed/ungrounded/timeout initial resolution falls back to the untouched original query as one source-grounded event. No repeated LLM retry loop is required in the critical path.

### 14.3 Query revision conflict

Every query mutation request carries `expected_query_revision`. If it does not match current state, return a conflict response and require the client to review the latest hypothesis. The server does not auto-rebase semantic edits.

### 14.4 Result revision conflict

Every result mutation/alternative commit carries `expected_result_revision` (or the existing trail revision field during migration). Stale alternatives are rejected. Alternative IDs are never reinterpreted against a newer revision.

### 14.5 Preview failure

A failed preview leaves constraints, active path, history, and revision unchanged. The active session remains usable.

### 14.6 Exhaustion

Rejecting the last feasible temporal mode is allowed. If no valid path remains, the session enters explicit `exhausted` state. The user may undo or exit.

The system must not prevent the user from rejecting a candidate merely because it is the final feasible hypothesis.

### 14.7 No distinct alternatives

This is not exhaustion. The active path remains valid, but the focused event has no sufficiently distinct alternative mode under current constraints. The UI reports that no distinct alternatives are available and does not fabricate low-quality alternatives.

### 14.8 Structural edit validation

Split, merge, reorder, and image reassignment are validated before commit. Invalid operations do not bump revision. A split with unresolved image assignment cannot create an intermediate invalid canonical state.

### 14.9 Undo

Undo is itself a committed revision that restores an equivalent previous state. Revision counters remain monotonic.

### 14.10 Snapshot/session expiry

If an EventTrail/Hypothesis Explorer session retains all selected-video evidence required to decode, it may continue after the originating short-lived search snapshot expires. If the hypothesis session itself expires, the user must reopen from a current search. The system must not silently re-run retrieval and pretend the new evidence is the same session.

---

## 15. Frontend interaction design

### 15.1 Query side

The existing `KisPanel` remains the KIS interaction container, but `EventList/EventCard` evolves from read-only/indirect-edit presentation into a Query Hypothesis Editor.

The initial fast path remains:

`query -> proposal looks correct -> Search`

The editor exposes direct structural affordances only when needed. It does not require explicit approval of every event.

The existing chat thread remains visible but proposals that would mutate canonical query state must open a preview/apply flow rather than silently changing intent.

### 15.2 Result side

The existing `EventTrailPanel` is evolved/renamed into the Hypothesis Explorer rather than duplicated.

The event rail remains useful for choosing the focused event. The evidence area gains:

- current occurrence;
- bounded event-conditioned alternatives;
- complete-path preview;
- direct/indirect path diff;
- Keep / Use / Reject controls.

The existing window controls and undo may remain where they fit the new interaction model.

### 15.3 Terminology migration

User-facing terminology should move away from `Approve` / `Decline` where practical:

- `Approve` -> `Keep`;
- `Decline` -> `Reject occurrence`;
- `Use` remains `Use` for explicit alternative/manual selection.

Internal compatibility names may remain temporarily during implementation but should not leak as two competing concepts in the UI or paper.

---

## 16. Logging and reproducibility

Hypothesis interaction should produce structured logs sufficient to replay the semantic decision path.

At minimum, committed/preview interactions should be attributable to:

- query revision;
- result revision, when applicable;
- action type;
- focused event ID;
- previewed alternative/mode ID, when applicable;
- committed constraint or query operation;
- source search snapshot/result ID;
- timestamp/latency.

A session should be reconstructible conceptually as:

```text
Q rev1
-> split E2
Q rev2
-> search
-> open V17
-> preview E2 alternative M2
-> reject E2 mode M1
-> preview M3
-> use M3
-> submit
```

This is more informative for research than generic chat-turn counts.

---

## 17. Evaluation design

Evaluation is deliberately split into three layers so that algorithm quality, query reliability, and human interaction are not conflated.

### 17.1 Experiment A: Query Hypothesis robustness

Build a pre-defined set of KIS queries covering:

- single-event descriptions;
- explicit multi-event chronology;
- long descriptions;
- camera/scene progressions;
- homogeneous cooking/ingredient sequences;
- ambiguous chronology.

Run the current free-form resolver and the source-grounded proposal resolver repeatedly under the deployment stack.

Report:

- event-count variability;
- event-boundary agreement/segmentation agreement;
- grounding violation rate;
- topology edit cost to a manually curated reference decomposition where available.

The claim is not that the LLM becomes perfectly deterministic. The claim is that query interpretation becomes observable, grounded, and cheaply correctable.

### 17.2 Experiment B: Offline temporal alternative quality

For selected-video event correction, compare:

- current local next-candidate/repeated rejection behavior;
- generic k-best paths;
- event-conditioned temporal modes.

Primary algorithm metrics:

- `TargetModeCoverage@K`: whether a ground-truth target occurrence lies inside one of the top-K exposed modes;
- `ModeRedundancy@K`: temporal redundancy/overlap among exposed alternatives;
- repeated-neighborhood behavior after rejection.

A mode-based decoder must demonstrate that it exposes meaningfully distinct alternatives before the UI interaction is evaluated.

### 17.3 Repeated Reject Rate

Define:

`RepeatedRejectRate = rejects whose next candidate remains in the same temporal neighborhood / total reject actions`

This directly captures the observed EventTrail failure mode.

The expectation is that temporal-mode rejection materially reduces this rate compared with cell-level decline.

### 17.4 Experiment C: 2x2 interaction study

Use a clean factorial design:

| Condition | Query side | Result side |
|---|---|---|
| A | current opaque/free-form resolver | current EventTrail cell decline |
| B | Query Hypothesis Editor | current EventTrail cell decline |
| C | current resolver | Hypothesis Explorer |
| D | Query Hypothesis Editor | Hypothesis Explorer |

This separates the effect of query-side reliability from result-side hypothesis exploration and allows measurement of the full system interaction.

### 17.5 Primary interaction metrics

Primary metrics:

- task success;
- time-to-target;
- interactions-to-target;
- videos inspected.

Hypothesis-specific secondary metrics:

- query edits;
- mode rejects;
- alternative previews;
- undo count;
- LLM calls per task;
- LLM latency per task;
- total interaction latency.

### 17.6 Structured-query subset

Report results both for all KIS tasks and for a pre-defined structured/multi-event subset. The subset criteria must be defined before evaluation, such as descriptions with at least two temporally distinct moments, explicit chronology, or scene/camera progression requiring multiple timestamps.

The subset must not be selected based on observed system performance.

### 17.7 Do not use retrieval accuracy alone as the main claim

The interaction contribution is not necessarily improved first-search recall. It is improved ability to reach and correct the intended structured hypothesis with fewer and more informative interactions.

---

## 18. Out of scope

This design explicitly excludes:

- transition-aware pairwise/camera-motion scoring;
- a new learned temporal encoder;
- online VLM reranking;
- AVS Harvest Workspace changes;
- persistent branch trees;
- arbitrary non-adjacent event graph editing;
- rule-based natural-language splitting;
- mandatory event approval before every search;
- automatically generated semantic labels for hypothesis clusters.

These may be evaluated later without changing the core hypothesis interaction architecture.

---

## 19. Source-level implementation boundaries

The eventual implementation plan should preserve the following ownership boundaries.

### Query hypothesis core

Expected primary areas:

- `src/hcmai/kis/models.py`
- `src/hcmai/kis/resolution/initial.py`
- `src/hcmai/kis/resolution/prompts.py`
- new focused query-hypothesis mutation/provenance modules under `src/hcmai/kis/`
- existing retrieval-plan projection under `src/hcmai/retrieval/plan.py`

### Chat migration

Expected primary areas:

- `src/hcmai/kis/feedback/models.py`
- `src/hcmai/kis/feedback/resolver.py`
- `src/hcmai/kis/feedback/service.py`

The plan should remove direct canonical topology authority from generic feedback rather than layering a second topology editor on top of it.

### Result hypothesis core

Expected primary areas:

- `src/hcmai/temporal/dp.py`
- `src/hcmai/orchestration/workflows/search/temporal.py`
- `src/hcmai/event_trail/decoding/decoder.py`
- `src/hcmai/event_trail/models.py`
- `src/hcmai/event_trail/actions/`
- `src/hcmai/event_trail/service.py`
- `src/hcmai/api/contracts/event_trail.py`

The plan should evolve EventTrail in place where possible instead of creating a parallel result-correction subsystem.

### Frontend

Expected primary areas:

- `frontend/src/features/kis/components/EventList.jsx`
- `frontend/src/features/kis/components/EventCard.jsx`
- `frontend/src/features/kis/components/KisPanel.jsx`
- `frontend/src/features/event-trail/components/EventTrailPanel.jsx`
- `frontend/src/features/event-trail/components/EvidenceInspector.jsx`
- `frontend/src/features/event-trail/hooks/useEventTrail.js`
- `frontend/src/api/eventTrail.js`

The implementation plan may split large components if required for clear responsibilities, but it should not introduce unrelated refactors.

---

## 20. Acceptance invariants

The implementation is conformant only if all of the following hold:

1. Initial LLM resolution can no longer silently replace the original query with freely authored English event semantics.
2. Every source-grounded event can be traced back to an original-query span or is explicitly marked as user-authored/overridden.
3. Query topology changes occur only through explicit revisioned operations; generic chat cannot silently restructure canonical state.
4. Search results are bound to the exact query revision that produced them.
5. The selected-video explorer exposes alternatives conditioned on the focused event, not merely generic top-k paths separated at the last event.
6. Alternatives are complete temporal hypotheses, not isolated frame suggestions.
7. Preview is non-mutating on both query and result sides.
8. Committing a temporal branch creates an explicit constraint and re-decodes; it never directly installs a client-selected path as canonical state.
9. Reject removes a bounded temporal mode/occurrence region rather than only the current midpoint cell.
10. Stale query revisions, stale result revisions, and stale alternative IDs are rejected without state mutation.
11. Exhaustion remains an explicit recoverable state and may be reached by a valid user rejection.
12. Query and result undo histories are scoped and revisions remain monotonic.
13. No persistent unbounded hypothesis tree is stored.
14. Advanced hypothesis interaction remains optional; fast-path Search and Submit remain available.
15. AVS and transition-aware scoring behavior are unchanged by this feature.

---

## 21. Design conclusion

The system should stop treating uncertainty as hidden model behavior that the user can only counter with another prompt or another decline. Instead, uncertainty becomes explicit structured state:

- the query is an inspectable event-topology hypothesis;
- a selected video is an inspectable temporal-path hypothesis;
- alternatives are event-conditioned complete-path explanations;
- user actions add explicit positive or negative constraints;
- every semantic mutation is revisioned, previewable, and recoverable.

This design turns the current LLM resolver and EventTrail from two loosely connected interactive features into one coherent hypothesis-interaction framework suitable for both competition use and empirical study.
