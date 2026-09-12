# HCMAI Research Schema & API Contract v1

> **Status:** Authoritative contract specification for query-to-path flow cleanup and research integration.
> **Reconstituted from:** `docs/superpowers/plans/2026-09-11-hcmai-contract-cleanup.md` (§2, §3, §5) pursuant to Task 0 pre-task requirement.
> **Scope:** Query preparation, temporal orchestration, corpus evidence access, and output projection.

---

## 1. Global Invariants

1. **Canonical Identity:**
   - Always preserve: `video_id`, `frame_id`, `frame_idx`, and timestamp (`timestamp_ms`, `start_ms`, `end_ms`).
   - `frame_id`: Internal canonical string identifier used for joins, stores, evidence, and internal references.
   - `frame_idx`: Competition-facing integer frame coordinate used for official submission.
   - Do not confuse `frame_idx` with keyframe order, image filename index, array index, or decode position.
   - Modality fusion, rerankers, and decoders must NEVER overwrite, invent, or drop canonical identifiers.

2. **Zero API Surface Growth for Production:**
   - Existing HTTP endpoints (`/api/v1/search`, `/api/v1/trake`, `/api/v1/query-candidates`) maintain wire compatibility.
   - No public HTTP endpoints added in cleanup.
   - The workflow input-mismatch error `InvalidQueryInputError` maps to HTTP 422, while an untyped canonical-corruption `ValueError` remains a 500 server failure.

3. **Evidence Integrity:**
   - Missing evidence is NOT negative evidence (distinguish absent store/record from zero score/empty counts).
   - ASR timestamps follow half-open intervals `[start_ms, end_ms)`.
   - Raw queries vs. explicit event lists have distinct processing semantics:
     - Raw string splitting (`split_query_events`) may filter blank lines.
     - Explicit event sequences (`normalize_event_texts`) MUST NOT silently drop events; empty or whitespace-only items trigger `ValueError`.

4. **No Premature Abstraction:**
   - Do not introduce plan registries, generic constraint engines, universal DTOs, or abstract provider factories.
   - Single owner per semantic operation.

---

## 2. Ownership Map — Một chức năng, một owner

| Operation | Final Owner | Consumer | Disallowed Call Sites |
|---|---|---|---|
| Normalize explicit event strings | `src/hcmai/temporal/planner.py:normalize_event_texts` | Query prep, temporal search, research plan validator | Modules must not inline join/strip/drop empty lists independently |
| Raw KIS → ordered events | `src/hcmai/temporal/planner.py:plan_query_events` | KIS workflow; query-candidates raw branch | Router, SLM adapter, or scorer |
| Sentence/line splitting primitive | `split_query_events` (status quo for verified callers) | Planner; callers needing raw split | Must not be used as semantic planner replacement |
| Validate LLM bundle fidelity | `src/hcmai/query_preparation/service.py` | Query generation | HTTP router or temporal DP |
| Select original vs. retrieval text | KIS/TRAKE workflow at ingress; research runner for research | Temporal service | Dense retrieval must not silently rewrite text; BM25 must not receive rewrite instead of literal |
| Score / fuse evidence | `src/hcmai/retrieval/evidence/hybrid.py` & existing components | Temporal service, baseline runner | `QueryPlan`, verifiers, or materializers |
| Numeric path decode | `src/hcmai/temporal/dp.py` | Temporal service, baseline methods sharing exact semantics | Workflows writing ad-hoc recurrence |
| Canonical metadata lookup | `src/hcmai/corpus/corpus.py:Corpus` | Temporal service, materializer, collector | DTOs or model adapters |
| Canonical path validation | `src/hcmai/orchestration/materializer.py:SearchMaterializer.validate_aligned_path` | Temporal materialization, KIS/TRAKE projection | Copying identity checks across individual output builders |
| KIS representative selection | `src/hcmai/orchestration/materializer.py:SearchMaterializer.build_kis_result` | KIS | Baselines or research copying midpoint rule ad-hoc |
| TRAKE HTTP projection | `src/hcmai/orchestration/materializer.py:SearchMaterializer.build_trake_path` (static) | TRAKE workflow | Verifiers or DP |
| Research verification | `src/hcmai/research/path_verification.py` (post-cleanup spec) | Research runner | Existing image rerankers |
| Evaluator output envelope | Existing baseline runner / output owner | Benchmark, future research integration | Experiments inventing arbitrary custom JSON schemas |

---

## 3. Naming Conventions and Units

| Concept | Canonical Name | Compatibility Note |
|---|---|---|
| Original raw text | `original_events` (internal); `events` (HTTP wire) | Do not rename wire field |
| Dense retrieval text | `retrieval_events` (internal); `dense_events` (HTTP wire) | Mapping only at trust boundary |
| BM25 caption query | `caption_events` (internal) | Distinct from corpus captions or dense context queries |
| Time coordinates | `timestamp_ms`, `start_ms`, `end_ms`, `*_ms` | Milliseconds integer; ASR intervals `[start_ms, end_ms)` |
| Position in NumPy matrix | `position` / `positions` | Never name this `frame_idx` |
| Event index in plan | `event_index` / `event_indices` (0-based) | Do not introduce redundant `event_id` |
| Frame competition coordinate | `frame_idx` | Canonical competition submission integer |
| Path arrays | `AlignedPath.frame_idxs` | `DPPath.frame_idx` preserved as existing exception |
| Decoder path score | `AlignedPath.score` | Exported as `decoder_score` in research; never overwritten by verifier score |
| Final retrieval count | `top_k` (HTTP wire) | Never confuse with candidate frames per event |
| Research path budget | `candidate_path_limit` | Scoped to experiment config |

### Method Naming Convention
- `normalize_*`: Transforms and returns normalized data.
- `validate_*`: Raises error on failure or returns `None`.
- `score_*`: Computes evidence / matrix scores.
- `align_*`: Solves temporal path / dynamic programming.
- `build_*`: Projects internal representation to output structure.
- `verify_*`: Performs semantic claim evaluation / verification.

---

## 4. Evidence Access & Optionality

1. **Object Counts Evidence:**
   - `ObjectCountsStore.get_counts(frame_id)` is the current optionality owner:
     - Returns `None` if the frame record is missing or its status is not completed.
     - Returns `{}` if completed detection found zero objects.
     - Returns a defensive copy when counts are present.
   - `Corpus.object_counts(frame_id)` retains the existing `{}` fallback for current consumers.
   - `Corpus.object_counts_optional(frame_id)` is deferred until a collector or another real consumer is ready; introducing and consuming it belongs in one focused follow-up change rather than this cleanup freeze.

2. **ASR Evidence:**
   - Timestamped intervals `[start_ms, end_ms)`.
   - Speech near a frame is timeline evidence, not inherently frame-native.

---

## 5. Behavioral Coverage & Acceptance Invariants

| Invariant | Unit Target | Integration Target | Golden / Replay Target |
|---|---|---|---|
| Explicit events do not silently drop | `normalize_event_texts` rejects empty / whitespace items with `ValueError` | Query prep → temporal search pipeline | Exact event list preserved in order |
| Candidate vs. KIS planner parity | Same planned events from identical raw input | API fake adapters | Known divergent queries produce identical planned events |
| User error 422 vs. server corruption 500 | `InvalidQueryInputError` mapped to 422 in router | TestClient error responses | Internal ValueError remains 500 |
| Same-video canonical path | `validate_aligned_path` rejects cross-video identity or coordinate drift | Temporal search → materializer | All frame IDs match video canonical records |
| Score semantics preserved | Exact numeric output from `SearchMaterializer` | KIS & TRAKE workflows | All 5 baseline methods match exactly on fixed score matrices |
| ASR half-open interval | Transcript tests | Materializer / evidence collector | Exact `[start_ms, end_ms)` fixtures |
| Empty vs. unavailable object counts | `ObjectCountsStore.get_counts` unit tests | Existing `Corpus.object_counts` fallback remains compatible | Optional Corpus access waits for a real collector consumer |
| No API surface growth | Schema tests on FastAPI app | OpenAPI diff verification | Endpoint path set & schemas match golden contract |
| Single scoring pass | Call counter on scorers | Runner execution | `call_count == 1` per evaluation run |
