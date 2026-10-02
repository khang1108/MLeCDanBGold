# Design Specification: KIS Critical Path Simplification & Motion-Aware Graph Decoding

- **Date:** 2026-10-02
- **Author:** Pair Programming Agent & User
- **Status:** Approved
- **Target Deadline:** 2026-10-05

---

## 1. Problem Statement & Motivation

The project prepares for the HCMAI multimodal video retrieval benchmark with the research paper titled:
> **From Events to Transitions: Motion-Aware Graph Decoding for Multi-Event Video Retrieval**

Currently, the KIS retrieval critical path carries significant overengineering and legacy overhead:
1. `SearchService._resolve_operation()` in `src/hcmai/orchestration/pipeline.py` contains ~230 lines handling legacy `initial_resolve`, `patch_events`, `global_rewrite`, and string parsing, whereas modern UI flow has migrated to `QueryHypothesis` sessions executing purely via `search_only`.
2. `KISIntent` in `src/hcmai/kis/models.py` repeatedly validates an arbitrary generic graph with `KISTemporalEdge(source, relation="before", target)` despite the domain topology strictly being a 1D adjacent event chain ($E_1 \to E_2 \to \dots \to E_n$).
3. `KISRetrievalPlan` in `src/hcmai/retrieval/plan.py` discards edges and retains only events, creating an architectural disconnect between graph formulation in the paper and the actual retrieval plan.
4. `build_retrieval_plan` uses duck typing (`intent: Any`, `getattr` + dict accesses) rather than typed interfaces.
5. Invariant checks are repeated across multiple layers on objects freshly created by internal code (e.g. `plan.event_ids == intent.event_ids` right after calling `build_retrieval_plan(intent)`).
6. Dead literal checks exist (e.g. `config.method != "rrf"` when the config field is fixed to `Literal["rrf"]`).
7. `QueryHypothesisService.__init__` uses heuristic `isinstance` parameter swapping rather than explicit parameters.

Cleaning this critical path simplifies both the implementation and experimental ablation, ensuring the codebase cleanly expresses "From Events to Transitions".

---

## 2. Design Decisions & Architecture

### 2.1. Component 1: `QueryHypothesis` -> `search_only` Canonical Path
- **Canonical Flow:**
  $$\text{Query} \xrightarrow{\text{QueryHypothesisService}} \text{KISIntent } (E_1 \to \dots \to E_n) \xrightarrow{\text{User Review}} \text{search\_only} \xrightarrow{\text{KISPipeline}} \text{Candidates \& Temporal Decoding}$$
- In `pipeline.py`:
  - Retain `search_kis` with primary focus on `query_hypothesis_session_id` and `operation.kind == "search_only"`.
  - Slim down or isolate legacy `_resolve_operation` so it does not obstruct the critical path.
  - Remove active dependencies in frontend `SearchWorkspace.jsx` on the legacy `kisSession` state machine and `parser.js`.

### 2.2. Component 2: Domain Model Simplification (`KISIntent` & Transitions)
- Instead of requiring arbitrary graph input, define:
  ```python
  @dataclass(frozen=True, slots=True)
  class QueryTransition:
      source: str
      target: str
  ```
- For sequential events $E_1, \dots, E_n$, transitions are deterministically $E_i \to E_{i+1}$ for $i \in [1, n-1]$.
- `KISIntent.temporal_edges`:
  - Retained as a field with default factory or auto-population from adjacent event IDs to maintain 100% backward compatibility for API serialization and frontend components like `KisPanel.jsx`.
  - Eliminate repeated graph validation and manual LLM edge generation.
- Keep `entities` and `bindings` as empty default lists without heavy referential checks in the retrieval path.

### 2.3. Component 3: `KISRetrievalPlan` Carrying Transitions
- Define in `src/hcmai/retrieval/plan.py`:
  ```python
  @dataclass(frozen=True, slots=True)
  class RetrievalTransition:
      source_id: str
      target_id: str
      previous_event: KISRetrievalEvent
      next_event: KISRetrievalEvent
  ```
- Update `KISRetrievalPlan`:
  ```python
  @dataclass(frozen=True, slots=True)
  class KISRetrievalPlan:
      events: tuple[KISRetrievalEvent, ...]
      transitions: tuple[RetrievalTransition, ...]  # derived deterministically from events
  ```
- Update `build_retrieval_plan`:
  ```python
  def build_retrieval_plan(
      intent: KISIntent,
      overrides: Mapping[str, Any] | None = None,
      *,
      dense_text_by_event: dict[str, str] | None = None,
      use_dense: bool = True,
      use_bm25: bool = True,
  ) -> KISRetrievalPlan:
  ```
  Eliminate duck-typing and `getattr` checks.

### 2.4. Component 4: Eliminate Duplicate Validations & Clean Constructor
- Remove redundant validation:
  - Remove post-builder checks such as `if plan.event_ids != tuple(event.id for event in intent.events)` in `pipeline.py`.
  - Remove dead literal assertions (e.g. `if config.method != "rrf"`).
- Clean `QueryHypothesisService.__init__`:
  ```python
  class QueryHypothesisService:
      def __init__(
          self,
          store: QueryHypothesisStore,
          resolver: KISIntentResolver,
          image_canonicalizer: Any = None,
          clock: Callable[[], float] = time.time,
      ) -> None:
  ```

### 2.5. Component 5: Baseline Freeze & Motion-Aware Graph Decoder
- **Freeze Baseline:**
  - `src/hcmai/temporal/dp.py` remains untouched as the Static Temporal Baseline:
    $$DP_i(t) = U_i(t) + \max_{s < t} [DP_{i-1}(s) - \lambda(t - s)]$$
- **New Module: `src/hcmai/temporal/transition_decoder.py`**:
  - Implements Motion-Aware Graph Decoding:
    $$DP_i(t) = U_i(t) + \max_{s < t} [DP_{i-1}(s) + \psi(s, t; E_{i-1}, E_i) - \lambda(t - s)]$$
  - Evaluates pairwise candidate frame transitions $(s, t)$ using motion/transition edge weights $\psi$.

---

## 3. Verification & Testing Strategy
- Ensure all existing unit and integration tests remain passing (566 existing tests).
- Add specific unit tests for `KISRetrievalPlan.transitions` and round-trip serialization.
- Add unit tests for `transition_decoder.py` comparing against the baseline `dp.py`.
