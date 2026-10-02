# KIS Critical Path Simplification & Motion-Aware Graph Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify the KIS critical path by removing legacy resolution overhead, making `KISRetrievalPlan` carry event transitions, eliminating duplicate internal validation, freezing baseline DP, and implementing `transition_decoder.py` for Motion-Aware Graph Decoding.

**Architecture:** 
1. Collapse execution to `QueryHypothesis` -> `search_only`.
2. Model domain transitions directly ($E_i \to E_{i+1}$) on `KISRetrievalPlan` instead of discarding graph edges.
3. Freeze `temporal/dp.py` as the Static Temporal Baseline, and implement `temporal/transition_decoder.py` incorporating pairwise transition edge scores $\psi(s, t; E_{i-1}, E_i)$.

**Tech Stack:** Python 3.12, Pydantic v2, NumPy, Pytest.
Virtual environment: `/home/phuckhang/MyWorkspace/HCMAI_2026/aic/bin/python`

**Spec:** `docs/superpowers/specs/2026-10-02-kis-critical-path-simplification-and-motion-graph-design.md`

## Global Constraints
- Target deadline: 2026-10-05.
- Preserve canonical identities: `video_id`, `frame_id`, `frame_idx`, `timestamp_ms`.
- Maintain full test suite pass rate (566 existing tests must remain green).
- Keep `temporal/dp.py` frozen as the reproducible baseline for experimental ablation.

---

### Task 1: Clean `QueryHypothesisService.__init__` and Concrete Typing in `build_retrieval_plan`

**Files:**
- Modify: `src/hcmai/kis/hypothesis/service.py:30-60`
- Modify: `src/hcmai/retrieval/plan.py:112-160`
- Test: `tests/kis/test_query_hypothesis_service.py`
- Test: `tests/retrieval/serving/test_serialization.py`

**Interfaces:**
- `QueryHypothesisService.__init__(self, store: QueryHypothesisStore, resolver: KISIntentResolver, image_canonicalizer: Any = None, clock: Callable[[], float] = time.time)`
- `build_retrieval_plan(intent: KISIntent, overrides: Mapping[str, Any] | None = None, ...)`

- [x] **Step 1: Inspect and run existing tests for QueryHypothesisService and plan serialization**
  Run:
  `/home/phuckhang/MyWorkspace/HCMAI_2026/aic/bin/pytest tests/kis/test_query_hypothesis_service.py tests/retrieval/serving/test_serialization.py -q`

- [x] **Step 2: Simplify `QueryHypothesisService.__init__`**
  Remove parameter-guessing heuristic logic (`isinstance(store_or_resolver, QueryHypothesisStore)`).
  Accept explicit `store: QueryHypothesisStore`, `resolver: KISIntentResolver`, `image_canonicalizer=None`, `clock=time.time`.

- [x] **Step 3: Remove duck-typing in `build_retrieval_plan`**
  In `src/hcmai/retrieval/plan.py`, replace `intent: Any` with `intent: KISIntent`.
  Directly access `event.id`, `event.text`, and clean up `override` extraction without dynamic `getattr` chains.

- [x] **Step 4: Verify tests pass**
  Run:
  `/home/phuckhang/MyWorkspace/HCMAI_2026/aic/bin/pytest tests/kis/test_query_hypothesis_service.py tests/retrieval/serving/test_serialization.py -q`

- [x] **Step 5: Commit changes**
  Commit message: `refactor(kis): clean QueryHypothesisService init and typed build_retrieval_plan`

---

### Task 2: Domain Model Simplification & `KISRetrievalPlan` Carrying Transitions

**Files:**
- Modify: `src/hcmai/kis/models.py:168-240`
- Modify: `src/hcmai/retrieval/plan.py:36-120`
- Modify: `src/hcmai/retrieval/serving/utils/serialization.py:20-55`
- Test: `tests/kis/test_models.py`
- Test: `tests/retrieval/serving/test_serialization.py`

**Interfaces:**
- `RetrievalTransition`: dataclass with `source_id: str`, `target_id: str`, `previous_event: KISRetrievalEvent`, `next_event: KISRetrievalEvent`.
- `KISRetrievalPlan`: has attribute `transitions: tuple[RetrievalTransition, ...]`.

- [x] **Step 1: Write unit test verifying `plan.transitions`**
  Add a test in `tests/retrieval/serving/test_serialization.py` (or a dedicated test) verifying that `KISRetrievalPlan` automatically carries `transitions` for multi-event intents ($E_1 \to E_2$).

- [x] **Step 2: Add `RetrievalTransition` to `src/hcmai/retrieval/plan.py`**
  Define `RetrievalTransition(source_id, target_id, previous_event, next_event)`.
  Update `KISRetrievalPlan` so `self.transitions` is deterministically computed for adjacent event pairs:
  ```python
  transitions: tuple[RetrievalTransition, ...] = field(init=False)
  ```
  or derived in `__post_init__`.

- [x] **Step 3: Simplify `KISIntent` edge validation**
  In `src/hcmai/kis/models.py`, allow `temporal_edges` to default to the sequential adjacent chain if omitted:
  `[(E1, E2), (E2, E3), ...]`.
  Remove redundant manual edge validation while keeping `temporal_edges` serialization for API consumers.

- [x] **Step 4: Update serialization utilities if needed**
  Verify `plan_to_schema` and `schema_to_plan` round-trip accurately with `transitions`.

- [x] **Step 5: Run tests and commit**
  Run:
  `/home/phuckhang/MyWorkspace/HCMAI_2026/aic/bin/pytest tests/kis/test_models.py tests/retrieval/serving/test_serialization.py -q`
  Commit message: `feat(retrieval): carry event transitions in KISRetrievalPlan and simplify edge modeling`

---

### Task 3: Eliminate Redundant Defensive Checks & Dead Literal Checks

**Files:**
- Modify: `src/hcmai/orchestration/pipeline.py:700-705`
- Modify: `src/hcmai/common/config.py` (if dead literal check exists in fusion/cache config)
- Test: `tests/orchestration/test_pipeline.py`

**Interfaces:**
- Remove redundant post-construction check: `if plan.event_ids != tuple(event.id for event in intent.events): raise ValueError(...)` in `search_kis()`.

- [x] **Step 1: Remove redundant post-builder check in `pipeline.py`**
  Remove lines 703-704 in `src/hcmai/orchestration/pipeline.py` where `plan.event_ids` is compared against `intent.events` right after `build_retrieval_plan(intent)`.

- [x] **Step 2: Audit and eliminate dead literal checks in retrieval configs**
  Inspect `src/hcmai/retrieval/` and remove checks asserting that `method == "rrf"` when `Literal["rrf"]` is the only legal type.

- [x] **Step 3: Run pipeline tests to verify green**
  Run:
  `/home/phuckhang/MyWorkspace/HCMAI_2026/aic/bin/pytest tests/orchestration/test_pipeline.py -q`

- [x] **Step 4: Commit changes**
  Commit message: `refactor(orchestration): remove redundant internal invariant assertions`

---

### Task 4: Streamline `SearchService` Execution Path around `QueryHypothesis` & `search_only`

**Files:**
- Modify: `src/hcmai/orchestration/pipeline.py:343-425, 676-725`
- Test: `tests/orchestration/test_pipeline.py`
- Test: `tests/test_kis_acceptance_smoke.py`

**Interfaces:**
- `SearchService.search_kis`: primary path executes `query_hypothesis_session_id` + `search_only`. Legacy `_resolve_operation` kept compact for backwards-compatible smoke tests.

- [x] **Step 1: Inspect `search_kis` execution flow**
  Ensure that when `request.query_hypothesis_session_id` is passed, `search_only` executes directly without legacy mutation branches.

- [x] **Step 2: Streamline `_resolve_operation`**
  Clean up redundant nested validation and eliminate unused mutation branches, keeping only what is required for test compatibility.

- [x] **Step 3: Run acceptance and pipeline tests**
  Run:
  `/home/phuckhang/MyWorkspace/HCMAI_2026/aic/bin/pytest tests/orchestration/test_pipeline.py tests/test_kis_acceptance_smoke.py -q`

- [x] **Step 4: Commit changes**
  Commit message: `refactor(pipeline): streamline KIS search execution path`

---

### Task 5: Freeze Baseline DP and Implement `temporal/transition_decoder.py`

**Files:**
- Modify: `src/hcmai/temporal/dp.py` (add explicit docstring marking this module as the frozen Static Temporal Baseline)
- Create: `src/hcmai/temporal/transition_decoder.py`
- Create: `tests/temporal/test_transition_decoder.py`
- Test: `tests/temporal/test_event_conditioned_dp.py`
- Test: `tests/temporal/test_transition_decoder.py`

**Interfaces:**
- `TransitionScoreMatrix`: matrix / function evaluating transition compatibility $\psi(s, t; E_{i-1}, E_i)$ between frame $s$ (for event $i-1$) and frame $t$ (for event $i$).
- `decode_transition_graph(scores: np.ndarray, transitions: Sequence[TransitionScoreMatrix] | None, gap_penalty: float = 0.0) -> list[DPPath]`
- Invariant: When transitions are `None` or zero-weighted, `decode_transition_graph` yields identical results to `align_video` in `dp.py`.

- [x] **Step 1: Write unit tests in `tests/temporal/test_transition_decoder.py`**
  - Test 1: Zero transition weight matches baseline `dp.py` outputs exactly.
  - Test 2: Positive transition weight favors pairs of candidate frames with high transition affinity.
  - Test 3: Respects chronological monotonicity ($s < t$).

- [x] **Step 2: Run test to confirm it fails (TDD Red)**
  Run:
  `/home/phuckhang/MyWorkspace/HCMAI_2026/aic/bin/pytest tests/temporal/test_transition_decoder.py -q`

- [x] **Step 3: Implement `src/hcmai/temporal/transition_decoder.py`**
  - Implement DP recurrence:
    $$DP_i(t) = U_i(t) + \max_{s < t} \left[ DP_{i-1}(s) + \psi(s, t; E_{i-1}, E_i) - \lambda(t - s) \right]$$
  - Provide clean backtracking and ranking of top-k chronological paths.

- [x] **Step 4: Run tests to confirm it passes (TDD Green)**
  Run:
  `/home/phuckhang/MyWorkspace/HCMAI_2026/aic/bin/pytest tests/temporal/test_transition_decoder.py tests/temporal/test_event_conditioned_dp.py -q`

- [x] **Step 5: Document `temporal/dp.py` as Frozen Baseline**
  Add header docstring explaining that `dp.py` is the frozen baseline for experimental ablation in the paper.

- [x] **Step 6: Commit changes**
  Commit message: `feat(temporal): implement motion-aware transition graph decoder alongside frozen baseline`

---

### Task 6: Full Suite Verification & Update `KNOWLEDGE.md`

**Files:**
- Modify: `KNOWLEDGE.md`
- Test: All repository tests (`tests/`)

- [x] **Step 1: Run full repository test suite**
  Run:
  `/home/phuckhang/MyWorkspace/HCMAI_2026/aic/bin/pytest tests/ -q`
  Confirm all tests pass without regressions.

- [x] **Step 2: Update `KNOWLEDGE.md`**
  Document:
  - Architecture transition: removal of overengineered generic graph resolution.
  - Integration of `KISRetrievalPlan.transitions` and `transition_decoder.py`.
  - Experimental ablation design: Baseline DP vs. Motion-Aware Transition Graph Decoding.

- [x] **Step 3: Commit updates**
  Commit message: `docs(knowledge): document critical path simplification and motion graph decoding architecture`
