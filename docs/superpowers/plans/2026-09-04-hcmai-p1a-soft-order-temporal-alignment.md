# HCMAI P1a Competition-Safe Soft-Order Temporal Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a competition-safe soft-order temporal decoder that permits at most one locally reversed adjacent-event transition within 3000 ms, replaces the soft decoder's forward span penalty with an 8000 ms free-zone + capped linear-tail pairwise cost, preserves exact strict-DP rollback, and keeps production decoding near `O(NF)` per video.

**Architecture:** Keep `src/hcmai/temporal/dp.py` as the trusted strict decoder. Add a separate `src/hcmai/temporal/soft_order.py` implementing a two-state DP (`inversion unused` / `inversion used`) with an `O(NF^2)` test oracle and an optimized `O(NF)` decoder. Dispatch between strict and soft decoders only in `TemporalSearchService`, with P0 evidence/query preparation frozen.

**Tech Stack:** Python 3, NumPy, Pydantic v2, pytest, existing `VideoEventScores`, `DPPath`, `AlignedPath`, and `TemporalSearchService`.

**Spec:** `docs/superpowers/specs/2026-09-04-hcmai-p1a-soft-order-temporal-alignment-design.md`

## Global Constraints

- Query preparation is frozen.
- P0 evidence scoring/fusion is frozen during P1a evaluation.
- Event count is dynamic `N`; never assume `N=4`.
- Do not regenerate or rebuild Caption/OCR/Object/ASR/Context/FAISS/BM25 artifacts.
- Do not modify `src/hcmai/temporal/dp.py` behavior.
- `alignment_mode="strict"` must continue to call the current `rank_paths()` implementation.
- `alignment_mode="soft_order_p1a"` uses the new decoder.
- Production P1a allows at most one local inversion per complete path.
- Local reverse window is 3000 ms.
- Same canonical position and equal-timestamp transition are invalid.
- Event skipping is invalid.
- Forward gaps `<= 8000 ms` are free.
- Forward gaps above 8000 ms use a capped linear-tail penalty.
- All temporal windows use `timestamps_ms`, never `frame_idx`.
- Per-video decoder state must never cross video boundaries.
- Keep existing KIS/TRAKE HTTP/result contracts unchanged.
- Default production alignment mode remains `strict` until the rollout gate passes.
- Every production behavior change follows RED → GREEN → REFACTOR.
- Before claiming P1a complete, run the full verification commands in Task 8 and report their actual output.

---

# Verified Current Source Boundaries

The latest reviewed source around P1a is:

```text
src/hcmai/temporal/dp.py
  DPPath
  AlignedPath
  cluster_starts()
  align_video()
  rank_paths()

src/hcmai/common/config.py
  AlignmentConfig
    lambda_gap
    event_power
    chunk_size
    cluster_delta

src/hcmai/orchestration/workflows/temporal_search.py
  TemporalSearchService.search()
  current strict rank_paths() call at the path-decoding boundary
```

Current strict decoder call:

```python
rows = rank_paths(
    scores,
    lambda_gap=self.config.lambda_gap,
    max_rows=top_k,
    event_power=self.config.event_power,
    cluster_delta=self.config.cluster_delta,
)
```

Current `DPPath` remains:

```python
@dataclass(frozen=True, slots=True)
class DPPath:
    video_id: str
    score: float
    frame_idx: tuple[int, ...]
    frame_ids: tuple[str, ...]
```

Do not add diagnostics fields to this public dataclass in P1a.

---

# File Structure

## Create

```text
src/hcmai/temporal/soft_order.py

tests/temporal/__init__.py
tests/temporal/fakes.py
tests/temporal/test_strict_regression.py
tests/temporal/test_soft_order_oracle.py
tests/temporal/test_soft_order.py
tests/orchestration/test_soft_order_temporal_search.py

scripts/evaluate_temporal_p1a.py
```

## Modify

```text
src/hcmai/common/config.py
src/hcmai/orchestration/workflows/temporal_search.py
src/hcmai/temporal/__init__.py
```

## Keep Behavior Unchanged

```text
src/hcmai/temporal/dp.py
src/hcmai/retrieval/evidence/*
```

---

# Task 1: Freeze the Existing Strict Decoder as a Regression Oracle

**Purpose:** Before adding a second decoder, characterize strict behavior so P1a cannot accidentally change existing production results.

**Files:**
- Create: `tests/temporal/__init__.py`
- Create: `tests/temporal/fakes.py`
- Create: `tests/temporal/test_strict_regression.py`
- Read only: `src/hcmai/temporal/dp.py`

**Interfaces:**
- Produces: `make_video_scores(...) -> VideoEventScores`
- Produces strict regression fixtures reused by later tasks.
- Consumes: existing `align_video()` and `rank_paths()` unchanged.

- [ ] **Step 1: Create a small `VideoEventScores` factory**

Create `tests/temporal/fakes.py`:

```python
from __future__ import annotations

import numpy as np

from hcmai.retrieval.retriever.video_scores import VideoEventScores


def make_video_scores(
    *,
    video_id: str = "V0",
    scores: list[list[float]],
    timestamps_ms: list[int] | None = None,
    frame_idx: list[int] | None = None,
) -> VideoEventScores:
    matrix = np.asarray(scores, dtype=np.float32)
    frame_count = matrix.shape[1]

    if timestamps_ms is None:
        timestamps_ms = [i * 1000 for i in range(frame_count)]
    if frame_idx is None:
        frame_idx = [i * 25 for i in range(frame_count)]

    return VideoEventScores(
        video_id=video_id,
        scores=matrix,
        frame_ids=np.asarray(
            [f"{video_id}:{idx}" for idx in frame_idx],
            dtype=object,
        ),
        frame_idx=np.asarray(frame_idx, dtype=np.int64),
        timestamps_ms=np.asarray(timestamps_ms, dtype=np.float64),
    )
```

If the actual `VideoEventScores` constructor uses tuple/list values rather than NumPy arrays, adapt only the constructor arguments to its exact dataclass contract; keep the helper API above unchanged.

- [ ] **Step 2: Write strict single-event regression**

Create `tests/temporal/test_strict_regression.py`:

```python
import pytest

from hcmai.temporal.dp import align_video, rank_paths

from .fakes import make_video_scores


def test_strict_single_event_selects_max_emission():
    video = make_video_scores(scores=[[0.1, 0.9, 0.3]])

    paths = align_video(video, lambda_gap=1e-5, paths=1)

    assert len(paths) == 1
    assert paths[0].frame_idx == (25,)
    assert paths[0].frame_ids == ("V0:25",)
    assert paths[0].score == pytest.approx(0.9)
```

- [ ] **Step 3: Write strict chronology regression**

```python
def test_strict_decoder_rejects_reversed_row_maxima():
    video = make_video_scores(
        scores=[
            [0.1, 0.9, 0.1, 0.1],
            [0.8, 0.1, 0.1, 0.7],
        ],
    )

    path = align_video(video, lambda_gap=0.0, paths=1)[0]

    assert path.frame_idx[0] < path.frame_idx[1]
```

- [ ] **Step 4: Freeze rank diversification behavior**

```python
def test_rank_paths_prefers_each_videos_best_level_before_second_level():
    a = make_video_scores(
        video_id="A",
        scores=[
            [0.9, 0.8, 0.7],
            [0.1, 0.8, 0.9],
        ],
    )
    b = make_video_scores(
        video_id="B",
        scores=[
            [0.8, 0.7, 0.6],
            [0.1, 0.7, 0.8],
        ],
    )

    rows = rank_paths([a, b], lambda_gap=0.0, max_rows=4)

    assert {rows[0].video_id, rows[1].video_id} == {"A", "B"}
```

- [ ] **Step 5: Run the strict regression tests**

```bash
PYTHONPATH=src python -m pytest -q tests/temporal/test_strict_regression.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/temporal
git commit -m "test(temporal): freeze strict alignment behavior"
```

---

# Task 2: Extend `AlignmentConfig` Without Changing Strict Defaults

**Purpose:** Introduce explicit P1a configuration while leaving current deployments on strict mode.

**Files:**
- Modify: `src/hcmai/common/config.py:300-313`
- Create/modify: `tests/temporal/test_soft_order.py`

**Interfaces produced:**
- `alignment_mode`
- `reverse_window_ms`
- `forward_free_gap_ms`
- `reverse_penalty_max`
- `forward_gap_penalty_per_window`
- `forward_gap_penalty_cap`
- `max_local_inversions`
- `allow_same_frame`

- [ ] **Step 1: Write the failing default-config test**

```python
from hcmai.common.config import AlignmentConfig


def test_alignment_config_defaults_to_exact_strict_mode():
    config = AlignmentConfig()

    assert config.alignment_mode == "strict"
    assert config.reverse_window_ms == 3_000
    assert config.forward_free_gap_ms == 8_000
    assert config.reverse_penalty_max == 0.15
    assert config.forward_gap_penalty_per_window == 0.05
    assert config.forward_gap_penalty_cap == 0.30
    assert config.max_local_inversions == 1
    assert config.allow_same_frame is False
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src python -m pytest -q \
  tests/temporal/test_soft_order.py::test_alignment_config_defaults_to_exact_strict_mode
```

Expected: fail because `alignment_mode` and P1a fields do not exist.

- [ ] **Step 3: Add the config fields**

Extend `AlignmentConfig`:

```python
class AlignmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lambda_gap: float = Field(default=1e-5, ge=0.0)
    event_power: float = Field(default=1.0, gt=0.0, le=1.0)
    chunk_size: int = Field(default=65_536, ge=1)
    cluster_delta: float = Field(default=0.0, ge=0.0)

    alignment_mode: Literal["strict", "soft_order_p1a"] = "strict"

    reverse_window_ms: int = Field(default=3_000, ge=0)
    forward_free_gap_ms: int = Field(default=8_000, ge=0)

    reverse_penalty_max: float = Field(default=0.15, ge=0.0)
    forward_gap_penalty_per_window: float = Field(default=0.05, ge=0.0)
    forward_gap_penalty_cap: float = Field(default=0.30, ge=0.0)

    max_local_inversions: Literal[1] = 1
    allow_same_frame: Literal[False] = False
```

- [ ] **Step 4: Add invalid-config tests**

```python
import pytest
from pydantic import ValidationError


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reverse_window_ms", -1),
        ("forward_free_gap_ms", -1),
        ("reverse_penalty_max", -0.01),
        ("forward_gap_penalty_per_window", -0.01),
        ("forward_gap_penalty_cap", -0.01),
    ],
)
def test_soft_order_config_rejects_negative_values(field, value):
    with pytest.raises(ValidationError):
        AlignmentConfig(**{field: value})


def test_soft_order_contract_rejects_multiple_inversions():
    with pytest.raises(ValidationError):
        AlignmentConfig(max_local_inversions=2)


def test_soft_order_contract_rejects_same_frame_mode():
    with pytest.raises(ValidationError):
        AlignmentConfig(allow_same_frame=True)
```

- [ ] **Step 5: Run GREEN**

```bash
PYTHONPATH=src python -m pytest -q tests/temporal/test_soft_order.py
```

Expected: PASS.

- [ ] **Step 6: Re-run strict regression**

```bash
PYTHONPATH=src python -m pytest -q tests/temporal/test_strict_regression.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/hcmai/common/config.py tests/temporal/test_soft_order.py
git commit -m "feat(temporal): add conservative soft-order config"
```

---

# Task 3: Build the Test-Only Brute-Force Soft-Order Oracle

**Purpose:** Establish exact semantics before optimizing them.

**Files:**
- Create: `tests/temporal/test_soft_order_oracle.py`
- Extend: `tests/temporal/fakes.py`
- No production soft-order code yet.

**Test-only interfaces:**

```python
@dataclass(frozen=True)
class OracleResult:
    score: float
    positions: tuple[int, ...]
    used_inversion: bool
    inversion_event_index: int | None
    inversion_ms: int | None
    reverse_penalty: float
    forward_penalty: float
```

```python
bruteforce_soft_order(
    scores: np.ndarray,
    timestamps_ms: np.ndarray,
    *,
    reverse_window_ms: int = 3000,
    forward_free_gap_ms: int = 8000,
    reverse_penalty_max: float = 0.15,
    forward_gap_penalty_per_window: float = 0.05,
    forward_gap_penalty_cap: float = 0.30,
    cluster_starts_: np.ndarray | None = None,
) -> OracleResult | None
```

- [ ] **Step 1: Implement transition-cost helpers in the test module**

```python
def forward_cost(
    delta_ms: float,
    *,
    free_ms: float,
    per_window: float,
    cap: float,
) -> float:
    assert delta_ms > 0.0
    if delta_ms <= free_ms:
        return 0.0
    if free_ms <= 0.0:
        return cap
    return min(cap, per_window * ((delta_ms - free_ms) / free_ms))


def reverse_cost(
    delta_ms: float,
    *,
    window_ms: float,
    penalty_max: float,
) -> float:
    assert window_ms > 0.0
    assert -window_ms <= delta_ms < 0.0
    return penalty_max * (abs(delta_ms) / window_ms)
```

- [ ] **Step 2: Define exact admissibility**

For predecessor `p` and current `t`:

```python
delta = timestamps_ms[t] - timestamps_ms[p]
```

A transition requires:

```text
p != t
delta != 0
different score cluster when cluster_delta is active
```

Forward:

```text
delta > 0
0 -> 0
1 -> 1
```

Reverse:

```text
-reverse_window_ms <= delta < 0
0 -> 1 only
```

No skip edge exists.

- [ ] **Step 3: Implement the brute-force two-state DP**

Initialize:

```python
neg_inf = float("-inf")
dp = np.full((n_events, n_frames, 2), neg_inf, dtype=np.float64)
parent_frame = np.full((n_events, n_frames, 2), -1, dtype=np.int64)
parent_state = np.full((n_events, n_frames, 2), -1, dtype=np.int8)

dp[0, :, 0] = scores[0]
```

For every event/current/predecessor:
- evaluate forward from state 0 to 0;
- evaluate forward from state 1 to 1;
- evaluate reverse from state 0 to 1;
- never evaluate reverse from state 1.

Candidate:

```python
candidate = (
    dp[event - 1, p, previous_state]
    + scores[event, t]
    - transition_penalty
)
```

- [ ] **Step 4: Define deterministic tie-breaking once**

Use epsilon:

```python
TIE_EPS = 1e-8
```

Order:
1. higher score;
2. forward over reverse;
3. smaller transition penalty;
4. smaller predecessor canonical position.

Use this exact ordering in both oracle and optimized decoder.

- [ ] **Step 5: Test normal forward path**

```python
def test_oracle_forward_path_inside_free_window_has_no_penalty():
    scores = np.asarray(
        [
            [0.9, 0.1, 0.1],
            [0.1, 0.9, 0.1],
            [0.1, 0.1, 0.9],
        ],
        dtype=np.float64,
    )
    timestamps = np.asarray([1000, 4000, 7000], dtype=np.float64)

    result = bruteforce_soft_order(scores, timestamps)

    assert result is not None
    assert result.positions == (0, 1, 2)
    assert result.score == pytest.approx(2.7)
    assert result.forward_penalty == pytest.approx(0.0)
    assert result.used_inversion is False
```

- [ ] **Step 6: Test one local reverse**

Use event peaks at:

```text
E1 @ 10s
E2 @ 20s
E3 @ 18s
E4 @ 24s
```

Assert:
- event identities remain E1/E2/E3/E4;
- one reverse transition occurs from E2 to E3;
- `inversion_ms == 2000`;
- reverse penalty equals `0.10`.

- [ ] **Step 7: Test exact boundaries**

Create separate tests:

```text
forward delta 8000 ms  -> cost 0
reverse delta -3000 ms -> valid, cost 0.15
reverse delta -3001 ms -> invalid
delta 0                -> invalid
```

- [ ] **Step 8: Test two inversions are impossible**

Construct row maxima that require two reverses and one weaker valid alternative.

Assert the valid alternative wins.

- [ ] **Step 9: Test irregular timestamps**

Use:

```python
timestamps = np.asarray([0, 700, 2500, 7200, 13100], dtype=np.float64)
```

Design scores so frame-distance logic would disagree with timestamp-window logic. Assert timestamp semantics win.

- [ ] **Step 10: Test dynamic event count**

Parametrize:

```python
@pytest.mark.parametrize("n_events", [1, 2, 3, 6, 10])
```

Assert result path length equals `n_events`.

- [ ] **Step 11: Test cluster exclusion**

Use:

```python
cluster_starts_ = np.asarray([0, 0, 2, 2], dtype=np.int64)
```

Transitions within one cluster are invalid in either direction.

- [ ] **Step 12: Run**

```bash
PYTHONPATH=src python -m pytest -q tests/temporal/test_soft_order_oracle.py
```

Expected: PASS.

- [ ] **Step 13: Commit**

```bash
git add tests/temporal/test_soft_order_oracle.py tests/temporal/fakes.py
git commit -m "test(temporal): define soft-order DP oracle"
```

---

# Task 4: Implement `soft_order.py` and Prove It Against the Oracle

**Purpose:** Add the production P1a decoder with the same semantics as Task 3 and near-linear predecessor search.

**Files:**
- Create: `src/hcmai/temporal/soft_order.py`
- Extend: `tests/temporal/test_soft_order.py`
- Read/import: `DPPath`, `cluster_starts` from `src/hcmai/temporal/dp.py`

**Production interfaces:**

```python
@dataclass(frozen=True, slots=True)
class SoftOrderParams:
    reverse_window_ms: int = 3000
    forward_free_gap_ms: int = 8000
    reverse_penalty_max: float = 0.15
    forward_gap_penalty_per_window: float = 0.05
    forward_gap_penalty_cap: float = 0.30
```

```python
@dataclass(frozen=True, slots=True)
class SoftOrderDiagnostics:
    used_local_inversion: bool
    inversion_event_index: int | None
    inversion_ms: int | None
    emission_score: float
    forward_gap_penalty: float
    reverse_penalty: float
    total_transition_penalty: float
```

```python
@dataclass(frozen=True, slots=True)
class SoftOrderDecodedPath:
    path: DPPath
    diagnostics: SoftOrderDiagnostics
```

```python
def align_video_soft_order(
    video: VideoEventScores,
    *,
    params: SoftOrderParams,
    paths: int = 1,
    event_power: float = 1.0,
    cluster_delta: float = 0.0,
) -> list[SoftOrderDecodedPath]:
    ...
```

```python
def rank_paths_soft_order(
    videos: Sequence[VideoEventScores],
    *,
    params: SoftOrderParams,
    max_rows: int = 100,
    event_power: float = 1.0,
    cluster_delta: float = 0.0,
) -> list[SoftOrderDecodedPath]:
    ...
```

## Task 4A: Cost helpers

- [ ] **Step 1: Write failing helper tests**

```python
def test_forward_cost_has_eight_second_dead_zone():
    params = SoftOrderParams()

    assert _forward_transition_cost(8_000.0, params) == 0.0
    assert _forward_transition_cost(16_000.0, params) == pytest.approx(0.05)


def test_forward_cost_is_capped():
    params = SoftOrderParams()

    assert _forward_transition_cost(200_000.0, params) == pytest.approx(0.30)


def test_reverse_cost_is_linear():
    params = SoftOrderParams()

    assert _reverse_transition_cost(-1_000.0, params) == pytest.approx(0.05)
    assert _reverse_transition_cost(-3_000.0, params) == pytest.approx(0.15)
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src python -m pytest -q \
  tests/temporal/test_soft_order.py::test_forward_cost_has_eight_second_dead_zone
```

Expected: module/import failure.

- [ ] **Step 3: Create immutable params and safe cost functions**

```python
def _forward_transition_cost(delta_ms: float, params: SoftOrderParams) -> float:
    if delta_ms <= 0.0:
        raise ValueError("forward transition requires positive delta")

    free = float(params.forward_free_gap_ms)
    if delta_ms <= free:
        return 0.0

    if free <= 0.0:
        return float(params.forward_gap_penalty_cap)

    excess = delta_ms - free
    return min(
        float(params.forward_gap_penalty_cap),
        float(params.forward_gap_penalty_per_window) * (excess / free),
    )


def _reverse_transition_cost(delta_ms: float, params: SoftOrderParams) -> float:
    window = float(params.reverse_window_ms)
    if not (window > 0.0 and -window <= delta_ms < 0.0):
        raise ValueError("reverse transition is outside configured window")
    return float(params.reverse_penalty_max) * (abs(delta_ms) / window)
```

- [ ] **Step 4: Run helper tests**

```bash
PYTHONPATH=src python -m pytest -q tests/temporal/test_soft_order.py
```

Expected: PASS for currently implemented tests.

## Task 4B: First make a clear quadratic production decoder agree with the oracle

- [ ] **Step 5: Add private `_align_video_quadratic()`**

Implement:

```python
def _align_video_quadratic(
    video: VideoEventScores,
    *,
    params: SoftOrderParams,
    paths: int,
    event_power: float,
    cluster_delta: float,
) -> list[SoftOrderDecodedPath]:
    ...
```

Mirror Task 3 semantics exactly.

- [ ] **Step 6: Validate timestamp ordering**

Before recurrence:

```python
timestamps = np.asarray(video.timestamps_ms, dtype=np.float64)
if np.any(np.diff(timestamps) < 0):
    raise ValueError(
        "soft-order decoder requires canonical timestamps in non-decreasing order"
    )
```

Equal timestamps are allowed in metadata but never form an admissible transition.

- [ ] **Step 7: Preserve event-power preprocessing**

Use the same behavior as strict DP:

```python
scores = np.asarray(video.scores, dtype=np.float64)
if event_power != 1.0:
    scores = np.clip(scores, 0.0, None) ** event_power
```

Compute clusters after this transformation.

- [ ] **Step 8: Compare quadratic decoder to the oracle**

For hand-written scenarios:

```python
oracle = bruteforce_soft_order(...)
decoded = _align_video_quadratic(...)[0]

assert decoded.path.score == pytest.approx(oracle.score)
assert decoded.path.frame_idx == tuple(
    int(video.frame_idx[p]) for p in oracle.positions
)
```

- [ ] **Step 9: Run GREEN**

```bash
PYTHONPATH=src python -m pytest -q \
  tests/temporal/test_soft_order.py \
  tests/temporal/test_soft_order_oracle.py
```

Expected: PASS.

## Task 4C: Add the optimized decoder

- [ ] **Step 10: Precompute cluster starts and ends**

Use current:

```python
starts = (
    cluster_starts(scores, cluster_delta)
    if cluster_delta > 0.0
    else np.arange(n_frames, dtype=np.int64)
)
```

Create:

```python
def _cluster_ends(starts: np.ndarray) -> np.ndarray:
    frame_count = len(starts)
    ends = np.empty(frame_count, dtype=np.int64)

    run_start = 0
    while run_start < frame_count:
        cluster_start = int(starts[run_start])
        run_end = run_start
        while (
            run_end + 1 < frame_count
            and int(starts[run_end + 1]) == cluster_start
        ):
            run_end += 1

        ends[run_start : run_end + 1] = run_end
        run_start = run_end + 1

    return ends
```

Admissibility:
- forward predecessor `p < starts[t]`;
- reverse predecessor `p > ends[t]`.

- [ ] **Step 11: Keep the existing distinct-cluster feasibility check**

```python
frames = np.arange(n_frames)
if int(np.count_nonzero(starts == frames)) < n_events:
    return []
```

Because P1a still requires one distinct cluster per event.

- [ ] **Step 12: Implement forward candidate computation in linear passes**

For every previous state row, split predecessors by timestamp:

```text
recent free:
T - free_ms <= Tprev < T

linear tail:
T - cap_threshold_ms < Tprev < T - free_ms

fully capped:
Tprev <= T - cap_threshold_ms
```

When:

```python
per_window = params.forward_gap_penalty_per_window
free_ms = params.forward_free_gap_ms
cap = params.forward_gap_penalty_cap
```

and `per_window > 0`:

```python
cap_excess_ms = (cap / per_window) * free_ms
cap_threshold_ms = free_ms + cap_excess_ms
```

Default cap threshold is 56 seconds.

Required data structures:
- monotonic deque for recent-free candidates;
- monotonic deque for the finite linear-tail transformed scores;
- prefix max for fully capped predecessors.

Linear-tail transform:

```python
alpha = per_window / free_ms
transformed = Dprev[p] + alpha * Tprev[p]
candidate = transformed_max - alpha * Tcur + per_window
```

Check algebra against `_forward_transition_cost()` before coding the optimized form.

Fully capped:

```python
candidate = max(Dprev[p]) - cap
```

Respect `p < starts[t]` when admitting predecessors.

- [ ] **Step 13: Implement reverse candidates with a right-to-left deque**

For reverse:

```text
Tcur < Tprev <= Tcur + reverse_window_ms
p > ends[t]
```

Use:

```python
beta = reverse_penalty_max / reverse_window_ms
transformed = Dprev_no_inv[p] - beta * Tprev[p]
candidate = beta * Tcur + transformed_max
```

This transition always moves state:

```text
0 -> 1
```

- [ ] **Step 14: Handle disabled reverse defensively**

If:

```python
params.reverse_window_ms == 0
```

return no reverse candidates. Never divide by zero.

This is useful for D1 evaluator mode even though production P1a defaults to 3000 ms.

- [ ] **Step 15: Track two score states**

Working rows:

```python
prev = np.full((n_frames, 2), -np.inf, dtype=np.float64)
cur = np.full((n_frames, 2), -np.inf, dtype=np.float64)

prev[:, 0] = scores[0]
```

Recurrence:

```text
cur[:,0] = emission + forward(prev[:,0])

cur[:,1] = emission + max(
    forward(prev[:,1]),
    reverse(prev[:,0])
)
```

- [ ] **Step 16: Store parent metadata**

```python
parent_frame = np.full((n_events, n_frames, 2), -1, dtype=np.int64)
parent_state = np.full((n_events, n_frames, 2), -1, dtype=np.int8)
transition_kind = np.full((n_events, n_frames, 2), INVALID, dtype=np.int8)
transition_penalty = np.zeros((n_events, n_frames, 2), dtype=np.float64)
```

Transition kinds:

```python
START = 0
FORWARD_FREE = 1
FORWARD_PENALIZED = 2
LOCAL_REVERSE = 3
INVALID = 4
```

- [ ] **Step 17: Apply deterministic tie-breaking**

```python
_TIE_EPS = 1e-8
```

Decision:
1. greater score by more than epsilon;
2. forward over reverse;
3. lower transition penalty;
4. smaller predecessor canonical position.

Do not add a hidden alternate tie rule.

- [ ] **Step 18: Backtrack path and diagnostics**

Endpoint candidates span both terminal states:

```text
(event N-1, frame t, state 0)
(event N-1, frame t, state 1)
```

Sort by final score/tie rule, reconstruct paths, and wrap existing `DPPath`.

Diagnostics:

```python
emission_score = sum(scores[event, position])
forward_gap_penalty = ...
reverse_penalty = ...
total_transition_penalty = (
    forward_gap_penalty + reverse_penalty
)
```

Assert in tests:

```python
decoded.path.score == pytest.approx(
    decoded.diagnostics.emission_score
    - decoded.diagnostics.total_transition_penalty
)
```

- [ ] **Step 19: Preserve multiple-path behavior**

`paths` requests top endpoint alternatives.

When state/endpoints reconstruct duplicate exact frame-position tuples, deduplicate by:

```python
tuple(positions)
```

and continue until either:
- `paths` unique paths are collected; or
- candidates are exhausted.

- [ ] **Step 20: Implement `align_video_soft_order()` using optimized logic**

The public soft-order entry point now calls only the optimized implementation.

The quadratic implementation may remain private during development, but remove it before merge if the test oracle already provides the required reference and it adds maintenance duplication.

- [ ] **Step 21: Implement `rank_paths_soft_order()`**

Preserve current stratified ranking:

```python
depth = math.ceil(max_rows / len(videos))
```

Request `depth` paths per video, then consume level 0 across videos, level 1, and so on.

Sort within each level by:

```python
decoded.path.score
```

- [ ] **Step 22: Add randomized optimized-vs-oracle tests**

```python
@pytest.mark.parametrize("seed", range(25))
def test_optimized_decoder_matches_bruteforce_oracle(seed):
    rng = np.random.default_rng(seed)
    n_events = int(rng.integers(1, 7))
    n_frames = int(rng.integers(max(2, n_events), 21))

    increments = rng.integers(200, 2500, size=n_frames)
    timestamps = np.cumsum(increments).astype(np.float64)
    scores = rng.uniform(-0.2, 1.0, size=(n_events, n_frames))
    ...
```

Compare:
- best score;
- best frame positions/frame_idx;
- inversion-used flag.

Use:

```python
np.testing.assert_allclose(..., rtol=1e-9, atol=1e-9)
```

- [ ] **Step 23: Add explicit boundary tests**

Required:
- one event;
- exactly 8s;
- exactly -3s;
- -3001 ms;
- equal timestamps;
- one inversion then forward;
- attempted second inversion;
- negative emissions;
- all-zero emissions;
- irregular timestamps;
- cluster reuse;
- multiple paths.

- [ ] **Step 24: Run the decoder suite**

```bash
PYTHONPATH=src python -m pytest -q \
  tests/temporal/test_strict_regression.py \
  tests/temporal/test_soft_order_oracle.py \
  tests/temporal/test_soft_order.py
```

Expected: PASS.

- [ ] **Step 25: Commit**

```bash
git add src/hcmai/temporal/soft_order.py tests/temporal
git commit -m "feat(temporal): add competition-safe soft-order decoder"
```

---

# Task 5: Integrate Decoder Dispatch Without Changing Public Search Results

**Purpose:** Wire P1a into runtime behind `alignment_mode`.

**Files:**
- Modify: `src/hcmai/orchestration/workflows/temporal_search.py`
- Modify: `src/hcmai/temporal/__init__.py`
- Create: `tests/orchestration/test_soft_order_temporal_search.py`

**Produced behavior:**
- strict mode uses `rank_paths()`;
- P1a mode uses `rank_paths_soft_order()`;
- both materialize existing `AlignedPath`;
- `TemporalSearchResult` is unchanged.

- [ ] **Step 1: Write strict-dispatch test**

Use orchestration-level monkeypatching:

```python
def test_temporal_search_strict_mode_uses_existing_rank_paths(
    monkeypatch,
    service_factory,
):
    calls = {"strict": 0, "soft": 0}

    def fake_strict(*args, **kwargs):
        calls["strict"] += 1
        return []

    def fake_soft(*args, **kwargs):
        calls["soft"] += 1
        return []

    monkeypatch.setattr(
        "hcmai.orchestration.workflows.temporal_search.rank_paths",
        fake_strict,
    )
    monkeypatch.setattr(
        "hcmai.orchestration.workflows.temporal_search.rank_paths_soft_order",
        fake_soft,
    )

    service = service_factory(AlignmentConfig(alignment_mode="strict"))
    service.search(["event"], top_k=5)

    assert calls == {"strict": 1, "soft": 0}
```

Reuse existing test corpus/evidence fixtures where possible.

- [ ] **Step 2: Write P1a dispatch/config propagation test**

Capture `params` passed into `rank_paths_soft_order()` and assert:

```python
params.reverse_window_ms == 3000
params.forward_free_gap_ms == 8000
params.reverse_penalty_max == 0.15
params.forward_gap_penalty_per_window == 0.05
params.forward_gap_penalty_cap == 0.30
```

- [ ] **Step 3: Run RED**

```bash
PYTHONPATH=src python -m pytest -q \
  tests/orchestration/test_soft_order_temporal_search.py
```

Expected: missing import/dispatch.

- [ ] **Step 4: Export soft-order interfaces**

Modify `src/hcmai/temporal/__init__.py`:

```python
from .soft_order import (
    SoftOrderDecodedPath,
    SoftOrderDiagnostics,
    SoftOrderParams,
    align_video_soft_order,
    rank_paths_soft_order,
)
```

Keep all strict exports.

- [ ] **Step 5: Add dispatch at the current path-decoding boundary**

```python
if self.config.alignment_mode == "strict":
    rows = rank_paths(
        scores,
        lambda_gap=self.config.lambda_gap,
        max_rows=top_k,
        event_power=self.config.event_power,
        cluster_delta=self.config.cluster_delta,
    )
else:
    params = SoftOrderParams(
        reverse_window_ms=self.config.reverse_window_ms,
        forward_free_gap_ms=self.config.forward_free_gap_ms,
        reverse_penalty_max=self.config.reverse_penalty_max,
        forward_gap_penalty_per_window=self.config.forward_gap_penalty_per_window,
        forward_gap_penalty_cap=self.config.forward_gap_penalty_cap,
    )
    decoded = rank_paths_soft_order(
        scores,
        params=params,
        max_rows=top_k,
        event_power=self.config.event_power,
        cluster_delta=self.config.cluster_delta,
    )
    rows = [item.path for item in decoded]
```

- [ ] **Step 6: Verify path materialization stays unchanged**

Add an end-to-end test proving soft decoding still yields:

```python
TemporalSearchResult(
    paths=tuple[AlignedPath, ...],
    retrieval_ms=...,
    alignment_ms=...,
)
```

No frontend/API field changes.

- [ ] **Step 7: Run orchestration tests**

```bash
PYTHONPATH=src python -m pytest -q \
  tests/orchestration/test_soft_order_temporal_search.py \
  tests/orchestration/test_hybrid_temporal_search.py
```

Expected: PASS.

- [ ] **Step 8: Re-run strict tests**

```bash
PYTHONPATH=src python -m pytest -q tests/temporal/test_strict_regression.py
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add \
  src/hcmai/orchestration/workflows/temporal_search.py \
  src/hcmai/temporal/__init__.py \
  tests/orchestration/test_soft_order_temporal_search.py
git commit -m "feat(temporal): dispatch soft-order alignment by config"
```

---

# Task 6: Finalize Internal Soft-Order Diagnostics

**Purpose:** Make every P1a win/regression explainable without changing public APIs.

**Files:**
- Modify: `src/hcmai/temporal/soft_order.py`
- Extend: `tests/temporal/test_soft_order.py`

- [ ] **Step 1: Test score reconciliation**

```python
def test_soft_order_diagnostics_reconcile_with_path_score():
    decoded = align_video_soft_order(...)[0]
    d = decoded.diagnostics

    assert d.total_transition_penalty == pytest.approx(
        d.forward_gap_penalty + d.reverse_penalty
    )
    assert decoded.path.score == pytest.approx(
        d.emission_score - d.total_transition_penalty
    )
```

- [ ] **Step 2: Test inversion metadata**

For a 2-second reverse:

```python
assert d.used_local_inversion is True
assert d.inversion_event_index == 2
assert d.inversion_ms == 2000
```

For no reverse:

```python
assert d.used_local_inversion is False
assert d.inversion_event_index is None
assert d.inversion_ms is None
assert d.reverse_penalty == 0.0
```

- [ ] **Step 3: Test diagnostics for multiple returned paths**

Request `paths=3` and verify each wrapper has diagnostics consistent with its own path score.

- [ ] **Step 4: Run**

```bash
PYTHONPATH=src python -m pytest -q tests/temporal/test_soft_order.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hcmai/temporal/soft_order.py tests/temporal/test_soft_order.py
git commit -m "feat(temporal): expose internal soft-order diagnostics"
```

---

# Task 7: Add a Reproducible D0-D3 Decoder Evaluation Harness

**Purpose:** Isolate recurrence changes while freezing query preparation and P0 evidence.

**Files:**
- Create: `scripts/evaluate_temporal_p1a.py`
- Reuse: `tests/fixtures/l26_v254_query.yaml`
- Create: `tests/temporal/test_p1a_ablation_config.py`

## Evaluation modes

```text
D0 strict_legacy
   existing strict decoder
   existing lambda_gap behavior

D1 strict_new_gap
   reverse disabled
   new 8s free-zone + capped linear-tail forward cost

D2 soft_order_legacy_gap
   one <=3s inversion
   legacy forward lambda_gap semantics

D3 soft_order_p1a
   one <=3s inversion
   new P1a forward gap model
```

D1 and D2 are evaluator-only. They are not runtime `alignment_mode` values.

- [ ] **Step 1: Add an internal evaluation gap mode**

In `soft_order.py`, add an internal enum or literal:

```python
class ForwardGapMode(str, Enum):
    LEGACY_LINEAR = "legacy_linear"
    P1A_DEAD_ZONE = "p1a_dead_zone"
```

Internal decoder options additionally include:

```python
allow_local_inversion: bool
```

Production `align_video_soft_order()` always selects:

```text
P1A_DEAD_ZONE
allow_local_inversion=True
```

Evaluator helpers may select other combinations.

- [ ] **Step 2: Define D0-D3 run specs**

```python
@dataclass(frozen=True)
class P1aRun:
    name: str
    strict_decoder: bool
    allow_local_inversion: bool
    forward_gap_mode: str


RUNS = (
    P1aRun("D0_strict_legacy", True, False, "legacy_linear"),
    P1aRun("D1_strict_new_gap", False, False, "p1a_dead_zone"),
    P1aRun("D2_soft_order_legacy_gap", False, True, "legacy_linear"),
    P1aRun("D3_soft_order_p1a", False, True, "p1a_dead_zone"),
)
```

- [ ] **Step 3: Write run-definition tests**

Assert each run changes only the intended transition semantics.

- [ ] **Step 4: Freeze `VideoEventScores` once per query**

Preferred evaluator flow:

```python
scores = evidence.score_events(
    original_events,
    retrieval_events,
    caption_events=caption_events,
    use_dense=use_dense,
    use_bm25=use_bm25,
)

for run in RUNS:
    result = decode_only(run, scores)
```

Do not rescore P0 evidence independently for each D-run.

- [ ] **Step 5: Keep candidate order and depth identical**

All D0-D3 runs receive the same:
- `VideoEventScores` objects;
- candidate video order;
- `event_power`;
- `cluster_delta`;
- `top_k`.

- [ ] **Step 6: Record one JSONL object per run/query**

Required keys:

```json
{
  "run": "D3_soft_order_p1a",
  "query_id": "l26_v254",
  "target_video_id": "L26_V254",
  "target_rank": 4,
  "target_score": 2.93,
  "top_video_id": "L26_V254",
  "top_score": 3.04,
  "path_frame_idx": [300, 500, 475, 700],
  "path_timestamps_ms": [12000, 20000, 19000, 28000],
  "used_local_inversion": true,
  "inversion_event_pair": [1, 2],
  "inversion_ms": 1000,
  "emission_score": 3.10,
  "forward_gap_penalty": 0.02,
  "reverse_penalty": 0.05,
  "alignment_ms": 8.7
}
```

Document event-pair indexing as zero-based.

- [ ] **Step 7: Preserve the known L26_V254 diagnostic regions**

```yaml
target_video_id: L26_V254
target_regions:
  hold_two_X: [300, 475]
  plate_X: [500, 525]
  dialogue: [550, 950]
```

These ranges never affect scoring.

- [ ] **Step 8: Measure decoder-only latency**

```python
started = perf_counter()
decoded = decode(...)
alignment_ms = (perf_counter() - started) * 1000.0
```

- [ ] **Step 9: Print a compact summary**

```text
run                    target_rank   inversion   p50_ms   p95_ms
D0_strict_legacy       ...
D1_strict_new_gap      ...
D2_soft_order_legacy   ...
D3_soft_order_p1a      ...
```

The script must not auto-declare P1a successful from one query.

- [ ] **Step 10: Verify CLI import/help**

```bash
PYTHONPATH=src python scripts/evaluate_temporal_p1a.py --help
```

Expected: exit code 0.

- [ ] **Step 11: Commit**

```bash
git add \
  src/hcmai/temporal/soft_order.py \
  scripts/evaluate_temporal_p1a.py \
  tests/temporal/test_p1a_ablation_config.py
git commit -m "chore(temporal): add P1a decoder ablation harness"
```

---

# Task 8: Full Verification, Benchmark, and Rollout Gate

**Purpose:** Decide whether P1a is safe to enable. This task changes no algorithm semantics.

## 8A. Compile

- [ ] **Step 1: Compile source/tests/scripts**

```bash
python -m compileall -q src scripts tests
```

Required: exit code 0.

## 8B. Focused numerical tests

- [ ] **Step 2: Run temporal decoder tests**

```bash
PYTHONPATH=src python -m pytest -q \
  tests/temporal/test_strict_regression.py \
  tests/temporal/test_soft_order_oracle.py \
  tests/temporal/test_soft_order.py \
  tests/temporal/test_p1a_ablation_config.py
```

Required: 0 failures.

## 8C. Orchestration regression

- [ ] **Step 3: Run orchestration tests**

```bash
PYTHONPATH=src python -m pytest -q \
  tests/orchestration/test_soft_order_temporal_search.py \
  tests/orchestration/test_hybrid_temporal_search.py \
  tests/orchestration/test_temporal_evidence_setup.py
```

Required: 0 failures.

If the current environment lacks FAISS or another native project dependency, final verification must be rerun in the actual project environment. A stub is not sufficient for the merge/default-rollout claim.

## 8D. Full repository tests

- [ ] **Step 4: Run all tests**

```bash
PYTHONPATH=src python -m pytest -q
```

Required before merge/default rollout: 0 failures.

If a failure is claimed pre-existing, prove it by checking out/running the base revision.

## 8E. Strict rollback equivalence

- [ ] **Step 5: Run the frozen query set in strict mode before/after**

Compare:
- ranked video IDs;
- path `frame_idx`;
- strict alignment score.

Required:

```text
path identity exactly equal
scores numerically equal at existing strict precision
```

## 8F. L26_V254 decoder diagnostic

- [ ] **Step 6: Run D0-D3 on the known query**

Example:

```bash
PYTHONPATH=src python scripts/evaluate_temporal_p1a.py \
  --query-file tests/fixtures/l26_v254_query.yaml \
  --output artifacts/p1a_l26_v254.jsonl
```

Inspect:
- whether D2/D3 uses an inversion;
- event pair;
- inversion duration;
- chosen plate/hold/dialogue positions;
- D0→D2 difference;
- D0→D1 difference.

Do not change the 3000 ms window to force this case to win.

## 8G. Frozen organizer queries

- [ ] **Step 7: Run D0-D3 across the full prepared query set**

For labeled/manually understood examples record:
- target rank;
- Recall@1/5/20 when meaningful;
- top-K relevance;
- temporal path coherence.

For unlabeled queries manually inspect:
- largest D3 wins;
- largest D3 regressions;
- every case using a local inversion, or a representative sample if there are many.

## 8H. Latency gate

- [ ] **Step 8: Compare D0 vs D3 decoder latency**

Report:
- p50;
- p95;
- max.

Required:
- no production `O(F^2)` scaling;
- p95 compatible with competition interaction.

Use measured D0 latency and UI budget; do not invent a threshold without measurement.

## 8I. Rollout decision

- [ ] **Step 9: Enable P1a only if all conditions hold**

```text
[ ] complete test suite green
[ ] strict rollback equivalent
[ ] optimized decoder matches brute-force oracle
[ ] no NaN/Inf
[ ] no cross-video path
[ ] no more than one inversion/path
[ ] all inversions <=3000 ms
[ ] simple chronological queries do not systematically regress
[ ] aggregate D3 quality >= D0 on evaluated set
[ ] D3 wins use semantically plausible local inversion
[ ] decoder p95 latency is competition-safe
```

Until all are true:

```yaml
alignment:
  alignment_mode: strict
```

remains default.

After all are true, changing to:

```yaml
alignment:
  alignment_mode: soft_order_p1a
```

is a separate explicit rollout action.

---

# Implementation Notes

## Distinct frame versus equal timestamp

P1a forbids:
- same canonical position;
- any transition with `delta_ms == 0`.

Even if two distinct canonical frames have the same timestamp, they cannot be consecutive event matches.

## One inversion is one reverse transition

Allowed:

```text
E1 10s
E2 20s
E3 18s  <- one reverse
E4 24s
```

Not allowed:

```text
E1 10s
E2 20s
E3 19s  <- reverse #1
E4 18s  <- reverse #2
```

## Use float64 in the decoder

Match current strict DP:

```python
scores = np.asarray(video.scores, dtype=np.float64)
timestamps = np.asarray(video.timestamps_ms, dtype=np.float64)
```

## Do not couple D1/D2 to production config

Only runtime modes:

```text
strict
soft_order_p1a
```

D1/D2 are evaluator controls.

## Preserve current per-video ranking levels

Soft order changes path decoding, not result-diversity policy.

---

# Recommended Commit Sequence

```text
1. test(temporal): freeze strict alignment behavior
2. feat(temporal): add conservative soft-order config
3. test(temporal): define soft-order DP oracle
4. feat(temporal): add competition-safe soft-order decoder
5. feat(temporal): dispatch soft-order alignment by config
6. feat(temporal): expose internal soft-order diagnostics
7. chore(temporal): add P1a decoder ablation harness
8. docs(temporal): record P1a evaluation results
```

Keep the first seven commits separate during active competition development for easy bisection/rollback.

---

# Definition of Done

P1a is complete only when:

```text
[ ] strict decoder behavior is unchanged
[ ] strict remains default
[ ] P1a config is validated
[ ] brute-force oracle exists
[ ] optimized decoder matches oracle
[ ] event count N is dynamic
[ ] forward <=8s is free
[ ] forward >8s uses capped linear tail
[ ] reverse <=3s is allowed with linear penalty
[ ] reverse >3s is invalid
[ ] equal timestamp/same frame is invalid
[ ] at most one reverse transition exists per path
[ ] no event skip exists
[ ] score-cluster reuse is forbidden
[ ] video isolation is preserved
[ ] deterministic ties are tested
[ ] existing DPPath/AlignedPath/API contracts remain unchanged
[ ] D0-D3 evaluator freezes P0 score matrices
[ ] decoder latency is measured
[ ] strict rollback equivalence is measured
[ ] rollout decision uses aggregate evidence, not only L26_V254
```
