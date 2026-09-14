# VBS Scoped Temporal Feedback — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Quyền thực hiện trong cuộc hội thoại này:** Người dùng implement. Assistant chỉ lập kế hoạch, research và review. Dòng hướng dẫn agent phía trên dành cho người thực thi được người dùng giao sau này; không cấp quyền assistant tự chạy implementation hoặc spawn agent.

**Goal:** Cho phép người dùng xác nhận/bác bỏ bằng chứng theo event–khoảng, giải lại FF-MDP trong một video, giữ điều kiện đã nhập và hoàn tác được.

**Architecture:** Tách mask thuần số học khỏi decoder; một exploration branch giữ score matrix bất biến, điều kiện và lịch sử riêng. Dùng lại `VideoEventScores`, `DPPath`, `AlignedPath` và canonical materializer. Public API, AnswerWorkspace và DRES giữ theo plan frontend; kế hoạch này khóa callable nội bộ và tiêu chí bàn giao trước khi hai người tích hợp transport.

**Tech Stack:** Python hiện có của dự án, NumPy, dataclasses, pytest; FastAPI/Pydantic và React chỉ tại bước tích hợp với thành viên frontend. Không thêm model, database hoặc framework agent cho MVP.

**Spec:** `docs/superpowers/specs/2026-09-13-vbs-scoped-temporal-feedback-design.md` — đọc cùng plan này. Tài liệu tích hợp có thẩm quyền: `2026-09-12-vbs-2027-dres-answer-workspace.md` của thành viên frontend.

## Global Constraints

- Branch implementation bắt buộc: `vbs`.
- Assistant: Research, đọc/audit code, đối chiếu prior work, đề xuất và phản biện; không tự implementation.
- Một nhánh cục bộ đang hoạt động trong một video.
- Giữ kết quả toàn corpus để quay lại.
- Exploration revision độc lập với workspace revision.
- Không tự nới điều kiện khi giải không thành công.
- Score matrix gốc không bị feedback của một người dùng sửa tại chỗ.
- “Chưa thấy” khác “đã xác nhận không phù hợp”. Không suy negative video từ vài frame thiếu bằng chứng.
- Khoảng đóng integer milliseconds: `0 <= start_ms <= end_ms`; cho phép khoảng một thời điểm; không tự snap.
- `event_index` zero-based gắn decomposition version; không tự chuyển feedback sang danh sách event khác.
- Giữ mọi hàng event và full re-decode; không incremental DP, không true k-best, không entity-aware.
- Không thay schema submission, session, AnswerWorkspace; không tự submit và không tạo DRES client/logging cạnh tranh.
- CPU embeddings và hạ tầng hiện có được giữ theo quyết định của người dùng.
- Deadline làm việc của nhóm: trước 22/09/2026 cho VBS; không lấy hướng entity-aware dành cho SOICT.

---

## 1. Phạm vi bằng chứng và quyết định triển khai

Source đã đối chiếu là `src_hcmai_v20.zip` nguyên bản, được đọc ở `audit_source/`. Không dùng prototype assistant từng tự viết làm baseline. Các path `src/...` bên dưới là relative path trong repository của người dùng, không phải yêu cầu sửa thư mục audit.

Archive không cung cấp đầy đủ frontend, tests và project environment để xác nhận command test có chạy ngay. Các test file dưới đây là **file dự kiến tạo**, không phải test đã tồn tại/đã pass. Chạy trong checkout thật có dependencies. Không thêm một dependency mới chỉ vì môi trường audit thiếu nó.

**Quyết định để tránh scope creep:** tạo một backend core có thể test độc lập, sau đó tích hợp qua callable. Không tự đặt endpoint exploration hay sửa `SearchResponse` trong tài liệu này: spec đã giao public-contract ownership cho bước phối hợp với người phụ trách frontend. Phần core hoàn thành chưa có nghĩa tính năng đã usable trên UI hoặc sẵn sàng demo.

### File map

| File                                                          | Thao tác        | Trách nhiệm                                                                                    |
| ------------------------------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------ |
| `src/hcmai/temporal/constraints.py`                         | Tạo             | Khoảng, điều kiện bất biến, hợp khoảng, mask, phát hiện miền mâu thuẫn/thiếu frame |
| `src/hcmai/temporal/dp.py`                                  | Sửa tối thiểu | Thêm optional keyword-only`allowed`; giữ objective/recurrence cũ                            |
| `src/hcmai/orchestration/workflows/temporal_search.py`      | Sửa             | Tách scoring đã có thành method tái dùng; decode/materialize một video có mask          |
| `src/hcmai/orchestration/workflows/temporal_exploration.py` | Tạo             | Binding, một branch, revisions, lịch sử, score retention và response nội bộ                |
| `tests/temporal/test_constraints.py`                        | Tạo             | Semantics khoảng/mask và lỗi input                                                            |
| `tests/temporal/test_dp_constraints.py`                     | Tạo             | Oracle DP nhỏ, mask/power/clustering, parity, immutable score                                   |
| `tests/orchestration/test_temporal_scoring.py`              | Tạo             | Refactor scoring giữ input/flags, canonical validation                                          |
| `tests/orchestration/test_temporal_exploration.py`          | Tạo             | Feedback/undo/stale/scoring count và replay xuyên suốt                                        |
| `docs/vbs/scoped-feedback-handoff.md`                       | Tạo             | Callable mapping, interaction fixtures và ownership tích hợp                                  |
| `docs/vbs/scoped-feedback-evaluation.md`                    | Tạo             | Protocol B–C, measurements và claim limits                                                     |

**Không sửa trong core:** `api/contracts/search.py`, `api/routers/search.py`, `kis.py`, `planner.py`, `retrieval/evidence/hybrid.py`, frontend, database, DRES. Đây là quyết định giảm bề mặt merge; không khẳng định các file này sẽ không cần đụng ở bước tích hợp. Giữ nguyên planner, sử dụng snapshot `events` từ kết quả KIS đang hiển thị; feedback không chạy planner lần nữa.

### Dependency và commits

Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6 → Task 7. Mỗi task là một commit có thể review. Task 6 là phối hợp tích hợp; có thể chuẩn bị tài liệu trong khi chờ code frontend nhưng không đánh dấu UI done trước khi rehearsal.

Trước khi bắt đầu trong repo thật:

```bash
git status --short
git branch --show-current
rg --files src/hcmai tests
```

Nếu đã ở `vbs`, tiếp tục; nếu branch tồn tại, chuyển bằng `git switch vbs`; nếu chưa tồn tại, `git switch -c vbs`. Nếu working changes cản switch, giữ chúng và xử lý trong checkout riêng, không reset/discard. Không tạo branch tên khác cho implementation này. Chỉ stage file của task, không `git add .`.

## 2. Task 1 — Điều kiện event–khoảng và mask thuần

**Files:** Create `src/hcmai/temporal/constraints.py`; Test `tests/temporal/test_constraints.py`.

**Interfaces:**

```python
from dataclasses import dataclass
from typing import Literal
import numpy as np

Interval = tuple[int, int]

@dataclass(frozen=True, slots=True)
class Conditions:
    window: Interval
    confirmed: tuple[Interval | None, ...]
    rejected: tuple[tuple[Interval, ...], ...]

DomainStatus = Literal["ready", "contradictory_conditions", "no_indexed_frames"]

# All functions below live in temporal/constraints.py.
def validate_interval(interval: Interval) -> None: ...
def merge_intervals(intervals: tuple[Interval, ...]) -> tuple[Interval, ...]: ...
def remaining_intervals(domain: Interval, rejected: tuple[Interval, ...]) -> tuple[Interval, ...]: ...
def build_mask(timestamps_ms: np.ndarray, conditions: Conditions) -> tuple[np.ndarray, DomainStatus]: ...
```

Dấu `...` trong các block **Interfaces** chỉ là ký hiệu signature, không phải implementation cần copy. Các bước dưới xác định thuật toán và nội dung kiểm tra; không tạo abstract class/protocol cho bốn function này.

- [ ] **1.1 Viết test event-scoped trước.**

```python
import numpy as np
from hcmai.temporal.constraints import Conditions, build_mask


def test_rejection_does_not_leak_to_other_event():
    c = Conditions((0, 40), ((10, 20), None), ((), ((10, 20),)))
    mask, status = build_mask(np.array([0, 10, 20, 30, 40]), c)
    assert status == "ready"
    assert mask.tolist() == [
        [False, True, True, False, False],
        [True, False, False, True, True],
    ]


def test_unindexed_point_is_not_snapped():
    c = Conditions((0, 40), ((15, 15),), ((),))
    mask, status = build_mask(np.array([10, 20]), c)
    assert status == "no_indexed_frames"
    assert not mask.any()


def test_rejections_can_cover_confirmation_without_any_frames():
    c = Conditions((0, 40), ((10, 20),), (((10, 15), (16, 20)),))
    _, status = build_mask(np.array([0, 30]), c)
    assert status == "contradictory_conditions"
```

- [ ] **1.2 Chạy `uv run pytest tests/temporal/test_constraints.py -q`.** Lần đầu phải fail do module/function chưa có, không vì môi trường không import được `hcmai`. Nếu môi trường lỗi, sửa setup theo repo trước khi đọc kết quả là test cơ chế.
- [ ] **1.3 Implement validation và interval algebra.** Từ chối bool/float, đầu âm, đảo đầu/cuối; xác nhận cùng số hàng với rejected, E≥1. Kiểm tra timestamps một chiều, integer, không âm, nondecreasing. Không yêu cầu timestamp unique vì decoder baseline có semantics tăng cột.

```python
def validate_interval(interval):
    if len(interval) != 2 or any(type(x) is not int for x in interval):
        raise ValueError("interval requires two integer milliseconds")
    if not 0 <= interval[0] <= interval[1]:
        raise ValueError("invalid closed interval")


def merge_intervals(intervals):
    for interval in intervals:
        validate_interval(interval)
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return tuple(merged)


def remaining_intervals(domain, rejected):
    validate_interval(domain)
    cursor, end = domain
    pieces = []
    for lo, hi in merge_intervals(rejected):
        if hi < cursor:
            continue
        if lo > end:
            break
        if lo > cursor:
            pieces.append((cursor, lo - 1))
        cursor = max(cursor, hi + 1)
        if cursor > end:
            break
    if cursor <= end:
        pieces.append((cursor, end))
    return tuple(pieces)
```

- [ ] **1.4 Implement `build_mask`.** Khởi tạo `np.zeros((E,F), dtype=bool)`; với từng event lấy giao window và confirmed nếu có. Giao rỗng hoặc `remaining_intervals()` rỗng là contradiction chứng minh từ khoảng, độc lập sampling. Nếu còn miền nhưng không có timestamp nào trong miền thì thiếu frame. Ưu tiên contradiction nếu nhiều event có lỗi khác nhau; không kết luận contradiction chỉ vì không có frame.

```python
# Inner loop, after input validation; accumulate flags over all event rows.
lo, hi = conditions.window
if confirmed is not None:
    lo, hi = max(lo, confirmed[0]), min(hi, confirmed[1])
pieces = () if lo > hi else remaining_intervals((lo, hi), rejected)
if not pieces:
    contradictory = True
for a, b in pieces:
    mask[event] |= (timestamps_ms >= a) & (timestamps_ms <= b)
if pieces and not mask[event].any():
    missing = True
# At return:
status = "contradictory_conditions" if contradictory else (
    "no_indexed_frames" if missing else "ready"
)
```

- [ ] **1.5 Bổ sung parameterized invalid cases `(-1,0)`, `(2,1)`, `(True,2)`, `(1.0,2)` đều `ValueError`; confirmed ngoài window là contradiction.** Kiểm tra biên 10 và20 đều được nhận, rejected trùng lặp idempotent, empty frame array trả thiếu frame nếu miền còn.
- [ ] **1.6 Chạy focused test và commit.**

```bash
uv run pytest tests/temporal/test_constraints.py -q
git add src/hcmai/temporal/constraints.py tests/temporal/test_constraints.py
git commit -m "feat(temporal): define event-scoped interval constraints"
```

## 3. Task 2 — Mask vào FF-MDP, giữ nguyên baseline

**Files:** Modify `src/hcmai/temporal/dp.py`; Test `tests/temporal/test_dp_constraints.py`.

**Consumes:** bool `[E,F]` từ Task1.

**Produces:** giữ positional parameters `align_video(video, lambda_gap, paths, event_power, cluster_delta, min_separation_ms)` và thêm `*, allowed: np.ndarray | None = None`. Return vẫn `list[DPPath]`. `rank_paths()` không đổi.

- [ ] **2.1 Viết fixture và test lỗi correctness khó thấy.**

```python
import numpy as np
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import align_video


def make_video(scores):
    values = np.array(scores, dtype=float)
    n = values.shape[1]
    return VideoEventScores(
        video_id="v", frame_ids=np.array([f"f{i}" for i in range(n)]),
        frame_idx=np.arange(n), timestamps_ms=np.arange(n) * 1000,
        scores=values,
    )


def test_power_cannot_revive_forbidden_state():
    video = make_video([[0.9, 0.1, 0.2], [0.1, 0.8, 0.9]])
    original = video.scores.copy()
    allowed = np.array([[True, False, False], [False, True, False]])
    rows = align_video(video, lambda_gap=0, event_power=2, allowed=allowed)
    assert rows[0].frame_ids == ("f0", "f1")
    np.testing.assert_array_equal(video.scores, original)


def test_masks_do_not_create_new_score_clusters():
    video = make_video([[1, 1, 1], [1, 1, 1]])
    allowed = np.array([[True, False, False], [False, False, True]])
    assert align_video(video, cluster_delta=0.1, allowed=allowed) == []
```

- [ ] **2.2 Chạy `uv run pytest tests/temporal/test_dp_constraints.py -q`; xác nhận fail do keyword `allowed` chưa tồn tại.**
- [ ] **2.3 Thêm validation mask trước early returns, rồi chèn mask sau khi đã tính `starts` từ scores baseline.** Không sửa cached `video.scores`, không đưa `-inf` vào clustering.

```python
# Immediately after scores shape is known, before n_frames < n_events return.
if allowed is not None:
    if allowed.dtype != np.bool_ or allowed.shape != scores.shape:
        raise ValueError("allowed must be a boolean event-by-frame mask")

# Keep event_power transform and starts = cluster_starts(...) unchanged.
# After starts is computed, before current = scores[0]:
# NOTE: np.asarray may share memory with video.scores when event_power == 1.0.
# np.where always returns a new array, but an explicit copy defends against
# future in-place operations being added between here and the recurrence.
if allowed is not None:
    scores = np.where(allowed, scores.copy(), -np.inf)
```

- [ ] **2.4 Thêm oracle brute-force cho matrix nhỏ.** Kiểm tra optimum, không kiểm tra đúng một path khi tie có nhiều đáp án. Dùng `cluster_delta=0` cho oracle; test clustering riêng phía trên.

```python
from itertools import combinations
import pytest


@pytest.mark.parametrize("power", [1.0, 2.0])
def test_masked_score_matches_exhaustive_oracle(power):
    rng = np.random.default_rng(12)
    for _ in range(20):
        video = make_video(rng.uniform(-0.5, 1, size=(3, 6)))
        mask = rng.random((3, 6)) > 0.35
        scores = video.scores if power == 1 else np.clip(video.scores, 0, None) ** power
        gap = 0.0001
        feasible = []
        for path in combinations(range(6), 3):
            if all(mask[e, j] for e, j in enumerate(path)):
                value = sum(scores[e, j] for e, j in enumerate(path))
                value -= gap * (video.timestamps_ms[path[-1]] - video.timestamps_ms[path[0]])
                feasible.append(float(value))
        result = align_video(video, lambda_gap=gap, event_power=power, allowed=mask)
        if not feasible:
            assert result == []
        else:
            assert result[0].score == pytest.approx(max(feasible))
            assert all(mask[e, j] for e, j in enumerate(result[0].frame_idx))


def test_all_true_mask_preserves_baseline():
    video = make_video([[1, .2, .4, .3], [.1, .7, .8, .2]])
    expected = align_video(video, paths=3)
    assert align_video(video, paths=3, allowed=np.ones((2, 4), dtype=bool)) == expected
```

- [ ] **2.5 Kiểm tra E=1, F<E, all-false row và reversed feasible order đều trả đúng/empty theo baseline; shape/dtype mask sai phải `ValueError`.** So sánh unmasked output với baseline fixture đã ghi trước sửa; all-true parity không tự chứng minh refactor giữ bản cũ.
- [ ] **2.6 Chạy và commit.**

```bash
uv run pytest tests/temporal/test_dp_constraints.py tests/temporal/test_constraints.py -q
git add src/hcmai/temporal/dp.py tests/temporal/test_dp_constraints.py
git commit -m "feat(temporal): apply admissibility masks after score transforms"
```

## 4. Task 3 — Tách scoring và decode một video

**Files:** Modify `src/hcmai/orchestration/workflows/temporal_search.py`; Test `tests/orchestration/test_temporal_scoring.py`.

**Interfaces thêm vào `TemporalSearchService`:**

```python
def score_videos(
    self, original_events: Sequence[str], *,
    retrieval_events: Sequence[str] | None = None,
    caption_events: Sequence[str] | None = None,
    use_dense: bool = True, use_bm25: bool = False,
) -> tuple[tuple[VideoEventScores, ...], float]: ...

def decode_video(
    self, video: VideoEventScores, *, allowed: np.ndarray,
) -> tuple[AlignedPath, ...]: ...
```

- [ ] **3.1 Ghi test bảo toàn inputs bằng mock.** Dùng `object.__new__` cho test delegation để không giả lập một Corpus production; test canonical dùng `_validate_video_scores` riêng ở bước3.4.

```python
from types import SimpleNamespace
from unittest.mock import Mock
from hcmai.orchestration.workflows.temporal_search import TemporalSearchService


def test_score_videos_preserves_source_inputs():
    service = object.__new__(TemporalSearchService)
    service.max_temporal_event_count = 8
    service.config = SimpleNamespace(chunk_size=64)
    service.evidence = SimpleNamespace(score_events=Mock(return_value=[]))
    service._validate_video_scores = Mock()
    videos, elapsed = service.score_videos(
        ["mở tủ", "đặt cốc"], retrieval_events=["opens cabinet", "places cup"],
        caption_events=["mở tủ", "đặt cốc"], use_dense=True, use_bm25=True,
    )
    service.evidence.score_events.assert_called_once_with(
        ("mở tủ", "đặt cốc"), ("opens cabinet", "places cup"),
        caption_events=("mở tủ", "đặt cốc"), use_dense=True, use_bm25=True,
    )
    assert videos == ()
    assert elapsed >= 0
```

- [ ] **3.2 Run test RED. Tách normalization/scoring/identity validation hiện có của `search()` vào `score_videos()`.** Giữ fallback `score_event_videos(..., chunk_size=...)` đã có; không sửa fusion. Thêm checks số retrieval/caption events khớp originals và ít nhất một source enabled. Validation lỗi phải xảy ra trước gọi model. `search()` giữ top_k check và return contract cũ.

```python
# search(): replace duplicated scoring/validation stage with:
scores, retrieval_ms = self.score_videos(
    original_events, retrieval_events=retrieval_events,
    caption_events=caption_events, use_dense=use_dense, use_bm25=use_bm25,
)
score_by_video = {video.video_id: video for video in scores}
# Keep rank_paths(...) and materialization/timing of the existing search.
```

`score_videos()` trả tuple sau validation; `retrieval_ms` vẫn đo scoring theo baseline, không giả vờ toàn elapsed là cùng một metric. Giữ canonical validation ngoài đoạn timer nếu baseline đang làm vậy.

- [ ] **3.3 Implement decode dùng đúng cấu hình baseline, paths=1.**

```python
rows = align_video(
    video, lambda_gap=self.config.lambda_gap, paths=1,
    event_power=self.config.event_power,
    cluster_delta=self.config.cluster_delta,
    min_separation_ms=self.config.path_min_separation_ms,
    allowed=allowed,
)
return tuple(self._materialize_aligned_path(row, video) for row in rows)
```

- [ ] **3.4 Test canonical validation và regression search.** Fake corpus `.frame(frame_id)` trả bản ghi canonical bằng `SimpleNamespace`; dùng `VideoEventScores` 1event/2frames, cố tình đổi timestamp một frame rồi assert `ValueError` chứa `timestamp`. Test decode giữ `frame_ids`, `frame_idxs`, `timestamps_ms`; không reconstruct timestamp từ fps. Với source mock deterministic, chạy `search()` và so `paths` với output ghi trước refactor, bỏ timing ra khỏi equality. Kiểm tra `search()` vẫn raise `ValueError` cho `top_k <= 0` sau refactor — validation này phải nằm trước `score_videos()` call, không được mất khi tách method.
- [ ] **3.5 Chạy focused tests và suite temporal hiện có được tìm bằng `rg --files tests`; commit.**

```bash
uv run pytest tests/orchestration/test_temporal_scoring.py tests/temporal/test_dp_constraints.py -q
git add src/hcmai/orchestration/workflows/temporal_search.py tests/orchestration/test_temporal_scoring.py
git commit -m "refactor(temporal): separate scoring from single-video decoding"
```

**Giới hạn chi phí:** v20 không có API scoring selected-video bảo đảm tương đương normalization toàn corpus. Bản đầu cho phép `score_videos()` một lần khi mở nhánh, giữ matrix video được chọn, thả các video còn lại. Feedback/undo không gọi scorer. Không claim branch-open nhanh hoặc zero additional retrieval; nếu giữ được matrix từ lượt search bằng integration hook sẵn có thì dùng lại, nhưng không tạo full-corpus cache cho mọi phiên.

## 5. Task 4 — Branch có binding, revision và undo

**Files:** Create `src/hcmai/orchestration/workflows/temporal_exploration.py`; Test `tests/orchestration/test_temporal_exploration.py`.

**Consumes:** Task1 `Conditions`, `build_mask`, `merge_intervals`; Task3 `score_videos`, `decode_video`; canonical `AlignedPath`.

**Types nội bộ mới, không phải public Pydantic DTO:**

```python
from dataclasses import dataclass
from typing import Literal
from hcmai.temporal.constraints import Conditions, Interval
from hcmai.temporal.dp import AlignedPath

@dataclass(frozen=True, slots=True)
class QueryBinding:
    query: str
    event_version: str
    events: tuple[str, ...]
    retrieval_events: tuple[str, ...]
    caption_events: tuple[str, ...] | None
    use_dense: bool
    use_bm25: bool
    scoring_revision: str

@dataclass(frozen=True, slots=True)
class ExplorationView:
    revision: int
    event_version: str
    video_id: str
    conditions: Conditions
    status: str
    paths: tuple[AlignedPath, ...]
    changed_event_indices: tuple[int, ...]
    comparison_available: bool
    can_undo: bool

# Exact callable surface of TemporalExploration (same module).
# __init__(temporal: TemporalSearchService)
# open(binding: QueryBinding, video_id: str, window: Interval) -> ExplorationView
# apply(*, expected_revision: int, event_version: str,
#       scoring_revision: str, action: Literal["confirm", "reject", "window", "unknown"],
#       event_index: int | None = None, interval: Interval | None = None) -> ExplorationView
# undo(*, expected_revision: int, event_version: str, scoring_revision: str) -> ExplorationView
# close(*, expected_revision: int) -> None
# current() -> ExplorationView
```

Một instance phục vụ **một exploration session/tab**, không singleton toàn bộ service. `QueryBinding` đóng gói request scoring chứ không thay thế Event/Frame. `ExplorationView` là snapshot trình bày trạng thái, không sao chép nội dung `SearchResult`. `scoring_revision` là deployment/index/config generation do backend cấp, không tin client tự khẳng định. Decoder config giữ snapshot theo generation đó.

Exception nội bộ cùng module: `ExplorationConflict(ValueError)` cho revision/binding mismatch, `ExplorationUnavailable(RuntimeError)` cho dữ liệu/scoring hoặc session chưa mở. Input không hợp lệ dùng `ValueError`. Adapter map thành contract hiện có; không serialize trace/cause chứa hạ tầng.

**Quy ước mới được chọn ở mức implementation:** branch object sống theo session; thao tác có lock trong cùng process, stale request bị từ chối. Không hứa cross-worker state. Khi tích hợp chọn existing session store nếu có; nếu chỉ in-memory thì chạy một backend worker cho exploration, đóng/restart mất nhánh và phải mở lại. Đây là limitation deployment, không ảnh hưởng AnswerWorkspace persistent của teammate.

- [ ] **4.1 Viết test lifecycle với scorer fake.** Fixture `service` dùng real decoder từ Task3/corpus nhỏ hoặc fake deterministic trả `AlignedPath`; đầu tiên test số lần scorer và binding độc lập decoder correctness.

```python
# Fixture helper in this test file. No production test utility module.
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.orchestration.workflows.temporal_exploration import (
    TemporalExploration, QueryBinding, ExplorationConflict,
)


def new_branch():
    video = VideoEventScores("v", np.array(["a", "b", "c"]),
        np.array([1, 2, 3]), np.array([10, 20, 30]), np.ones((2, 3)))
    service = SimpleNamespace(
        score_videos=Mock(return_value=((video,), 1.0)),
        decode_video=Mock(return_value=()),
    )
    binding = QueryBinding("A rồi B", "q1", ("A", "B"), ("A", "B"), None,
                           True, False, "index-config-1")
    branch = TemporalExploration(service)
    branch.open(binding, "v", (0, 40))
    return branch, service, video


def test_feedback_and_undo_do_not_rescore():
    branch, service, video = new_branch()
    initial = branch.current()
    after = branch.apply(expected_revision=initial.revision, event_version="q1",
        scoring_revision="index-config-1", action="confirm", event_index=0, interval=(10, 20))
    restored = branch.undo(expected_revision=after.revision, event_version="q1",
                           scoring_revision="index-config-1")
    assert restored.conditions == initial.conditions
    assert restored.revision > after.revision
    assert service.score_videos.call_count == 1
    np.testing.assert_array_equal(video.scores, np.ones((2, 3)))
```

- [ ] **4.2 Run RED rồi implement `open`.** Từ chối open trên branch đang active; phải close trước. Validate binding và khoảng trước scoring. Select đúng video_id từ kết quả `score_videos`; không có video → `ExplorationUnavailable`, không coi là irrelevant. Copy riêng toàn bộ NumPy arrays của video chọn, đặt `writeable=False`, thả tuple toàn corpus khỏi branch. Ban đầu confirmed toàn None/rejected rỗng, window rõ ràng từ caller; revision bắt đầu1, history rỗng. Chỉ publish state sau khi acquisition thành công.

```python
from dataclasses import replace

def freeze_video(video):
    arrays = {}
    for name in ("frame_ids", "frame_idx", "timestamps_ms", "scores"):
        value = getattr(video, name).copy()
        value.setflags(write=False)
        arrays[name] = value
    return replace(video, **arrays)
```

`freeze_video` là private helper trong module này. Không `setflags` lên array của scorer dùng chung. Sau `open()`, assert `self._video.scores.flags.writeable is False` và tương tự cho `timestamps_ms`, `frame_idx`, `frame_ids` trong test lifecycle — đảm bảo mutation trên branch không thể vô tình sửa shared scorer data. QueryBinding phải có E≥1, nonblank normalized events, các event arrays cùng E, ít nhất một source; dùng normalize helper hiện có, không âm thầm đổi snapshot decomposition sau open.

- [ ] **4.3 Implement guard trước mọi mutation.**

```python
# Under the branch's lock, before touching history/mask/scoring:
if expected_revision != self._revision:
    raise ExplorationConflict("stale exploration revision")
if event_version != self._binding.event_version:
    raise ExplorationConflict("stale event version")
if scoring_revision != self._binding.scoring_revision:
    raise ExplorationConflict("stale scoring revision")
```

`self._binding`, `_video`, `_conditions`, `_history: list[Conditions]`, `_revision`, `_view`, `_lock` là private fields của `TemporalExploration`; initialize trong constructor. `_history` chỉ giữ conditions, không giữ copies score matrix mỗi lượt. Existing request/session layer giữ corpus result snapshot; branch không duplicate cả response toàn corpus.

- [ ] **4.4 Implement mutation bằng immutable `replace`.** Confirm/reject/unknown cần event_index hợp lệ; window bắt buộc event_index=None. Confirm/reject/window bắt buộc interval hợp lệ; unknown bắt buộc interval=None. Không chấp nhận tham số thừa rồi silently ignore.

```python
# confirm action:
confirmed = list(old.confirmed)
confirmed[event_index] = interval
new = replace(old, confirmed=tuple(confirmed))

# reject action:
rejected = list(old.rejected)
rejected[event_index] = merge_intervals(rejected[event_index] + (interval,))
new = replace(old, rejected=tuple(rejected))

# window action:
new = replace(old, window=interval)

# unknown: return current view; no history push, revision or decoder call.
```

Nếu `new == old`, trả current, không tăng revision. Với mutation có thay đổi: tạo mask/decode dự kiến trước; khi thành công hoặc là semantic failure (`contradictory_conditions`, `no_indexed_frames`, `no_valid_path`) thì push old conditions, commit new conditions/view, revision+1. **Semantic failure giữ điều kiện lỗi để undo/sửa; lỗi hạ tầng không commit mutation**, giữ trạng thái trước và raise `ExplorationUnavailable`. Không blanket-catch programming bugs như `AttributeError` thành “video không phù hợp”.

- [ ] **4.5 Implement undo/close.** Undo dùng guard, history rỗng là no-op; lấy checkpoint cuối, decode rồi commit, pop sau khi decode không gặp lỗi hạ tầng; revision tăng, không quay lại số revision cũ. Close kiểm revision, giải phóng matrix/history và đánh dấu inactive; `current()` sau close raise unavailable. UI quay lại global snapshot nó đang giữ, không gọi retriever chỉ để phục hồi màn hình.
- [ ] **4.6 Thêm tests về stale, reconfirm, unknown, reject union và isolation.**

```python
import pytest


def test_stale_event_feedback_does_not_mutate():
    branch, _, _ = new_branch()
    before = branch.current()
    with pytest.raises(ExplorationConflict, match="event"):
        branch.apply(expected_revision=before.revision, event_version="q2",
            scoring_revision="index-config-1", action="reject", event_index=0, interval=(10, 20))
    assert branch.current() == before


def test_unknown_is_not_negative_feedback():
    branch, service, _ = new_branch()
    before = branch.current()
    count = service.decode_video.call_count
    after = branch.apply(expected_revision=before.revision, event_version="q1",
        scoring_revision="index-config-1", action="unknown", event_index=1)
    assert after == before
    assert service.decode_video.call_count == count
```

Test hai branch dùng cùng source matrix: apply branch1 không thay conditions/path branch2. Test revision cũ cho apply/undo bị từ chối mà scorer không chạy. Reconfirm `(10,10)`→`(20,20)` rồi undo phải phục hồi `(10,10)`. Thay event hoặc scoring generation yêu cầu nhánh mới; lưu snapshot nhánh cũ trong session history để đối chiếu trước close, không remap index.

- [ ] **4.7 Chạy và commit.**

```bash
uv run pytest tests/orchestration/test_temporal_exploration.py -q
git add src/hcmai/orchestration/workflows/temporal_exploration.py tests/orchestration/test_temporal_exploration.py
git commit -m "feat(exploration): retain scoped feedback with revisioned undo"
```

## 6. Task 5 — Trạng thái, diff và replay xuyên suốt

**Files:** Modify `temporal_exploration.py` và test file Task4.

**Produces:** `ExplorationView.status` thuộc `ok`, `contradictory_conditions`, `no_indexed_frames`, `no_valid_path`. Binding/infra exceptions theo Task4, không trộn với verdict nội dung. `changed_event_indices` so canonical `(frame_id, timestamp_ms)`, không so score.

- [ ] **5.1 Implement helper `_evaluate(conditions: Conditions) -> tuple[str, tuple[AlignedPath, ...]]` trong class.**

```python
allowed, status = build_mask(self._video.timestamps_ms, conditions)
if status != "ready":
    return status, ()
paths = self._temporal.decode_video(self._video, allowed=allowed)
return ("ok", paths) if paths else ("no_valid_path", ())
```

Không gắn `video_valid=false` vào bất kỳ trạng thái nào. “Không path” có thể do strict order hoặc clustering. Tất cả rows vẫn ở decoder, dù chỉ sửa một event.

- [ ] **5.2 Implement diff giữa view trước và path mới.**

```python
comparison_available = bool(previous.paths and paths)
changed = ()
if comparison_available:
    before, after = previous.paths[0], paths[0]
    changed = tuple(i for i, pair in enumerate(zip(
        zip(before.frame_ids, before.timestamps_ms, strict=True),
        zip(after.frame_ids, after.timestamps_ms, strict=True), strict=True
    )) if pair[0] != pair[1])
```

Lượt open hoặc chuyển từ/to empty path: comparison_available=False, không báo “không có gì thay đổi” từ tuple rỗng. Conditions luôn trả riêng; frame của event đã confirm có thể đổi bên trong khoảng.

- [ ] **5.3 Replay bằng real decoder và fake canonical corpus nhỏ.** Dữ liệu 3events/5frames timestamps `[10,20,30,40,50]`; scores phía dưới, lambda_gap=0, event_power=1, cluster_delta=0. Dùng `TemporalSearchService` thật với scorer fake và corpus `.frame` map đủ fields `frame_id`, `video_id`, `frame_idx`, `timestamp_ms`.

```python
scores = np.array([
    [9, 8, 0, 0, 0],
    [0, 9, 8, 0, 0],
    [0, 0, 0, 9, 8],
], dtype=float)
# Expected trace, frame indices are zero-based in this fixture:
# open [0,60]                    -> (0,1,3)
# confirm event0 [10,20]          -> (0,1,3)
# reject event2 [40,40]           -> (0,1,4)
# undo                           -> (0,1,3)
# confirm event0 [20,20]          -> (1,2,3)
# window [0,15]                  -> contradictory_conditions
# undo                           -> (1,2,3)
```

Assert each expected tuple, unchanged score matrix, scorer count1, revisions tăng từng mutation thực và can_undo đúng. Đây là functional evidence, không dùng trace tự tạo để claim retrieval accuracy.

- [ ] **5.4 Phân biệt hai lỗi không phải contradiction.** Một confirmation point25 khi timestamps chỉ10/20/30 → thiếu frame. Confirm event0=30 và event1=10, cả hai miền có frame → no_valid_path dưới strict order; không bắt buộc xây prover về thứ tự để đổi nhãn contradiction.
- [ ] **5.5 Inject materializer unavailable và request race.** Infrastructure exception đã biết phải được map `ExplorationUnavailable` tại adapter/service boundary, giữ revision/history cũ. Hai request chung expected_revision, lock cho phép nhiều nhất một mutation có hiệu lực; request thứ hai stale. Không cần đưa LLM hay GPU vào các tests này.
- [ ] **5.6 Chạy một lượt focused suite đủ cơ chế và commit.**

```bash
uv run pytest tests/temporal/test_constraints.py tests/temporal/test_dp_constraints.py tests/orchestration/test_temporal_scoring.py tests/orchestration/test_temporal_exploration.py -q
git add src/hcmai/orchestration/workflows/temporal_exploration.py tests/orchestration/test_temporal_exploration.py
git commit -m "test(exploration): cover constrained replay and failure semantics"
```

## 7. Task 6 — Bàn giao tích hợp với frontend/DRES owner

**Files:** Create `docs/vbs/scoped-feedback-handoff.md`. Các sửa đổi transport/UI thuộc change của người phụ trách frontend, áp trên checkout mới nhất của họ. Không invent đường dẫn component chưa có trong archive.

**Consumes:** callable Task4; `SearchResponse.events/dense_events/bm25_caption_events/use_dense/use_bm25`; canonical `AlignedPath`.

**Produces:** mapping reviewable dưới đây, fixture replay Task5 và một walkthrough chạy được qua UI sau integration. Task này chỉ hoàn tất khi hai phía nối xong; backend core vẫn có thể merge trước trong `vbs`.

- [ ] **6.1 Ghi bảng mapping vào tài liệu handoff.**

| Luồng                      | Backend input/output                                                                                        | Owner và điều kiện                                                        |
| --------------------------- | ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Mở kết quả video         | Snapshot events và nguồn scoring →`QueryBinding`; selected `video_id`, explicit window → `open()` | Backend cấp event/scoring generation; frontend giữ original search response |
| Confirm/reject              | `apply()` gồm expected exploration revision + event version + event index + interval                     | Controls của frontend chỉ gọi khi user chủ động                         |
| Tìm quanh mốc             | `apply(action="window", event_index=None, interval=...)`                                                  | UI hiển thị bounds; không tự confirm                                      |
| Chưa rõ                   | `apply(action="unknown", event_index=...)` hoặc UI không mutate                                         | Không negative; nếu ghi interaction thì nhãn riêng                       |
| Undo                        | `undo()` với current exploration binding/revision                                                        | Không dùng AnswerWorkspace revision                                         |
| Refresh query/decomposition | Lưu view cũ để đối chiếu; close rồi open binding mới                                               | Không tự remap feedback                                                     |
| Return global               | `close()` và restore snapshot đã giữ                                                                  | Không xóa global results khi mở branch                                     |
| Chọn đáp án             | Canonical result/inspector → existing candidate dialog                                                     | Không gọi workspace mutation từ core                                       |
| Result logging              | successful materialized result → logger chung của teammate                                                | Không thêm client DRES hoặc double log                                     |

- [ ] **6.2 Rà transport trên checkout frontend mới nhất bằng các lệnh sau rồi ghi mapping cụ thể vào PR tích hợp của owner.**

```bash
rg -n 'create_search_router|SearchRequest|SearchResponse|X-VBS-User-ID' src/hcmai
rg -n 'AnswerCandidateDialog|onAddCandidate|workspaceAction|X-DRES-Log-Status' frontend/src
rg -n 'revision|session|service_container' src/hcmai/api src/hcmai/orchestration
```

Đây là dependency thật, không phải câu hỏi thiết kế đang chặn core. Plan của teammate hiện mô tả DRES/AnswerWorkspace, chưa có exploration transport. **Không tuyên bố endpoint đã tồn tại.** Public additive contract của exploration được review ở PR tích hợp; không nhét feedback vào `retrieval_events` hoặc reuse `/trake` để lách schema. Existing `/api/v1/search` giữ backward compatibility. Không có transport thì chưa claim UI feature done.

- [ ] **6.3 Thống nhất session lifetime thực thi.** Một tab có exploration handle riêng, lifecycle remove on close/disconnect/expiry theo infrastructure hiện có; identity không phụ thuộc DRES đã connect. Giới hạn một matrix/video/active branch; impose bound active sessions theo capacity server và timeout hiện có, không tạo unbounded registry. Không gọi đó là cache throughput đã tối ưu. Nếu multi-worker đang chạy, dùng session store/routing hiện có hoặc chạy exploration một worker; không triển khai in-memory multi-worker rồi bỏ qua mất trạng thái.
- [ ] **6.4 Gửi fixture input/output của replay Task5 cho frontend trong PR/tài liệu, không nhắn người khác tự động.** Dùng `dataclasses.asdict(view)` cho fixture nội bộ; production serializer dùng canonical schema đã thống nhất. Không tự phát hành đây là JSON API v1. `scoring_revision` backend-owned, client chỉ echo để phát hiện stale.
- [ ] **6.5 Rehearsal controls và timestamp với teammate.**

| Ca                                                    | Expected                                                                                               |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Response revision4 tới sau revision5                 | UI vẫn giữ5; so trong cùng branch handle, không chỉ integer                                       |
| A confirmed`[80000,85000]`, frame đổi81000→83000 | Badge condition giữ; diff frame báo đổi                                                            |
| B rejected một khoảng                               | A/C vẫn dùng được khoảng đó                                                                    |
| Nhánh failure → undo                                | Điều kiện/path hồi phục; global list còn                                                         |
| Inspector currentTime=12.3456                         | Existing dialog nhận12346ms; không snap                                                              |
| Card timestamp_ms=12000                               | Dialog nhận12000ms; không suy từ frame_idx                                                          |
| Chọn đáp án từ exploration                       | Chỉ mở editable dialog; Enter lưu candidate; submit riêng                                          |
| DRES/logging lỗi                                     | Retrieval/branch còn, log status theo plan teammate                                                   |
| Hai tabs                                              | Không nhìn thấy/ghi đè feedback của nhau trừ khi có feature chia sẻ được thiết kế riêng |

- [ ] **6.6 Giữ regression boundary.** Teammate chạy test contracts/UI/DRES đã có trong plan của họ; backend chạy focused suite Task5 sau merge. Không tái tạo tests cho toàn DRES trong plan này. Stage tài liệu handoff và commit `docs(vbs): define scoped feedback integration handoff`.

## 8. Task 7 — Đánh giá đủ để viết claim và đóng scope

**Files:** Create `docs/vbs/scoped-feedback-evaluation.md`. Kết quả người dùng chạy được lưu kèm benchmark metadata trong thư mục experiments hiện hành của repo; không sửa artifact benchmark cũ.

**Goal:** Đo giá trị feedback tích lũy, không trì hoãn triển khai để tìm model hay sweep thêm baseline.

- [ ] **7.1 Ghi protocol trước khi nhìn kết quả C.** A=retriever+player+context; B=cùng local search với tất cả controls/khả năng diễn đạt constraints như C nhưng không tự giữ bộ điều kiện giữa lượt, người dùng nhập lại; C=giữ feedback+undo. Primary B–C; nếu thiếu thời gian, A chỉ làm contextual baseline, ưu tiên B–C công bằng. B không bị khóa chỉ một constraint để tạo lợi thế giả cho C.
- [ ] **7.2 Khóa dataset/query split và task semantics.** Giữ metadata n=19KIS/9QA/2TRAKE của baseline hiện có. Chọn KIS cho claim chính, có cả thành công/failure ngoài bốn top1 đúngvideo/saimoment. Query đã biết đáp án đánh dấu development; không tính như unseen human task. Tự tạo query temporal chỉ ghi synthetic. Không gọi TRAKE là task VBS, không suy HCMAI accuracy thành V3C accuracy.
- [ ] **7.3 Lập run table cụ thể.**

```text
run_id, participant_id, query_id, condition, order_group,
seen_before, dataset_revision, scoring_revision, time_budget_ms,
success, completion_ms, stop_reason,
condition_reentries, confirmed_evidence_revisits,
retrieval_calls, scoring_ms, decode_ms, interaction_count
```

Budget human session được chọn một lần trước thu thập và giữ bằng nhau B/C; điền con số thực tế trong protocol run, không trích số luật VBS chưa xác minh. Completion timer tính cả waiting và thao tác. Failure giữ trong denominator; báo completion rate và times với quy tắc censor tại budget, không chỉ mean success cases.

Operational definitions: `condition_reentries` là nhập lại cùng event–interval–polarity đã biểu đạt trước trong task; `confirmed_evidence_revisits` là mở lại interval đã confirm để kiểm tra lại, tính theo explicit revisit event trong log, không suy mọi autoplay frame là revisit. Ghi definition thống nhất cho B/C; nếu logger không đo được đáng tin, bỏ claim metric đó thay vì đoán từ click tổng.

- [ ] **7.4 Phân bổ task giữa conditions để giảm nhớ đáp án.** Một người không làm cùng target đã biết lần lượt B rồi C để coi đó là paired unbiased effect. Dùng nhóm query khác nhau có mức khó tương đương, counterbalance across participants nếu có. **Proxy mức khó:** dùng retrieval difficulty từ baseline benchmark — cụ thể top-1 video rank (có/không hit) và path score gap giữa best path và runner-up. Phân mỗi condition nhận query từ cả nhóm easy (top-1 hit + moment-accurate) và hard (top-1 hit nhưng moment sai hoặc miss video). Ghi proxy dùng trong protocol trước khi nhìn kết quả C. Nếu chỉ có developers, báo rõ participants/developer-only và sample size, gọi exploratory study, không tuyên bố significance/general usability.
- [ ] **7.5 Kiểm tra kỹ thuật nhỏ trước human session.** Replay Task5 pass; cached mutation scorer count0; report branch-open riêng với warm feedback latency. Một lượt benchmark gốc trước/sau để kiểm tra regression khi no feedback; chỉ lặp lại nếu thấy risk cụ thể. Không sweep model hoặc dùng A6000 để che lỗi semantics.
- [ ] **7.6 Viết claim theo kết quả thực tế.**

| Bằng chứng có được               | Claim phù hợp                                                                    |
| -------------------------------------- | ---------------------------------------------------------------------------------- |
| Chỉ functional tests và demo         | Hệ thống hỗ trợ cơ chế này; chưa chứng minh giảm công sức              |
| Human B–C nhỏ, kết quả thuận lợi | Exploratory evidence trong sample/setting đã nêu                                |
| C chỉ thắng A                        | Whole workflow benefit; chưa isolate tích lũy feedback                          |
| B–C không cải thiện                | Báo giới hạn; nhấn controllability/undo và phân tích lỗi, không bịa gain |
| Chỉ simulated feedback                | Decoder/constraint effectiveness dưới policy mô phỏng; không user study       |

Căn cứ prior research và số liệu benchmark giữ theo spec mục3–4; chưa claim first hoặc prior systems không có cơ chế tương tự. Không thêm một vòng literature survey vào critical path trừ khi cần xác minh câu cụ thể trong paper.

- [ ] **7.7 Đóng paper/demo package.** Screenshot có event, interval constraints, revised path, undo và candidate handoff; một diagram pipeline; một bảng benchmark với n; một bảng B–C nếu đã thu; section limitations (small data, developer participants, sampling, local branch, no novelty proof). Không đưa agent/multiagent vào title/contribution khi chưa implement/evaluate.
- [ ] **7.8 Commit `docs(vbs): specify feedback evaluation and claim boundaries`.**

## 9. Lịch đề xuất và thứ tự cắt scope

Đây là allocation để người dùng thực hiện từ13/09, không phải ước lượng đã đo hay yêu cầu chờ đến ngày đó mới làm bước tiếp.

| Ngày     | Deliverable                                               |
| --------- | --------------------------------------------------------- |
| 13–14/09 | Tasks1–3: mask, DP correctness, scoring split            |
| 15–16/09 | Tasks4–5: branch state, replay, cache isolation          |
| 17/09     | Task6: nối controls/frontend contract, candidate handoff |
| 18–19/09 | Task7: run nhỏ B–C, screenshot, failure analysis        |
| 20/09     | Chốt figures và draft contribution/limitations          |
| 21/09     | Rà paper/package và submit theo mốc nhóm trước22/09 |

Nếu trễ: bỏ agent, bỏ UI diff animation, bỏ extra alternatives, bỏ A study nếu B–C còn đủ, giữ một branch và best path. **Không cắt** correctness mask, revisions/undo, preservation of canonical timestamp, hoặc no-feedback regression. Không đổi retriever/model sát deadline nếu chưa có lỗi bắt buộc.

## 10. Definition of Done và self-review

- [ ] Người dùng implement trên `vbs`, core tests pass với log thực tế.
- [ ] No feedback giữ baseline; allowed mask không bị event_power/clustering phá.
- [ ] Confirm/reject có event scope; reconfirm/undo đúng; unknown không negative.
- [ ] Branch giữ score bất biến một video; mutation không rescore; branch-open cost báo riêng.
- [ ] Event/scoring/revision mismatch không mutate; session/tab isolation và deployment worker semantics được xác nhận.
- [ ] Phân biệt contradiction/thiếu frame/no path/infra; không auto-relax và không kết luận video sai.
- [ ] Full replay chạy được qua UI, global result phục hồi, stale response không overwrite.
- [ ] Candidate dialog giữ timestamp chuẩn và submission flow teammate; logging dùng chung.
- [ ] Protocol/measurements và screenshot đủ support claim; không đánh dấu nghiên cứu thành công chỉ vì tests pass.

**Self-review khi lập plan:** đã đối chiếu spec mục5–15 với Tasks1–7; agent/model là ngoài MVP theo spec, không gap. Public exploration transport là dependency phối hợp chưa hiện hữu trong source/plan frontend, được ghi rõ ở Task6 thay vì bịa schema. Các type mới chỉ đóng gói constraints, scoring binding và view nội bộ; không tạo Frame/Event/AnswerWorkspace thứ hai. Lệnh test/commit là hướng dẫn cho người dùng, chưa được chạy cho implementation. Không có quyền tự chuyển từ plan sang coding trong cuộc hội thoại này.
