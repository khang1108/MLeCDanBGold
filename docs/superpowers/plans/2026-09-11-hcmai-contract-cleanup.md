HCMAI Contract Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Không tự dispatch agent. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thống nhất ownership, naming, schema, API và output trong query-to-path flow, không làm phình abstraction hoặc làm lệch baseline mà không ghi nhận.

**Architecture:** Một owner cho mỗi semantic operation; giữ riêng validation theo trust boundary và projection theo consumer. Cleanup production trước, sau đó hai thành viên phát triển research qua contract đã chốt; không trộn model/verifier mới vào PR cleanup.

**Tech Stack:** Python, Pydantic, frozen dataclasses, NumPy, FastAPI và pytest hiện có. Không thêm runtime dependency để cleanup.

**Spec:** `HCMAI_Research_Schema_API_Contract_v1.md` — spec gốc được viết trong scratch session ngoài repo. **TODO: chuyển spec vào `docs/specs/` trước khi bắt đầu Task 0.** Nếu spec gốc không khôi phục được, tái tạo từ ownership map (§2), naming table (§3) và coverage matrix (§5) trong plan này — đó là nguồn authority thay thế. Không tạo bản spec nội dung khác.

## Global Constraints

- Chỉ là kế hoạch; chưa sửa code hoặc chạy test/benchmark trong lượt lập plan.
- Giữ `QueryCandidateSet`, `VideoEventScores`, `DPPath`, `AlignedPath`, `TemporalSearchResult` hiện có.
- Không thêm public HTTP endpoint, không sửa payload của team interactive.
- Không thêm plan registry, generic constraint engine, universal DTO hoặc provider factory.
- Không đổi embedding, indexes, fusion objective, DP recurrence, gap parameters hoặc 5 method identifiers trong cleanup.
- Canonical `frame_id`, `video_id`, competition `frame_idx`, timestamp milliseconds phải giữ nguyên.
- Bugfix planner/error handling là thay đổi hành vi có chủ đích, không được gộp vào claim “behavior-preserving refactor”.
- Không xóa symbol dựa trên text search đơn thuần; phải kiểm tra exports, fixtures, dynamic dispatch và callers ngoài runtime.

## 1. Phạm vi và source authority

Nguồn chính: **live checkout tại `src/hcmai/`** trong repository hiện tại. Plan ban đầu tham chiếu `src_hcmai_v19.zip` (SHA-256 `79b36b77b066dbe155ff313cb2234956441e1d0c72fb4f187df9cd3d4da37d45`) từ scratch session — zip đó không có trong repo và không cần thiết vì code đã được tích hợp vào working tree. Nếu cần đối chiếu, dùng `git log` trên các file trong scope.

Đường dẫn task dưới đây tương đối với repository root: `src/hcmai/...`, `tests/...`, `baseline/...`. Đây là live checkout chạy được; Task 0 xác nhận test collection và ghi baseline trước khi edit.

Cleanup giới hạn ở query preparation, temporal orchestration, corpus evidence access và output projection liên quan. Không refactor ingestion, frontend, history, deployment hay toàn bộ legacy tests.

## 2. Ownership map — một chức năng, một owner

| Chức năng                       | Owner cuối cùng                                                        | Consumer                                                          | Không được làm ở đâu                                     |
| --------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------- | ---------------------------------------------------------------- |
| Normalize explicit event strings  | `temporal/planner.py:normalize_event_texts`                            | Query prep, temporal search, research plan validator              | Mỗi module tự join/strip/drop empty khác nhau                 |
| Raw KIS → ordered events         | `temporal/planner.py:plan_query_events`                                | KIS workflow; query-candidates raw branch                         | Router, SLM adapter hoặc scorer                                 |
| Sentence/line splitting primitive | `split_query_events` giữ hiện trạng cho callers đã xác nhận     | Planner; các caller thực sự cần raw split                     | Không dùng như semantic planner thay thế                     |
| Validate LLM bundle fidelity      | `query_preparation/service.py`                                         | Query generation                                                  | HTTP router hoặc temporal DP                                    |
| Select original/retrieval text    | KIS/TRAKE workflow ở ingress; research runner cho research              | Temporal service                                                  | Dense không tự rewrite; BM25 không nhận rewrite thay literal |
| Score/fuse evidence               | `retrieval/evidence/hybrid.py` và components hiện có                | Temporal service, baseline runner                                 | QueryPlan/verifier/materializer                                  |
| Numeric path decode               | `temporal/dp.py`                                                       | Temporal service, baseline methods nếu thực sự cùng semantics | Workflow tự viết recurrence                                    |
| Canonical metadata lookup         | `Corpus`                                                               | Temporal service, materializer, collector                         | DTO hoặc model adapter                                          |
| Canonical path validation         | `SearchMaterializer.validate_aligned_path`                             | Temporal materialization, KIS/TRAKE projection                    | Copy identity check ở từng output builder                      |
| KIS representative selection      | `SearchMaterializer.build_kis_result`                                  | KIS                                                               | Baseline/research copy midpoint rule                             |
| TRAKE HTTP projection             | `SearchMaterializer.build_trake_path` static method                    | TRAKE workflow                                                    | Verifier hoặc DP                                                |
| Research verification             | `research/path_verification.py`, theo spec, chưa implement ở cleanup | Research runner                                                   | Existing image reranker                                          |
| Evaluator output envelope         | Existing baseline runner/output owner                                    | Benchmark, research integration sau                               | Mỗi experiment viết một JSON format riêng                    |

Không gộp HTTP shape validation với canonical validation: HTTP bảo vệ client input; canonical check bảo vệ identity với corpus. Hai lớp có thể kiểm tra cùng độ dài nhưng khác trust boundary, không mặc định là duplicate.

## 3. Naming và units

| Khái niệm                               | Tên chốt                                                         | Compatibility                                                                   |
| ----------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| Văn bản gốc                            | `original_events` internal; `events` ở HTTP hiện có         | Không rename wire field                                                        |
| Văn bản dense retrieval                 | `retrieval_events` internal; `dense_events` response hiện có | Mapping chỉ ở boundary                                                        |
| BM25 caption-field query                  | `caption_events` existing internal                               | Không nhầm với corpus captions hoặc query riêng cho context dense          |
| Thời gian                                | `timestamp_ms`, `start_ms`, `end_ms`, `*_ms`               | ASR interval`[start,end)`                                                     |
| Vị trí trong NumPy matrix               | `position`/`positions`                                         | Không đặt tên frame_idx                                                     |
| Event reference trong fixed-boundary plan | `event_index`/`event_indices`, zero-based                      | Không thêm event_id trùng chức năng                                        |
| Một frame competition coordinate         | `frame_idx`                                                      | Giữ giá trị canonical                                                        |
| Path arrays                               | Existing`AlignedPath.frame_idxs`                                 | `DPPath.frame_idx` là existing exception, giữ để tránh rename-only churn |
| Decoder score                             | Existing`AlignedPath.score`                                      | Research export gọi`decoder_score`, không overwrite bằng verifier score    |
| Final retrieval rows                      | `top_k` ở HTTP                                                  | Không đổi thành candidate frames per event                                  |
| Research candidate path budget            | `candidate_path_limit`                                           | Chỉ ở experiment config khi consumer xuất hiện                              |

Convention: `normalize_*` trả normalized data; `validate_*` fail hoặc trả None; `score_*` tính evidence; `align_*` giải path; `build_*` project output; `verify_*` đánh giá claim. Không rename tất cả method đang ổn chỉ để đúng convention mới.

## 4. PR sequence và gates

| PR | Nội dung                                                       | Loại                                     | Owner      |
| -- | --------------------------------------------------------------- | ----------------------------------------- | ---------- |
| 0  | Inventory, characterization, golden fixtures                    | Tests/docs                                | A          |
| 1  | Shared explicit-event normalization                             | Cleanup + explicit invalid-input behavior | A          |
| 2  | Planner parity và request mismatch error                       | Bugfix được đánh dấu                | A          |
| 3  | Canonical path checks và output projection owner               | Cleanup                                   | A          |
| 4  | Missing-vs-empty count access                                   | Additive data-access contract             | B          |
| 5  | Audit wiring/legacy branches, chỉ xóa cái chứng minh unused | Cleanup                                   | A+B review |
| 6  | Regression, coverage, docs và freeze                           | Release gate                              | Cả hai    |

Dependency: 0 → 1 → 2 → 3 → 5 → 6. PR4 chạy song song sau PR0 và phải xong trước PR6. A/B không cùng sửa file ownership đã phân.

## Task 0 — Establish executable checkout và regression fixture

**Files:** existing repository config/test configuration; create `tests/architecture/test_query_path_contract.py`; add fixtures trong `tests/fixtures/query_path_contract/`; update existing baseline README với compatibility notes.

**Consumes:** existing 5 baseline outputs và existing tests. **Produces:** một regression manifest ghi revision, source fingerprints, test command, score/timestamp semantics.

- [ ] Chạy read-only `git status --short`, `rg --files -g 'AGENTS.md' -g 'pyproject.toml' -g 'pytest.ini' -g '*requirements*'`. Đọc applicable instructions; không tự tạo project config thay config thiếu.
- [ ] Kiểm tra `pytest --collect-only -q` bằng environment của repo. Nếu không collect được, ghi dependency/import blocker và giải quyết trong scope trước khi refactor.
- [ ] Chọn existing tests của planner, query prep, KIS, TRAKE, temporal service và corpus; giữ nguyên assertions làm characterization.
- [ ] Tạo fixtures nhỏ không GPU: attribute sentence, trailing question, “trước đó”, explicit TRAKE, singleton path, missing/empty objects, ASR boundary. Ghi expected values thủ công từ semantics đã duyệt, không regenerate golden sau mỗi refactor.
- [ ] Test schema surface bằng existing FastAPI app fixture: so sánh path set và schemas của ba endpoints với golden; không hardcode route factory thứ hai trong test.
- [ ] Commit test/docs riêng. Gate: biết rõ test nào pass/fail trước cleanup; failure tồn tại trước không được báo do refactor.

**Test assertion mẫu trên baseline data:**

```python
assert after.video_id == before.video_id
assert after.frame_ids == before.frame_ids
assert after.frame_idxs == before.frame_idxs
assert after.timestamps_ms == before.timestamps_ms
assert after.score == before.score
```

Không so sánh latency exact. Nếu numeric kernel không đổi, kỳ vọng exact scores; không nới tolerance để che regression.

## Task 1 — Một normalizer cho explicit events

**Modify:** `src/hcmai/temporal/planner.py`, `query_preparation/service.py`, `orchestration/workflows/temporal_search.py`.
**Tests:** `tests/temporal/test_planner.py`, `tests/query_preparation/test_service.py`, `tests/orchestration/test_temporal_search.py`.

**Produces:** `normalize_event_texts(events: Sequence[str]) -> tuple[str, ...]` ở planner (public). `_normalize_parts()` giữ nguyên private — nó phục vụ raw-query splitting với semantics khác (cho phép drop blank lines, strip punctuation). Hai function không được hợp nhất. Reject empty sequence, raw str/bytes thay vì sequence-of-events, non-string elements và whitespace-only elements. Không silently drop event.

**Modification ở `temporal_search.py`:** workflow hiện đang xử lý original/retrieval event arrays inline (filter blank, normalize whitespace). Thay bằng gọi `normalize_event_texts()` cho cả hai arrays để đảm bảo index alignment không bị lệch khi blank event bị drop ở một array mà không ở array kia.

- [ ] Viết failing tests:

```python
def test_normalize_event_texts():
    assert normalize_event_texts(["  A  B ", " C\nD"]) == ("A B", "C D")

@pytest.mark.parametrize("events", [[], [" "], ["A", ""], [None], "ABC"])
def test_normalize_rejects_invalid_events(events):
    with pytest.raises(ValueError):
        normalize_event_texts(events)
```

- [ ] Run `pytest tests/temporal/test_planner.py -q`; xác nhận test mới fail vì function chưa có.
- [ ] Implement normalization một nơi; query prep translate `ValueError` thành existing `QueryPreparationError` để giữ service boundary. Temporal search gọi function cho original/retrieval arrays thay vì filter blank originals rồi làm lệch index.
- [ ] Không thay low-level split primitive: nó được phép bỏ blank lines khi xử lý raw query; explicit event list không được bỏ event. Không hợp nhất hai semantics này.
- [ ] Run ba test files nêu trên; check valid-input parity và invalid input không shift mapping.
- [ ] Commit `refactor: centralize explicit event normalization` với release note ghi blank-event rejection rõ ràng.

## Task 2 — Planner parity và lỗi request có kiểu

**Modify:** `src/hcmai/orchestration/pipeline.py`, `orchestration/workflows/kis.py`, `api/routers/search.py`.
**Create:** `src/hcmai/orchestration/errors.py` nếu cần chứa `InvalidQueryInputError(ValueError)` dùng bởi workflow và router, không đặt lỗi trong HTTP layer.
**Tests:** `tests/api/test_query_candidates.py`, `tests/orchestration/test_kis_pipeline.py`, `tests/api/test_search_contract.py` mới nếu chưa có equivalent.

**Consumes:** `normalize_event_texts`, existing `plan_query_events`. **Produces:** raw-query candidates và KIS có cùng planned events; user retrieval override mismatch trả 422 có chủ đích.

**Call site xác nhận:** `pipeline.py:generate_query_candidates()` (L403) gọi `split_query_events(request.query or "")` khi `request.events is None`. Đây là call site cần migrate sang `plan_query_events`.

- [ ] Viết integration test dùng fake query-prep adapter capture events và fake temporal capture events. Cùng query attribute/trailing-question/earlier cue phải capture cùng tuple và order. Explicit TRAKE không được reorder.
- [ ] Chạy tests và xác nhận parity case fail trên code cũ.
- [ ] Thay raw branch `split_query_events(request.query or "")` tại `pipeline.py:generate_query_candidates()` (L403) bằng `plan_query_events(request.query or "")`. Không sửa `split_query_events` public behavior; migrate call site cần semantic planning thôi.
- [ ] KIS mismatch raise `InvalidQueryInputError`; search router chỉ catch loại này và trả 422. Không catch toàn bộ ValueError: canonical corruption vẫn là server/data failure.
- [ ] Test oracle:

```python
assert candidates.original_events == search.events
assert mismatch_response.status_code == 422
assert canonical_corruption_does_not_become_422
```

Trong implementation, assertion cuối dùng fake service raise plain ValueError và TestClient có `raise_server_exceptions=False`; expect 500, không catch-all 422.

- [ ] Run API và workflow tests. Invalidate old generated candidate cache bằng prompt/cache namespace revision hiện có nếu event-planning change yêu cầu; không dựng cache mới.
- [ ] Commit riêng `fix: align query candidate planning with KIS`. Ghi rõ: wire schema không đổi, raw-query output semantics có đổi; phối hợp team interactive và không reuse stale candidate bundles.

## Task 3 — Một owner cho canonical path checks và projection

**Modify:** `src/hcmai/orchestration/materializer.py`, `orchestration/workflows/temporal_search.py`, `orchestration/workflows/trake.py`.
**Tests:** `tests/orchestration/test_temporal_search.py`, `tests/orchestration/test_trake_pipeline.py`, `tests/unit/orchestration/test_materializer.py` hoặc existing equivalent đã collect ở Task 0.

**Produces:** `SearchMaterializer.validate_aligned_path(path: AlignedPath) -> None`, `SearchMaterializer.build_trake_path(path: AlignedPath) -> TRAKEPath` static projection.

**Validation extraction pattern:** `build_kis_result()` hiện chứa inline canonical checks (L32-52 materializer.py). Extract thành `validate_aligned_path()`, sau đó `build_kis_result()` gọi `self.validate_aligned_path(path)` làm bước đầu tiên — inline checks được xóa, không duplicate.

**TRAKE validation requirement:** `trake.py` workflow phải gọi `materializer.validate_aligned_path(path)` trước `SearchMaterializer.build_trake_path(path)`. Hiện tại `_build_path()` (L108 trake.py) không validate gì — đây là gap cần đóng. `build_trake_path()` là static projection thuần, không validate vì caller đã validate.

- [ ] Test path nhiều frame với frame đầu đúng nhưng frame sau sai video/index/timestamp; validator phải reject. Test KIS giữ upper-middle representative. Test TRAKE giữ tất cả arrays và score. **Test TRAKE path đi qua validate trước build.**
- [ ] Run targeted tests trước để thấy missing public validation/projection cases fail.
- [ ] Extract canonical path checks từ `build_kis_result()` vào `validate_aligned_path()`; `build_kis_result()` gọi `self.validate_aligned_path(path)` làm bước đầu — xóa inline checks cũ. Static TRAKE projection chỉ đổi representation.
- [ ] TRAKE workflow gọi `materializer.validate_aligned_path(path)` trước `SearchMaterializer.build_trake_path(path)` — đóng validation gap hiện tại.
- [ ] Temporal materialization giữ score-matrix membership checks tại boundary numerical output, assemble AlignedPath, rồi delegate canonical path check. Không xóa `_validate_video_scores` vì nó validate numerical input, không cùng chức năng output-path validation.
- [ ] TRAKE chuyển `_build_path()` implementation sang materializer; remove wrapper sau khi migrate tất cả callers, không giữ alias vô thời hạn.
- [ ] Không tạo `CanonicalPathService`, `PathRegistry`, `PathDTOFactory`. Không chuyển Corpus vào `temporal/dp.py`.
- [ ] Run targeted tests + golden output; assert same tuples/scores. Commit `refactor: consolidate canonical path projection`.

## Task 4 — Preserve unavailable versus empty object evidence

**Modify:** `src/hcmai/corpus/corpus.py`.
**Tests:** `tests/corpus/test_corpus.py`.

**Produces:** `Corpus.object_counts_optional(frame_id: str) -> dict[str,int] | None`. Existing `object_counts()` delegates rồi giữ public empty fallback.

- [ ] Viết tests absent store, absent record, failed record → None; completed-empty → {}; completed-nonempty → copy của counts. Test caller mutate returned dict không làm đổi store.
- [ ] Run tests và xác nhận thiếu method gây fail.
- [ ] Minimal implementation shape:

```python
def object_counts_optional(self, frame_id: str) -> dict[str, int] | None:
    if self._object_counts is None:
        return None
    return self._object_counts.get_counts(frame_id)

def object_counts(self, frame_id: str) -> dict[str, int]:
    return self.object_counts_optional(frame_id) or {}
```

**Verified:** `ObjectCountsStore.get_counts()` (catalog.py:L141-147) đã trả `None` cho absent record hoặc non-completed status, trả `dict(record.counts)` (copy) cho completed. Semantics khớp với implementation shape — không cần sửa store.

- [ ] Run corpus tests, filter/materializer compatibility tests. Không đổi method get_counts bên dưới vốn đã có semantics cần thiết.
- [ ] Merge gate: nếu chưa có consumer thật (research collector) sẵn sàng trước PR6 freeze, **drop PR4 khỏi cleanup scope** thay vì giữ unmerged PR vô thời hạn. Consumer xuất hiện sau sẽ tạo PR riêng trên contract đã chốt.
- [ ] Commit `feat: preserve optional object-count evidence semantics` khi consumer gate đạt.

## Task 5 — Remove verified duplicate/unused wiring, không xóa theo cảm tính

**Inspect/conditionally modify:** `orchestration/pipeline.py`, `workflows/kis.py`, `workflows/trake.py`, `orchestration/setup.py`, `query_preparation/service.py`, `api/contracts/query_candidates.py`, tests tương ứng.

**Consumes:** Task 0 call-site inventory; passing tests Tasks 1–3. **Produces:** mỗi symbol bị xóa có caller migration hoặc bằng chứng không còn consumer.

- [ ] Search `rg -n 'query_preparation|score_event_videos|_build_path|split_query_events|_normalize_events' src tests baseline`. Kiểm tra exports/constructor kwargs và monkeypatch fixtures.
- [ ] KIS/TRAKE currently store unused `query_preparation`. Nếu không có external/internal consumer đã xác nhận, bỏ constructor parameter và assignment cùng setup call sites trong một PR. Giữ `SearchService.query_preparation` vì query-candidates endpoint dùng thật.
- [ ] Legacy `getattr` fallback không tự động là dead code: nếu vẫn có baseline adapter hoặc tests phản ánh supported mode thì giữ và document. Nếu chỉ tests cũ dùng, quyết định compatibility trước rồi migrate fixture đúng contract và remove fallback cùng test path mới.
- [ ] Không xóa image reranking package chỉ vì temporal path không gọi; phải audit consumers ngoài flow này.
- [ ] Không gộp Pydantic request/response classes thành inheritance tree chỉ để share ba fields. Giữ duplicate boundary constraints khi phục vụ error locality.
- [ ] Với fixed count=5, chỉ extract domain constant khi actual duplicate consumers được migrate cùng; không cho config tùy ý trong khi validator vẫn fixed. Preserve HTTP five-bundle behavior.
- [ ] Run import/collection checks, architecture tests, affected unit tests. Commit theo từng deletion có lý do, không tạo một commit “cleanup everything”.

## Task 6 — Output consistency, coverage và freeze

**Modify:** `baseline/README.md`, relevant module READMEs và tests; `baseline/runner.py` chỉ khi có proven output duplication cần migration, không thay evaluator envelope trong cleanup.

- [ ] Ghi output mapping: HTTP KIS/TRAKE giữ schemas; baseline evaluator giữ existing run schema; research thêm sidecar keyed by run/query/video/ordered frame IDs, không overwrite `.metrics.json` cũ.
- [ ] Kiểm tra runner đang có `run_cases()` và `_response_body()` trước khi tạo research runner. Không gọi private `_response_body()` từ module khác; nếu consumer thứ hai cần output builder, promote đúng function đó thành public helper trong owner hiện có và test cả hai, không copy implementation.
- [ ] Ghi source requested/effective readiness và fusion mode trong experiment metadata; không cần thêm wire fields production ở cleanup.
- [ ] Run full collected CPU suite; run targeted branch coverage cho planner, query prep, materializer và affected workflows bằng pytest-cov nếu environment đã có.

```bash
pytest tests/temporal tests/query_preparation tests/orchestration tests/api tests/corpus -q
pytest tests/temporal tests/query_preparation tests/orchestration tests/api tests/corpus --cov=hcmai.temporal.planner --cov=hcmai.query_preparation --cov=hcmai.orchestration.materializer --cov-branch --cov-report=term-missing
```

- [ ] Không claim coverage nếu plugin/dependencies thiếu. Không đặt mục tiêu 100% toàn repo; yêu cầu tất cả new semantic branches có test và không giảm coverage của touched modules so với Task 0 measurement.
- [ ] Replay 5 baselines trên fixture cố định với same matrices; output identities/order/scores bằng trước. Real-corpus benchmark chỉ chạy khi configured artifacts/models có sẵn; không fabricate pass từ mock.
- [ ] Planner bugfix được đánh giá riêng: giữ old artifacts, sinh run revision mới; nếu benchmark QueryCase.events đã explicit thì ghi lý do không bị raw planner change ảnh hưởng.
- [ ] Review OpenAPI schema diff, golden JSON, effective sources, latency units; không chỉnh golden tự động để làm tests xanh.
- [ ] Freeze contract và commit docs/tests. Sau gate này mới merge QueryPlan/verifier research từ spec; không có model training hoặc new retrieval algorithm thuộc cleanup plan.

## 5. Coverage matrix — acceptance theo hành vi

| Invariant                                  | Unit                              | Integration              | Golden/Replay           |
| ------------------------------------------ | --------------------------------- | ------------------------ | ----------------------- |
| Explicit events không silently drop       | Normalizer                        | Query prep → temporal   | Event order             |
| Candidate/KIS planner parity               | Planner cases                     | API fake adapters        | Known divergent queries |
| User error 422, data corruption không 422 | Typed error                       | TestClient               | Response schema         |
| Same-video canonical path                  | Path validator                    | Temporal → materializer | All frame IDs           |
| Score semantics giữ nguyên               | Projection                        | KIS/TRAKE                | 5 baseline outputs      |
| ASR half-open interval                     | Existing transcript tests         | Materializer/collector   | Start/end fixture       |
| Empty khác unavailable counts             | Corpus                            | Collector consumer       | Evidence packet         |
| No API surface growth                      | Contracts                         | OpenAPI                  | Schema golden           |
| No duplicated scoring for logging          | Fake scorer call counter (Task 6) | Runner                   | Call count=1 normal run |

## 6. Hai người làm song song

**A (phuckhang)** sở hữu planner/query/workflow/API migration; **B (collaborator)** sở hữu evidence semantics và collector fixtures. A làm Tasks 0–3, 5–6; B bắt đầu Task 4 sau khi fixture/input conventions ở Task 0 được chốt. Không cùng refactor temporal service và Corpus trong một nhánh.

Mỗi PR bàn giao: changed symbols, consumers migrated, tests run, intentional behavior changes, remaining blockers. Chỉ integrate khi fixture thống nhất; không chờ verifier model hoàn hảo để kiểm tra contract.

## 7. Stop conditions và definition of done

Dừng refactor khi không xác định được caller ngoài repo, source snapshot khác live checkout, canonical output đổi ngoài intentional planner fix, hoặc cần thêm endpoint để vượt giới hạn task. Báo rõ quyết định cần người dùng chốt, không mở rộng scope ngầm.

Done khi: một owner cho mỗi operation trong bảng; không có duplicate replacement function còn tồn tại; old public schemas giữ nguyên; intended behavior changes có release notes; regression/coverage evidence đã ghi; 5 baselines giữ identities/scores ở cleanup-only path; schema research dùng được mà không tạo production dead code.

Chưa done không được ghi “coverage đầy đủ”, “không còn dead code” hay “không regression” chỉ từ rg/static trace.
