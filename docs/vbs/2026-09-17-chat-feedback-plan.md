# Chat Feedback for KIS — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task. Steps use checkbox syntax. This deliverable is a plan, not permission to implement.

**Goal:** Dùng chat panel hiện tại để sửa intent, cải thiện retrieval và sửa temporal path bằng feedback có phạm vi rõ ràng.

**Architecture:** Một feedback resolver trả structured action; backend kiểm tra phạm vi rồi thực thi qua KIS retrieval và EventTrail hiện có. Canonical intent, retrieval overrides và temporal constraints là ba loại state riêng; không rewrite toàn câu cho mọi feedback. Chat là giao diện chính, các nút Approve/Use/Decline/Undo là shortcut gọi cùng cơ chế.

**Tech Stack:** Python, FastAPI, Pydantic, NumPy; React và react-scripts hiện có; LLMClient.generate_structured hiện có. Không thêm framework agent hoặc vector database.

**Spec:** Section 1 trong chính tài liệu này là design specification; executor đọc toàn bộ trước khi làm. Baseline đã inspect: `src_v1.6(1).zip`. Các đường dẫn dưới đây tương đối với project root (nơi chứa `src/hcmai` và `frontend`). Archive không cung cấp đủ runtime artifacts để đo retrieval thật; không được báo tăng accuracy khi chỉ test synthetic.

## Global constraints

- Một event: retrieval repair được search toàn corpus và đổi video.
- Nhiều event: khi đang sửa candidate/path đã chọn, khóa video đó. Chưa chọn video thì yêu cầu chọn, không tự lấy top 1 làm anchor.
- Chỉ explicit user approval/frame selection mới tạo anchor; DP proposal không phải xác nhận của người dùng.
- Không thêm transition model, caption regeneration, VLM verifier, entity tracking, training hoặc benchmark suite lớn ở đợt này.
- Không gọi LLM để chọn frame_idx, timestamp, video_id hoặc tự quyết định bỏ khóa video.
- Không auto-submit đáp án DRES. Dùng frame làm anchor khác với gửi đáp án.
- Giữ ảnh đã attach, evidence identity, scoring revision và backend local/remote parity.
- Chỉ một pending feedback turn trên client; lỗi không được làm mất state đang dùng.
- Test nhỏ, tập trung semantic invariants; không refactor unrelated code.

## 1. Design specification

### 1.1 Hiện trạng đã xác minh

| Thành phần | Hiện tại | Thay đổi cần có |
|---|---|---|
| `frontend/src/features/kis/parser.js` | Intent đã tồn tại thì plain text bị từ chối; cần `E#:` hoặc `/llm-rewrite` | Plain text sau initial search đi vào feedback endpoint |
| `frontend/src/features/kis/session.js` | Quản lý draft, intent và revision; chưa có conversation turns | Thêm chat transcript, selected context, feedback state và Undo checkpoint |
| `kis/rewriter.py` | Global rewrite bắt buộc giữ số/order event | Không dùng nó để split; thêm topology repair có mapping |
| `orchestration/pipeline.py::search_kis` | Gán `event.text` vào canonical/dense/BM25 text | Cho retrieval text override độc lập, không sửa canonical intent |
| `event_trail/service.py` | Đã có approve/use_frame/decline/undo, state revision và checkpoints | Reuse; bổ sung targeted repair, không tạo solver thứ hai |
| `event_trail/decoder.py` | Mask + selected-video DP trên score matrix đã lưu | Giới hạn block sửa, đóng băng phần không thuộc block; có thể thay evidence sau retrieval refinement |
| `event_trail/models.py::TrailCheckpoint` | Chỉ constraints + submission selection | Không đủ Undo semantic/retrieval changes; cần checkpoint cấp feedback |

Đính chính tiền đề: DP bảo đảm thứ tự trên một video, không chứng minh video đúng về nội dung. Anchor feedback có thể thêm thông tin mới. Không claim anchor-conditioned decoding là novelty: code đã có phần này.

### 1.2 Các hành động feedback

Resolver trả đúng một tagged action mỗi lượt:

| Action | Payload/model responsibility | Execution |
|---|---|---|
| `edit_intent` | Event IDs và replacement text có giới hạn | Chỉ sửa event được chỉ định; tăng semantic revision |
| `restructure` | Replace một contiguous block bằng các event mới; old-to-new mapping | Server canonicalize E1..En và before edges; tạo evidence mới |
| `refine_retrieval` | Event IDs và một retrieval description ngắn/event | Canonical intent không đổi; update retrieval overrides; rescore đúng scope |
| `anchor` | Event ID; dùng selected frame đã được client gửi và server xác thực | Reuse Approve/UseFrame semantics, không lấy timestamp do LLM bịa |
| `reject_candidate` | Event ID và selected/current candidate reference | Reject occurrence, không reject cả video; tìm lại đúng scope |
| `repair_event` | Event ID cần tìm lại | Local repair khi nhiều event; global khi một event |
| `clarify` | Một câu hỏi ngắn | Không mutate intent/results/constraints |

Undo và New Search là typed UI actions, không cần LLM. Tin nhắn yêu cầu đổi video trong multievent trả clarification/actionable suggestion tạo New Search; không tự mở khóa. Một tin nhắn có nhiều yêu cầu phụ thuộc nhau mà không biểu diễn an toàn được trong một action thì hỏi chọn bước trước, không tự chạy chuỗi agent.

MVP dùng **một retrieval override/event**, chưa multi-query expansion. Đây là refinement cơ bản để kiểm tra giá trị feedback, không claim giải quyết compositional binding hoàn chỉnh. Canonical description vẫn hiển thị; ghi rõ search description đã đổi. Chi tiết chưa từng có nhưng user vừa bổ sung là intent edit, không âm thầm giấu trong retrieval override.

### 1.3 Context và LLM budget

Gửi original user query, current intent, retrieval overrides, selected result/event/frame reference, danh sách anchors và tối đa 4 lượt chat gần nhất. Selection lấy từ click card hoặc viewer; hiển thị context chip để người dùng thấy “đây” đang chỉ gì. Không gửi toàn corpus, không gửi ảnh khi route chỉ có text.

Một `generate_structured` call, temperature 0, configurable `max_tokens=1024` ban đầu. Prompt: không giải thích suy luận; không thêm vật/người/hành động không được user nêu; chi tiết đồng thời không tách thành temporal events; chỉ sửa scope được yêu cầu. Trả output ngắn có schema; schema/token/provider failure giữ state, báo retry; không gọi fallback rewrite hay vòng self-critique. Dùng client/provider đang cấu hình, không hardcode tham số reasoning đặc thù model.

Server dựng câu xác nhận từ action đã thực thi, không lấy lời model làm bằng chứng thực thi: “Giữ E1. Tìm lại theo mô tả ... trên toàn corpus.” Chỉ hiện applied sau commit.

### 1.4 Phạm vi temporal repair

Với target E_i, lấy confirmed anchor gần nhất bên trái và bên phải theo event order. Nếu có cả hai, tìm strictly giữa hai timestamp. Nếu thiếu một bên, dùng đầu/cuối video; actual candidates là indexed frames trong miền đó. Không đặt cửa sổ 10/30 giây tùy ý.

- Event đầu: trước right anchor; event cuối: sau left anchor.
- Không có anchor nhưng đã chọn multievent video: tìm trong video đó, hiển thị “chưa có anchor”; không gọi proposal là confirmed.
- Nếu anchors không kề nhau, block unconfirmed ở giữa được joint DP lại để giữ thứ tự. Mọi thay đổi phụ trong block phải hiện trong diff.
- Các event ngoài block giữ nguyên frame assignment; anchors trong state giữ chính xác frame ID, không chỉ timestamp (hai frame có thể cùng timestamp).
- Decline dùng rejection cell hiện có, scoped theo video + event + occurrence. Không loại cả video chỉ vì một frame sai.
- Miền rỗng/không có path: trả exhausted và giữ last valid path để xem; không nới anchor, không đổi video.
- Người dùng sửa nội dung event đã anchored: yêu cầu xác nhận bỏ anchor của event đó trước khi áp dụng; anchors khác giữ nguyên.

### 1.5 Topology repair và video scope

`restructure` chỉ xảy ra khi user yêu cầu split/merge/order correction. Server rebuild IDs và edges; không đổi ý nghĩa những event ngoài block. Transfer anchors/images chỉ khi old-to-new mapping là 1:1 và nội dung không đổi. Mapping 1:n hoặc n:1 có ảnh/anchor cần hỏi user cách gán hoặc đồng ý clear; không copy anchor sang mọi event mới.

Scope dựa vào trạng thái đã chọn trước action: sửa một multievent candidate thì giữ video ngay cả khi merge xuống một event; việc mở global trở lại phải là thao tác rõ ràng. Single-event chưa khóa video được split thì chạy global multievent search mới, sau khi chọn result mới mở local repair. Đây là quy tắc lifecycle để tránh số event thay đổi làm scope đổi ngầm.

### 1.6 State, revisions và Undo

Mở một bounded server feedback session từ intent + result snapshot đang active; verify snapshot revision/result identity. Session này sở hữu chat turns, original query, retrieval overrides, exclusions, active KIS result snapshot và optional trail reference. Nó không nhân bản corpus score matrices; giữ references vào stores hiện có.

`feedback_revision` tăng sau mỗi committed action/Undo. `intent.revision` chỉ tăng khi semantic content/topology đổi. Trail revision giữ cơ chế hiện có. Requests có request_id và expected revisions; duplicate request_id cùng payload trả response đã commit; request_id khác payload trả conflict. Bounded session/turn cache dùng TTL/limits theo pattern EventTrail store.

Atomicity: stage intent/overrides/evidence/results/trail changes, validate, rồi commit dưới session lock với revision recheck; không giữ lock suốt LLM/network inference. Lỗi giữa chừng không mutate active state. Unified feedback checkpoint giữ reference trước/sau để Undo cả intent, overrides, exclusions, results và trail; không chỉ gọi Trail Undo cho semantic edits. Undo khôi phục nội dung nhưng cấp revision mới để stale requests vẫn bị từ chối. Checkpoint references phải còn sống trong TTL; expired trả 410, hướng dẫn search lại, không tự regenerate kết quả khác rồi gọi đó là Undo.

### 1.7 Proposed HTTP contract (mới, không phải API đang có)

```text
POST /kis/feedback/open
  { intent, original_query, evidence_snapshot_id, use_dense, use_bm25, top_k }
  -> { session_id, feedback_revision, state }

POST /kis/feedback/{session_id}/turn
  { request_id, expected_feedback_revision, expected_kis_revision,
    expected_trail_revision?, message, selected_result_id?,
    selected_event_id?, selected_frame_id? }
  -> { status: applied|clarification|exhausted,
       feedback_revision, intent, retrieval_overrides,
       results, evidence_snapshot_id, trail?, assistant_message,
       changed_event_ids, scope, can_undo }

POST /kis/feedback/{session_id}/undo
  { request_id, expected_feedback_revision }
  -> same state envelope
```

Reuse existing contracts for KISIntent, SearchResult and EventTrailStateResponse. 409 stale revision, 410 expired state, 422 invalid/missing references; provider failure follows existing gateway error handling. Empty or contradictory retrieval is not HTTP 500. Never use result IDs from one snapshot with another. Register static feedback routes before conflicting dynamic routes if router ordering requires it.

## 2. Implementation tasks

### Task 1 — Contracts, session state và typed feedback actions

**Create:** `src/hcmai/kis/feedback/models.py`, `store.py`, `__init__.py`; `src/hcmai/api/contracts/feedback.py`; `tests/test_kis_feedback_state.py`.

**Interfaces:** `FeedbackTurnRequest`, `FeedbackStateResponse` as Section 1.7; `FeedbackAction` discriminated union as Section 1.2. `FeedbackSessionStore.get(session_id)` and `.locked(session_id)` follow EventTrail store pattern. Internal `FeedbackCheckpoint` holds immutable prior state references, not copied score arrays.

- [ ] Add failing tests for stale revision, duplicate request and Undo content restore with fresh revision.
- [ ] Run `PYTHONPATH=src pytest tests/test_kis_feedback_state.py -q`; verify failures concern missing behavior, not missing dependencies.
- [ ] Implement models/store with bounded TTL/history, selected-reference validation and a checkpoint retained for every applied turn.

```python
# Required invariants in the new tests (fixture helpers defined in this test file).
before = session.state
after = apply_prepared_refinement(session, event_id="E1", text="Two people hanging a blue banner.")
assert after.intent == before.intent
assert after.feedback_revision == before.feedback_revision + 1
restored = undo_feedback(session)
assert restored.retrieval_overrides == before.retrieval_overrides
assert restored.feedback_revision > after.feedback_revision
```

- [ ] Run the focused tests; commit `feat: add revisioned feedback state contracts`.

### Task 2 — Bounded LLM feedback resolver

**Create:** `src/hcmai/kis/feedback/resolver.py`, `prompts.py`; `tests/test_kis_feedback_resolver.py`.

**Modify:** `src/hcmai/orchestration/setup.py` only for injection using existing LLMClient.

**Interfaces:** `FeedbackResolver.resolve(context: FeedbackResolveContext, message: str) -> FeedbackAction`; `FeedbackResolveContext` lives in feedback/models.py and exposes Section 1.3 fields. Resolver has no access to corpus or state mutation.

- [ ] Add parameterized fake-LLM tests: edit extra yellow-shirt detail; split sequential cooking steps; refine banner action without changing intent; “đây” without selection returns clarify; no timestamp/ID invention.
- [ ] Run `PYTHONPATH=src pytest tests/test_kis_feedback_resolver.py -q` and observe failure before implementation.
- [ ] Implement one structured inference call and validate event IDs against context. Scope is assigned by policy, never trusted from model output.

```python
action = llm.generate_structured(
    messages, FeedbackResolution, temperature=0.0, max_tokens=1024
).action
validate_action_references(action, context)
return action
```

`FeedbackResolution` wraps the discriminated action union; `validate_action_references(action, context)` is defined in resolver.py. Fake model payload cannot directly create frames or external actions.

- [ ] Assert LLM call_count == 1; invalid schema leaves state unchanged. Rerun tests and commit `feat: resolve chat feedback into scoped actions`.

### Task 3 — Query repair và retrieval refinement executor

**Create:** `src/hcmai/kis/feedback/service.py`; `tests/test_kis_feedback_execution.py`.

**Modify:** `src/hcmai/orchestration/pipeline.py`, `src/hcmai/retrieval/plan.py` only if needed for builder API; keep transport shape unchanged if existing dense_text/bm25_text suffice.

**Interfaces:** `FeedbackService.open(request) -> FeedbackStateResponse`, `.turn(session_id, request) -> FeedbackStateResponse`, `.undo(session_id, request) -> FeedbackStateResponse`. Extract a shared `build_retrieval_plan(intent, overrides)` helper used by search_kis and feedback. Overrides type is `dict[str, RetrievalOverride]`, containing dense_text and bm25_text; canonical_text always comes from intent.

- [ ] Write tests with stub retrieval: refinement preserves semantic intent; single-event query searches full corpus and may select another video; rejection removes only rejected occurrence before per-video ranking/top-k; split creates correct rows/edges; unchanged event images survive.
- [ ] Run `PYTHONPATH=src pytest tests/test_kis_feedback_execution.py -q` (red).
- [ ] Build retrieval plan without routing refinement through `_resolve_operation(global_rewrite)`.

```python
override = overrides.get(event.id)
row = KISRetrievalEvent(
    event_id=event.id, canonical_text=event.text,
    dense_text=override.dense_text if override else event.text,
    bm25_text=override.bm25_text if override else event.text,
    image_refs=tuple(event.images),
)
```

- [ ] Apply global exclusions to full-corpus score masks before rank_paths, not after slicing top 20. Exclusion records carry video_id, event identity and canonical rejection interval. Score matrix comes from `search_plan_artifact`; bounded snapshot policy must not silently truncate the corpus used for re-ranking.
- [ ] Implement server canonicalization and explicit mapping for restructure. Recompute evidence on changed text/topology; never reuse stale event rows by position.
- [ ] Keep same one-query-per-event fusion in MVP; new retrieval text works on local and remote gateway paths via existing KISRetrievalPlan serialization.
- [ ] Test provider/retrieval error rollback and duplicate submit idempotency; run tests; commit `feat: execute semantic and retrieval feedback`.

### Task 4 — Local temporal repair using existing EventTrail

**Modify:** `src/hcmai/event_trail/decoder.py`, `service.py`, `models.py`; `src/hcmai/kis/feedback/service.py`.
**Create:** `tests/test_kis_feedback_temporal.py`.

**Interfaces:** `TemporalConstraintDecoder.repair(video, constraints, decoder_config, current_path, target_event_index) -> DecodeOutcome`. A pure `repair_block(anchors, target_event_index) -> tuple[int, int]` returns inclusive unconfirmed block indices, bounded by nearest anchors. Validate target is not still anchored before decline/repair.

- [ ] Add synthetic fixture: timestamps `[10,20,30,40,50]` seconds, three events, anchors E1@10 and E3@50; E2 must stay strictly inside; outside block unchanged.
- [ ] Cover first/last/no-anchor cases and duplicate-timestamp frame identity. Cover contradictory anchors and empty candidate domain; no global fallback.
- [ ] Run `PYTHONPATH=src pytest tests/test_kis_feedback_temporal.py -q` (red).
- [ ] Build masks from existing constraints; additionally pin outside-block frame IDs and all anchors exactly by frame position. Enforce strict timestamp anchor bounds explicitly, not only DP column ordering.

```python
# Pin by ID rather than timestamp; integrated into decoder's existing mask.
for e, frame_id in fixed_assignments.items():
    mask[e] &= video.frame_ids == frame_id
paths = temporal.decode_video(video, allowed=mask, decoder_config=decoder_config)
```

- [ ] Reuse rejection_cell. For local refine, obtain fresh selected-video scores with score_video, replace evidence transactionally, preserve validated constraints and call repair. Mark indirect candidate diffs for unconfirmed events inside block.
- [ ] Keep last_valid_path on exhaustion; test Undo restores evidence/constraints, not just visible frame.
- [ ] Run tests; commit `feat: repair bounded event blocks from chat feedback`.

### Task 5 — API wiring and ownership of mutable state

**Create:** `src/hcmai/api/routers/feedback.py`; `tests/test_kis_feedback_api.py`.
**Modify:** `src/hcmai/api/routers/__init__.py`, `src/hcmai/app.py`, `src/hcmai/orchestration/setup.py`, `src/hcmai/orchestration/pipeline.py` for dependency exposure, not a second monolithic action switch.

- [ ] Add FastAPI dependency-stub tests for Section 1.7: open, apply, clarify, Undo, stale/expired and foreign snapshot references.
- [ ] Run `PYTHONPATH=src pytest tests/test_kis_feedback_api.py -q` (red).
- [ ] Wire routes to FeedbackService; reuse existing auth/ownership boundaries where present, never expose another user's session through caller-provided IDs. Snapshot store isolation must be checked during implementation.
- [ ] Route chat and direct shortcuts through the same feedback transaction while a feedback session is active. Old EventTrail API may remain for existing consumers, but UI must not mutate the same trail through two independent owners.
- [ ] Verify local and remote retrieval stubs yield equivalent state contracts; no remote HTTP schema changes unless strictly needed.
- [ ] Run tests; commit `feat: expose transactional KIS feedback endpoints`.

### Task 6 — Chat panel UX and selected context

**Create:** `frontend/src/api/feedback.js`, `frontend/src/features/kis/components/FeedbackThread.jsx`, `frontend/src/features/kis/feedbackSession.js`; colocated `.test.js`/`.test.jsx` files.
**Modify:** `frontend/src/features/kis/components/KisPanel.jsx`, `QueryComposer.jsx`, `frontend/src/features/kis/parser.js`, `session.js`, `frontend/src/features/search/components/SearchWorkspace.jsx`, `frontend/src/features/event-trail/hooks/useEventTrail.js`.

**Interfaces:** API exports `openFeedbackSession(payload)`, `sendFeedbackTurn(sessionId, payload)`, `undoFeedback(sessionId, payload)`. `feedbackSession.js` owns pure reducers for pending/applied/clarification/error/undo state. Existing SearchWorkspace orchestrates initial search and feedback; FeedbackThread only renders turns and Undo callback.

- [ ] Add failing tests: free text accepted after initial search; selected-context chip included; clarification does not replace results; failed turn keeps draft/results; stale asynchronous response cannot overwrite newer state; Undo restores semantic and result display.
- [ ] Run from frontend: `CI=true npm test -- --watchAll=false --runInBand feedbackSession FeedbackThread`.
- [ ] Initial text retains current searchKis flow. After success, open feedback session lazily on first follow-up using stored original query and snapshot. Do not require a new initial search just to chat.

```javascript
if (!state.currentIntent) return submitInitialSearch();
return sendFeedbackTurn(state.feedbackSessionId, {
  request_id: requestId,
  expected_feedback_revision: state.feedbackRevision,
  expected_kis_revision: state.currentIntent.revision,
  message: state.draft,
  selected_result_id: state.selectedContext?.resultId,
  selected_event_id: state.selectedContext?.eventId,
  selected_frame_id: state.selectedContext?.frameId,
});
```

The surrounding submit handler ensures sessionId exists, generates requestId once per attempt (reuse for transport retry), and keeps current staged image behavior. For initial milestone, text feedback with newly staged images requests explicit attach-to-event flow rather than dropping images.

- [ ] Show messages, action summary, changed-event highlights and scope chip (`All videos`/locked video). Event cards remain visible; clicking card sets context, does not approve it.
- [ ] Use `Feedback` label after initial search; retain New Search and Undo. Button Approve/Use/Decline bypasses LLM but enters same action executor/checkpoint path. Keep competition Submit explicit and separate.
- [ ] Reset selection after global results change; preserve valid anchors/local context; clear pending feedback state on New Search. Controls-triggered search must synchronize/reopen session so it cannot keep stale score snapshots.
- [ ] Run focused tests and existing KisPanel/session/parser/EventTrail hook tests; commit `feat: use KIS chat as feedback interface`.

### Task 7 — Cleanup, smoke test and handoff

**Modify:** existing parser/session tests; `src/hcmai/README.md`; `frontend/src/features/workspace/queryHistory.js` if saved feedback snapshots need additional fields.

- [ ] Remove parser rule that blocks natural feedback on an existing intent. Remove mandatory slash-command instructions from composer copy. Keep explicit E#: and direct edit functionality as optional shortcuts only if still reachable; no duplicate rewrite pipeline.
- [ ] Remove stale unused imports/handlers and obsolete tests introduced by replacement. Use `rg` call-site checks before deleting; do not delete the existing global rewriter if external API callers still use it. Do not retain new dead compatibility adapters “just in case”.
- [ ] Ensure existing query history records only committed result changes; add concise action/scope/revision/latency to existing history/logging, not a new analytics subsystem. Replay is read-only and must not call live feedback endpoints.
- [ ] Run new backend test files together, then available existing KIS/EventTrail/temporal regression tests. Resolve actual test paths in full repo; archive has no root dependency manifest, so use project's existing Python environment rather than inventing pins.
- [ ] From frontend run `CI=true npm test -- --watchAll=false --runInBand` and `npm run build`. Report unrelated failures separately, never silently skip them.
- [ ] Manual smoke: banner one-event refinement searches globally; split request creates multiple events; selected multievent E2 repair stays between confirmed anchors; last event searches after preceding anchor; exhausted repair preserves state; Undo restores state; New Search releases video lock.
- [ ] Known diagnostic cases only: banner `L21_V022 / 26081`, cooking `L26_V422 / 4283`. Ground truth is accepted; do not hardcode IDs in retrieval policy. Cooking screenshot shows target video among results, but exact path correctness/FPS not verified. First camera-motion query has no supplied GT. Original Vietnamese queries for later screenshots are not available; use displayed resolved text only as labeled smoke input, not original-query benchmark evidence.
- [ ] Document commands/output and limitations. Do not claim recall improvement from UI smoke tests or two examples. Commit `chore: retire obsolete composer flow and document feedback`.

## 3. Acceptance checklist

- [ ] User can send ordinary chat feedback after search without special syntax.
- [ ] Canonical intent changes only for semantic repair; retrieval refinement is inspectable separately.
- [ ] Single-event repair considers corpus beyond old top 20; candidate rejection does not silently exclude entire video.
- [ ] Multievent local repair never changes video, confirmed anchors or unrelated blocks.
- [ ] Split/merge handles IDs/images/anchors deliberately; simultaneous details do not become fake temporal order.
- [ ] First, last, only-event and no-anchor cases have deterministic scope behavior.
- [ ] Clarification, provider errors, stale responses and exhaustion preserve active valid state.
- [ ] Undo reverses complete committed turn; direct buttons and chat share state ownership.
- [ ] No silent DRES submission, agent loop, automatic video unlock or redundant legacy flow.

## 4. Research hooks, not deliverables in this implementation

Record action type, scope, affected events, before/after rank when available, latency and number of turns. Do not expose GT to feedback resolver. A later study can compare full-query rewrite with scoped feedback at equal interaction budget. Transition-aware scoring and multi-view retrieval require separate evidence/design; neither is needed to ship this feedback foundation. No novelty claim or paper-readiness claim is made here.
