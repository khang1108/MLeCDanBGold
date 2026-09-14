# VBS Explore Inspector — Frontend Integration Plan

> **For agentic workers:** Use `superpowers:executing-plans` only when explicitly assigned implementation. Follow the checkboxes; no parallel-agent workflow is needed.

**Goal:** Nối backend exploration v22.1 vào inspector của frontend v5, dùng controls tiếng Anh đã duyệt và giữ nguyên global results.

**Architecture:** Một panel trong `ImageModal`, một hook được gọi ở `AppContent` là nơi đang sở hữu inspector, một API module. Backend bổ sung thin transport cho `TemporalExploration`; không đưa logic DP vào frontend. State sống ngoài modal để đóng inspector không mất nhánh.

**Tech Stack:** React/hooks, `requestJson` hiện có, FastAPI/Pydantic hiện có, backend core v22.1. Không thêm framework hoặc dependency.

**Spec:** `docs/superpowers/specs/2026-09-13-vbs-scoped-temporal-feedback-design.md`; bổ sung thiết kế inspector và labels đã được người dùng duyệt trong hội thoại. Lifecycle theo `2026-09-13-vbs-v22-minimal-followup-plan.md`.

## Global Constraints

- Người dùng implement trên `vbs`; assistant chỉ design/review.
- Research MVP: không timeline editor, drag handles, global store, distributed sessions, agent hoặc entity-aware.
- Feedback không phải answer approval. Approve/Decline chỉ tác động event và interval đang chọn.
- Giữ global response, một active branch/tab; không tự thay global ranking bằng local path.
- Teammate sở hữu migration TRAKE/submission-file → AnswerWorkspace/DRES. Không thực hiện lại migration trong change này.
- Frontend v5 vẫn dùng submission-file và frame_idx; chưa có candidate dialog VBS. Không gọi legacy `requestSubmission` từ exploration.
- Hai controls chọn mốc chỉ sửa draft; không gọi backend khi tua video hoặc nhập số.
- Không tự snap thời gian, không tự nới ràng buộc, không tự xác nhận nội dung và không tự submit.
- Chỉ hai automated scenarios mới cho transport và UI; một walkthrough xuyên suốt. Không build bộ testing infrastructure mới.

## 1. Labels và layout đã khóa

| Vị trí | Label |
|---|---|
| Inspector trước khi mở branch | **Explore** |
| Event selector | **Event** |
| Khoảng feedback | **Start**, **End** |
| Lấy đầu/cuối từ player | **Use current time as start**, **Use current time as end** |
| Gửi positive/negative | **Approve**, **Decline** |
| Mở phần phạm vi nhánh | **Search range** |
| Áp phạm vi nhánh | **Search this range** |
| Khôi phục điều kiện | **Undo** |
| Bỏ branch, đóng inspector, trở về grid | **Back to results** |

Panel đặt cạnh/dưới player theo layout hiện có, không mở modal lồng nhau. Phía trên có danh sách events với thumbnail/timestamp của path và event đang chọn. Phía dưới là feedback draft và hai nút Approve/Decline, tiếp theo điều kiện đang có, Undo, Back to results. Search range là phần thu gọn riêng với hai bounds riêng.

Dùng một dòng nhắc ngắn: **“Applies to the selected event and time range.”** Tên trạng thái hiển thị: **Approved range**, **Declined ranges**, **Updated moment**. Không gọi similarity là confidence hoặc correctness.

Khi chưa có path, giữ danh sách event và conditions; không giả thumbnail cũ là kết quả hiện tại. Có thể giữ player để người dùng xem tiếp. Không cần nút Unknown: chưa thao tác thì chưa có feedback.

## 2. File map và ownership

| File | Thao tác |
|---|---|
| `frontend/src/api/exploration.js` | Tạo client mỏng dùng requestJson |
| `frontend/src/features/alignment/hooks/useTemporalExploration.js` | Tạo state/actions, pending, stale guard |
| `frontend/src/features/alignment/components/ExplorationPanel.jsx` | Tạo panel; chỉ draft input và render |
| `frontend/src/App.jsx` | Gọi hook một lần; truyền view/actions tới inspector |
| `frontend/src/features/search/components/SearchWorkspace.jsx` | Giữ full live KIS response và chuyển snapshot khi mở inspector |
| `frontend/src/features/frames/components/ImageModal.jsx` | Gắn panel, capture video time và seek event path |
| `frontend/src/features/alignment/components/AlignmentAccordion.jsx` | Optional click event để mở inspector tại event đó; không nhét controls feedback trên từng card |
| Styles hiện đang chứa `.modal-frame-stack` | Thêm styles panel cùng nơi, không tạo design system |
| `src/hcmai/api/contracts/exploration.py` | Tạo request/envelope transport, reuse canonical path fields |
| `src/hcmai/api/routers/exploration.py` | Tạo routes và registry nhỏ trong một process |
| `src/hcmai/app.py` | Register router, cấp temporal service và registry theo app lifetime |
| `tests/api/test_exploration_routes.py` | Một lifecycle scenario mới |
| `frontend/src/features/alignment/components/ExplorationPanel.test.jsx` | Một interaction scenario mới |

SearchWorkspace, App và ImageModal là điểm giao với teammate: áp diff nhỏ trên bản họ đang làm, không thay toàn bộ file bằng bản v5. Tìm file CSS bằng `rg -n 'modal-frame-stack' frontend/src` và thêm đúng file trả về.

## 3. Task 1 — Thin backend transport

**Dependency:** Backend v22.1 đã có core nhưng chưa có exploration endpoints. Contract dưới đây là **đề xuất bổ sung của plan này**, không phải API đã tồn tại. Frontend/backend dùng chung tên này khi implement; không thay SearchRequest/SearchResponse/DRES contracts.

### Contract tối thiểu

| Method/path | Request | Response |
|---|---|---|
| POST `/api/v1/exploration` | query, events, retrieval_events, caption_events, use_dense, use_bm25, video_id, window | handle, scoring_revision, view |
| GET `/api/v1/exploration/{handle}` | — | handle, scoring_revision, view |
| POST `/api/v1/exploration/{handle}/actions` | expected_revision, event_version, scoring_revision, action, event_index?, interval? | handle, scoring_revision, view |
| DELETE `/api/v1/exploration/{handle}?expected_revision=N` | — | 204 |

Open request: event arrays giữ nguyên snapshot đang xem, E và giới hạn theo config hiện có. `window`/`interval` là `[start_ms,end_ms]`, số nguyên strict, không nhận bool/float. `caption_events` nullable. Không thêm frame/score matrix vào request.

Action gồm `confirm`, `reject`, `window`, `undo`. Undo gọi method `undo()` hiện có; các action khác gọi `apply()`. Approve map confirm, Decline map reject. Không cần publish unknown vì UI không mutate khi chưa rõ.

`view` mang fields của ExplorationView, `paths` giữ đúng AlignedPath fields; JSON dùng arrays cho tuples. Không tạo domain AlignedPath/Event/Frame mới. Pydantic chỉ là transport adapter. Reject extra request fields và giữ error messages đọc được với `requestJson` hiện có.

- [ ] Tạo open validation và mapping sang QueryBinding. Backend sinh event_version mới bằng UUID mỗi open, không chạy planner lại. `scoring_revision` là generation của service/index/config được nạp lúc startup; không tin phiên bản client tự tạo. MVP dùng service immutable trong process; reload config/index bằng restart, branch cũ hết hiệu lực.
- [ ] Cấp handle UUID mới cho mỗi instance, registry giữ `(branch, scoring_revision, last_access)`. Một worker cho demo; không thêm Redis/database. Cleanup lazy sau30 phút inactivity và khi DELETE, tối đa16 handles gồm opens đang pending; hết chỗ trả503 với thông báo, không eviction branch đang thao tác. Dùng lock nhỏ bảo vệ reservation/removal và không xóa entry in-flight. Đây là cap deployment tạm thời, không benchmark tối ưu.
- [ ] Run scoring/decode ngoài event loop bằng `run_in_threadpool` như router search hiện có. Lock core v22.1 bảo vệ apply/undo/close; registry không được thay handle cũ sang instance mới.
- [ ] Map stale conflict→409, handle mất→404, invalid input→422, dependency unavailable→503. Các semantic statuses như no_valid_path vẫn200 với conditions để sửa/undo. Không catch mọi exception thành video irrelevant.
- [ ] GET trả snapshot core; cần để đồng bộ sau timeout/409. Không auto-retry mutation: request timeout có thể đã được backend áp. Client đọc lại current, rồi người dùng quyết định thao tác tiếp.
- [ ] Tạo một route scenario bằng fake scorer/real core: open→confirm→undo→delete→GET old handle404. Kiểm envelope/canonical timestamp; không viết lại DP tests. Run riêng file test này.

**Không có DRES dependency để Explore hoạt động.** Khi logger chung của teammate đã có, gắn successful retrieval results qua adapter của họ; không tạo DRES client trong router này.

## 4. Task 2 — Snapshot và hook ngoài modal

**Interfaces đề xuất:**

```javascript
// api/exploration.js — each uses requestJson and forwards signal
openExploration(body, { signal })
getExploration(handle, { signal })
actOnExploration(handle, body, { signal })
closeExploration(handle, revision, { signal })

// hook, called once in AppContent
useTemporalExploration()
// returns { session, pending, error, open, act, undo, close, refresh }
// session is the server envelope; draft inputs belong to the panel.
```

- [ ] Khi live KIS response về, SearchWorkspace giữ **full response** trong state/ref riêng, không chỉ events/results. Snapshot gồm query, events, dense_events, bm25_caption_events, source flags. onFrameClick truyền snapshot này trong selection; không lấy query đang gõ vì nó có thể khác query đã trả results.
- [ ] AppContent sở hữu hook và inspector cùng cấp. ImageModal chỉ nhận session/actions; đóng modal bằng X/Escape giữ branch. Mở lại cùng query/video nối lại branch hiện có. Back to results mới gọi close và đóng modal.
- [ ] Khi bắt đầu query mới, Replay hoặc chuyển sang video khác để Explore: invalidate UI generation, đóng branch cũ rồi tạo branch mới. Không đưa feedback sang query/video mới. Filter/image/manual-video/replay chưa có đủ scoring snapshot: không hiện Explore trong MVP. Không đoán snapshot từ history thiếu fields.
- [ ] Hook serialize mutation qua pending: chặn double-click Approve/Decline/Undo khi chờ. Lưu generation counter cho open/close và kiểm `(generation, handle, revision)` trước áp response; AbortController không thay thế stale guard.
- [ ] Sau mutation timeout hoặc409, gọi GET đúng handle, cập nhật current nếu còn cùng generation; không gửi lại action tự động. Nếu GET cũng lỗi, giữ view cuối cùng với thông báo chưa đồng bộ và chặn mutation đến khi refresh thành công hoặc Back to results. 404 clear branch; global grid vẫn còn.
- [ ] Nếu open bị hủy nhưng backend vẫn tạo branch, response thành công đến muộn được cleanup bằng handle nếu nhận được; nếu không nhận được, TTL dọn. Không thêm retry/outbox/persistence.

Ví dụ mapping open từ snapshot:

```javascript
const body = {
  query: snapshot.query,
  events: snapshot.events,
  retrieval_events: snapshot.use_dense ? snapshot.dense_events : snapshot.events,
  caption_events: snapshot.use_bm25 ? snapshot.bm25_caption_events : null,
  use_dense: snapshot.use_dense,
  use_bm25: snapshot.use_bm25,
  video_id: selectedFrame.video_id,
  window: [0, Math.floor(videoDurationSeconds * 1000)],
};
```

Chỉ enable Explore khi metadata duration hữu hạn/dương và snapshot hợp lệ. Window ban đầu là toàn video, hiển thị rõ; không tự chọn ±5s quanh representative rồi giới hạn tất cả events. Reload trang mất branch client; chấp nhận cho MVP, không localStorage recovery.

## 5. Task 3 — Panel và tương tác player

**Props:** `ExplorationPanel` nhận `events`, `session`, `pending`, `error`, `onApprove`, `onDecline`, `onUndo`, `onSearchRange`, `onBack`, `onSeek`, `readCurrentTimeMs`. Chỉ giữ selected event index, feedback draft và search-range draft. Backend conditions là nguồn sự thật, không render draft như đã approved.

- [ ] Thêm Explore trong inspector của live KIS. Panel hiện sau open; event mặc định là event người dùng vừa chọn nếu có, nếu không là0. Event chips dùng `E1`, `E2` cho display nhưng payload zero-based.
- [ ] Lấy mốc trực tiếp từ videoRef lúc click, không từ playbackTime cập nhật theo timeupdate.

```javascript
const readCurrentTimeMs = () => {
  const seconds = videoRef.current?.currentTime;
  return Number.isFinite(seconds) && seconds >= 0
    ? Math.round(seconds * 1000)
    : null;
};
```

Đọc không được thì disable/thông báo; không fallback sang frame_idx. Input draft hiển thị seconds với ba chữ số thập phân, chuyển về integer ms khi gửi; validate finite, nonnegative, start≤end. Không tự đổi hai đầu nếu nhập ngược.

- [ ] Giữ draft đầu/cuối rỗng lúc đầu; hai nút Use current time điền từng đầu. Không lấy aligned point làm confirmed mặc định. Sau khi đổi selected event, clear draft để tránh vô tình dùng khoảng của event trước.
- [ ] Approve/Decline chỉ gửi khi hai đầu hợp lệ. Core đã tự re-decode; không thêm nút Run hoặc lượt search thứ hai. No frame/no path giữ conditions, hiện thông báo ngắn và Undo; không tự mở rộng khoảng.
- [ ] Search range có draft riêng, khởi tạo từ session.view.conditions.window. Search this range map `action=window`, không dùng selected event. Nếu range loại event đã approve, hiển thị status backend và cho Undo; không giấu conflict bằng cách xóa approval.
- [ ] Click timestamp của event seek video hiện tại tới timestamp đó, không remount video/chạy lại open. Render thumbnails bằng canonical frame_ids/keyframeUrl. Không tạo Frame giả bằng frame_idx suy ra từ timestamp.
- [ ] Render conditions và Updated moment riêng. Nếu comparison_available=false, không diễn giải changed_event_indices rỗng là path không đổi. Không tự seek player khi path cập nhật; để user chủ động chọn event để xem.
- [ ] Viết một component scenario: chọn E2, capture start/end, bấm Approve → callback nhận event_index1 và đúng interval; Undo gọi callback riêng. Mock callbacks, không fake cả backend/browser/video pipeline. Các nút draft không được gọi onApprove. Chạy riêng file test rồi build frontend một lần.

## 6. Task 4 — Candidate handoff và một walkthrough

- [ ] Rebase/merge các edits nhỏ trên frontend của teammate. Giữ migration TRAKE/AnswerWorkspace/DRES do họ sở hữu; không bê legacy submit callbacks vào panel.
- [ ] Khi candidate dialog VBS có sẵn, inspector action đọc currentTime tại click và gửi canonical video_id + timestamp_ms tới callback của họ. Nếu chọn indexed event frame thì giữ timestamp của frame đó; nếu chọn arbitrary playback moment thì provenance frame chỉ có khi thực sự khớp. Không gửi explanation/conditions vào answer body.
- [ ] Nếu migration chưa có, demo exploration vẫn chạy và seek được; ghi rõ candidate handoff chưa tích hợp. Không làm một dialog/submission tạm khác chỉ để gọi feature “done”.
- [ ] Chạy một query xuyên suốt: Explore→Approve E1→Decline khoảng cho E3→xem Updated moment→Undo→đóng/mở inspector vẫn còn branch→Back to results giữ grid. Cuối cùng chọn candidate qua dialog teammate nếu đã có và kiểm selected timestamp. Lưu screenshot/clip cho paper.

## Điểm dừng và kiểm chứng

Hai automated scenarios nêu trên đủ cho phần mới; reuse core tests đã pass. Không thêm snapshot UI hàng loạt, coverage target, load tests hoặc sweep model. Nếu walkthrough phát hiện lỗi cụ thể mới mở thêm kiểm tra tương ứng.

Core/API/controls usable là một milestone. Candidate handoff và logging dùng chung là milestone tích hợp tiếp theo. Cả hai không chứng minh feedback giúp người dùng nhanh hơn; sau khi UI chạy, thu B–C theo protocol hiện có, tách query rehearsal khỏi query chưa biết đáp án.

**Self-review:** Labels đúng approval; state ngoài modal phù hợp App.jsx thực tế; một branch/global snapshot; draft tách constraints; event range khác branch window; API bổ sung được ghi là đề xuất mới; không trộn legacy submission với DRES; không có implementation được thực hiện bởi assistant.
