# VBS v22 Minimal Follow-up — Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` if explicitly assigned implementation. Follow the checkboxes below; no subagent workflow is needed for this small change.

**Goal:** Sửa lỗi stale request sau reopen và làm rõ tọa độ VBS, rồi chuyển sang demo/nghiên cứu với lượng kiểm tra tối thiểu.

**Architecture:** Giữ `TemporalExploration` hiện tại. Revision tăng theo đời instance; sửa tài liệu handoff tại chỗ. Không thêm service, schema, framework hoặc lớp session mới trong bản sửa này.

**Tech Stack:** Python hiện có; một test pytest dùng lại fixtures v22; Markdown.

**Spec:** `docs/superpowers/specs/2026-09-13-vbs-scoped-temporal-feedback-design.md`.

## Global Constraints

- Người dùng implement trên branch `vbs`; assistant chỉ viết plan và review.
- Source đối chiếu: `src_hcmai_v22.zip`. Đây là bổ sung hẹp cho plan trước, không chạy lại toàn bộ kế hoạch.
- Ưu tiên research: chỉ thêm **một test regression** cho lỗi đã tái hiện; chạy file exploration hiện có một lần. Không thêm coverage target, load test, CI gate hoặc benchmark sweep.
- Giữ FF-MDP, mask, scoring, decoder snapshot và các schema canonical hiện có.
- Không mở rộng agent/entity-aware hoặc thiết kế lại frontend/DRES.
- Hoàn thành bản sửa không đồng nghĩa hoàn thành UI hoặc chứng minh hiệu quả tương tác.

## Bằng chứng review làm căn cứ

131 tests thuộc temporal và các service scoring/exploration/search đã pass trong môi trường review. Tuy nhiên, request revision1 được giữ từ nhánh cũ vẫn áp dụng được sau close→open về revision1; stale close cũng đóng được nhánh mới. KIS pipeline regression chưa chạy được trong môi trường review do thiếu `faiss`; không coi đó là lỗi implementation đã xác nhận.

Chi tiết reset revision xuất phát từ plan trước của assistant. Plan này thay thế quy tắc “mọi lần open bắt đầu revision1” bằng “lần open đầu tiên là1; các lần mở lại tiếp tục tăng”.

## File map

| File | Thay đổi |
|---|---|
| `src/hcmai/orchestration/workflows/temporal_exploration.py` | Sửa ba chỗ quản lý revision trong open/close |
| `tests/orchestration/test_temporal_exploration.py` | Thêm một test dùng fixtures đã có |
| `docs/vbs/scoped-feedback-handoff.md` | Sửa revision semantics và câu về frame_idx |

Không tạo file production mới. Không sửa code trong thư mục audit của assistant.

## Task 1 — Sửa revision tái sử dụng

**Interfaces:** Giữ nguyên `open`, `apply`, `undo`, `close`, `current` và `ExplorationView`. `expected_revision` hiện có đủ phân biệt vòng đời nếu không tái sử dụng giá trị trên cùng instance.

- [ ] **1. Thêm test sau vào file test exploration hiện có.** Imports `pytest`, exceptions và fixtures `_new_branch`, `_binding` đều đã có trong v22.

```python
def test_old_requests_cannot_affect_reopened_branch() -> None:
    branch, _, _ = _new_branch()
    old_revision = branch.current().revision
    branch.close(expected_revision=old_revision)
    branch.open(_binding(), "v", (0, 40))
    reopened = branch.current()

    with pytest.raises(ExplorationConflict, match="revision"):
        branch.apply(
            expected_revision=old_revision,
            event_version="events-1",
            scoring_revision="scores-1",
            action="reject",
            event_index=0,
            interval=(10, 20),
        )
    with pytest.raises(ExplorationConflict, match="revision"):
        branch.undo(
            expected_revision=old_revision,
            event_version="events-1",
            scoring_revision="scores-1",
        )
    with pytest.raises(ExplorationConflict, match="revision"):
        branch.close(expected_revision=old_revision)

    assert reopened.revision > old_revision
    assert branch.current() == reopened
    branch.close(expected_revision=reopened.revision)
    with pytest.raises(ExplorationUnavailable):
        branch.current()
```

- [ ] **2. Chạy đúng test này trước sửa.** Expected: FAIL vì request cũ không bị từ chối.

```bash
uv run pytest tests/orchestration/test_temporal_exploration.py::test_old_requests_cannot_affect_reopened_branch -q
```

- [ ] **3. Sửa tối thiểu trong `TemporalExploration`.** Giữ `self._revision = 0` ở constructor. Trong `open()`, sau khi scoring/evaluation thành công và trước `_make_view`, tính revision tiếp theo; thay hai chỗ hardcode1. Trong `close()`, bỏ dòng reset revision0, giữ nguyên phần giải phóng state/history.

```python
# open(): immediately before building the new view
revision = self._revision + 1

# In the existing _make_view(...) call:
revision=revision,

# In the existing publication block, after evaluation succeeds:
self._revision = revision

# close(): remove only self._revision = 0
# Keep clearing _binding, _video, _decoder_config, _conditions,
# _history and _view exactly as before.
```

Không tăng revision trước scoring để tránh publish một phần khi open thất bại. `apply`/`undo` giữ nguyên. Lock hiện có đủ cho thay đổi này.

- [ ] **4. Chạy file exploration một lần sau sửa.** Đây là regression check cho code vừa đụng; không chạy toàn repository chỉ để hoàn thành bản sửa nhỏ.

```bash
uv run pytest tests/orchestration/test_temporal_exploration.py -q
```

Expected: test mới và các test hiện có đều pass. Nếu lỗi do dependencies môi trường, dùng environment của dự án đã chạy v22; không đổi dependencies production để phục vụ bản review.

## Task 2 — Chỉnh handoff, khóa đúng ranh giới tích hợp

**File:** `docs/vbs/scoped-feedback-handoff.md`.

- [ ] **1. Thay câu open luôn bắt đầu revision1 bằng đoạn sau.**

> The first successful open starts at revision 1. Later opens on the same instance continue increasing the revision; close does not reset it. Requests from a closed lifecycle must not mutate or close a reopened branch.

- [ ] **2. Thay câu “frame_idx is the competition coordinate” bằng đoạn sau.**

> `frame_idx` preserves the original HCMAI frame coordinate as metadata. VBS/DRES submissions use the organizer media-ID mapping and the selected `timestamp_ms`, with equal start/end under the agreed point-answer contract. Never derive the submission timestamp from `frame_idx`.

- [ ] **3. Thêm một dòng về handle vào mục session hiện có.**

> When the integration layer creates a new exploration instance, issue a fresh handle and retire the old handle. Do not bind delayed requests from an old handle to the current instance. Responses are matched by handle and revision.

Đây là rule cho transport owner khi nối UI, không phải yêu cầu xây Redis, distributed lock hay registry mới ở bản sửa này. Một process/in-memory theo scope demo đã thống nhất vẫn phù hợp; không nhận claim multi-worker support khi chưa có routing.

- [ ] **4. Review diff và commit chung hai task.** Không viết test riêng cho Markdown.

```bash
git diff --check
git diff --stat
git add src/hcmai/orchestration/workflows/temporal_exploration.py tests/orchestration/test_temporal_exploration.py docs/vbs/scoped-feedback-handoff.md
git commit -m "fix(exploration): reject stale requests across reopen"
```

## Sau bản sửa — Một walkthrough, rồi thu bằng chứng nghiên cứu

Bước này thực hiện khi teammate đã nối transport/controls. Không nằm trong điều kiện để commit bản sửa core, và không cần một bộ E2E automation mới.

- [ ] Chạy **một query** có ba event trên UI: mở video → confirm A → reject khoảng cho C → xem path đổi → undo → trở về global results → chọn candidate qua dialog. Quan sát timestamp và điều kiện còn đúng. Lưu một video ngắn hoặc chuỗi screenshot phục vụ demo/paper.
- [ ] Nếu luồng trên chạy được, chuyển sang B–C theo protocol đã có ở `docs/vbs/scoped-feedback-evaluation.md`; ghi completion/time và n thực tế. Không dùng query vừa rehearsal như một target chưa biết đáp án.

Walkthrough xác nhận tính năng dùng được; không chứng minh giảm công sức. B–C phục vụ claim nghiên cứu: cùng local-search tools, B nhập lại điều kiện, C giữ feedback. Không cần thêm model, benchmark sweep hoặc test framework trước khi thu dữ liệu này. Nếu chưa có human results, paper chỉ mô tả cơ chế và demonstration, không claim hiệu quả đã được chứng minh.

## Điểm dừng

Bản sửa này xong khi một test regression mới cùng file exploration pass, handoff hết mâu thuẫn và diff chỉ gồm ba file đã nêu. Không tiếp tục hardening theo rủi ro giả định. UI integration còn lại theo ownership của teammate; nghiên cứu tiếp theo tập trung bằng chứng tương tác và cách viết contribution.

**Self-review:** Không đổi schema/public signatures; không phát sinh service; chỉ một test mới cho lỗi tái hiện; tách core fix khỏi UI integration và human evaluation. Đây là kế hoạch cho người dùng thực hiện, chưa phải code đã được sửa.
