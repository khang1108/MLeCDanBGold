# Review nhánh TRAKE `origin/feat/dense-retrieval`

## 1. Phạm vi review

- Nhánh được review: `origin/feat/dense-retrieval`
- Commit đầu nhánh tại thời điểm review: `96785f2`
- Nhánh đích dự kiến: `main` tại `7e1a4de`
- Merge base: `c080139` — S2-T05 concurrent modality retrieval
- Ngày review: 2026-08-06
- Quy mô diff: 26 file, khoảng 1.058 dòng thêm và 45 dòng xóa

Review này chỉ đánh giá implementation và khả năng tích hợp. Không thay đổi
thuật toán hoặc source code thuộc sở hữu của TRAKE teammate.

## 2. Kết luận

**Chưa nên merge nhánh vào `main` ngay.**

Nhánh đã xây dựng được baseline TRAKE chạy từ HTTP request đến ranked
submission, đồng thời lõi monotonic DP Top-1 khớp exhaustive oracle trên các
fixture nhỏ. Tuy nhiên còn ba merge blocker:

1. Rescoring gọi `index.search(..., len(mapping))`, tức vẫn search toàn corpus.
2. Structured parser chưa được wire vào `TRAKEPipeline` ở composition root;
   query không có `events` và không dùng dấu `|`/newline sẽ bị từ chối.
3. Integration test mới treo tại `TestClient.__enter__`, khiến full suite không
   thể hoàn tất trong môi trường repository hiện tại.

Nhánh cũng tách khỏi `main` trước S2-T06/T07 và xung đột thực tế với stack
S2-T08–T11 tại các shared file. Nên rebase sau khi shared infrastructure mới
được merge.

## 3. Nghiệp vụ đã triển khai

### 3.1 Unified API và task dispatch

Luồng online hiện tại:

```text
POST /api/v1/search
  -> TaskRequest discriminator
  -> SearchService
  -> PipelineRegistry
  -> TRAKEPipeline
  -> TRAKEResponse
```

Router chuyển từ `SearchRequest/SearchResponse` sang union
`TaskRequest/TaskResponse`. `TRAKEPipeline` được đăng ký mặc định và health có
thể báo capability `trake`.

### 3.2 Event parsing

`TRAKEPipeline` chọn ordered events theo thứ tự:

1. dùng `request.events` nếu caller cung cấp;
2. nếu không có parser, tách query bằng `|` hoặc newline;
3. nếu có parser, gọi structured provider để tách atomic visual events, giữ
   thứ tự và dịch sang tiếng Anh;
4. khi provider lỗi, fallback về delimiter split nếu query cho phép.

Nhánh bổ sung:

- `TRAKEParseInferenceRequest`;
- `TRAKEParseResponse`;
- `LLMService.parse_trake()`;
- HTTP inference endpoint `POST /v1/trake/parse`;
- local và remote parser adapters.

### 3.3 Video shortlisting và rescoring

Với mỗi event, pipeline:

1. retrieve Top-`candidate_count` frames;
2. hợp các `frame_id` tìm được;
3. ánh xạ các frame sang candidate `video_id` bằng canonical mapping;
4. batch encode events bằng visual encoder;
5. tạo ma trận similarity `scores[event, frame]`;
6. tách ma trận thành từng candidate video.

Canonical `frame_id`, `video_id`, `frame_idx` và `timestamp_ms` đều được đọc
từ index mapping, không suy diễn từ FPS, timestamp hay filename.

### 3.4 Monotonic DP alignment

Với mỗi video, aligner tìm:

```text
t1 < t2 < ... < tN
```

và tối ưu:

```text
sum(event-frame similarity)
  - lambda_gap * (timestamp_last - timestamp_first)
```

Implementation dùng running prefix maximum và có độ phức tạp `O(N × M)`, với
`N` là số event và `M` là số frame của video. Kết quả có đúng một canonical
frame cho mỗi event, cùng video và đúng thứ tự.

### 3.5 Ranked alternatives và submission

Ranking ưu tiên best path của mỗi video trước khi đưa second path của một
video vào kết quả. CSV exporter tạo row:

```text
<video_name>,<frame_1>,...,<frame_N>
```

Exporter không ghi header, bỏ `.mp4` và từ chối batch trộn số lượng event.

## 4. Findings trước merge

### B01 — Blocker: full-corpus vector search sau shortlist

**SOURCE:** `src/hcmai/agents/trake/shortlist.py`, hàm
`event_video_scores()`.

Implementation gọi:

```python
scores, positions = index.search(event_vectors, len(mapping))
```

Sau đó còn tạo một ma trận `full` cùng kích thước. Với `E` events và `F`
frames, riêng ba mảng chính chiếm xấp xỉ:

```text
scores:    E × F × 4 bytes
positions: E × F × 8 bytes
full:      E × F × 4 bytes
```

Tổng gần `16 × E × F` bytes, chưa tính FAISS, vectors và mapping. Shortlist chỉ
giảm số video vào DP, không giảm corpus search.

**Yêu cầu trước merge:** dùng vectors/postings của S2-T09 để rescore đúng tập
frame thuộc candidate videos; không gọi search với `top_k=index.ntotal`.

### B02 — Blocker: production parser chưa được wire

**SOURCE:** `src/hcmai/orchestration/pipeline.py` đăng ký:

```python
TRAKEPipeline(self.retrieval, self.config)
```

Không có `TrakeQueryParser(self.llm.parse_trake)`. Vì vậy inference endpoint và
parser adapters không được dùng bởi default online pipeline.

Production chỉ chạy nếu request truyền sẵn `events`, hoặc query dùng `|` hay
newline. Delimiter regex không hỗ trợ `->`; query dạng `E1 -> E2 -> E3` không
có trường `events` sẽ trả HTTP 422.

**Yêu cầu trước merge:** wire parser tại composition root, xác nhận official
delimiter và thêm integration test không truyền sẵn `events`.

### B03 — Blocker: integration test treo

**SOURCE:** `tests/integration/test_trake_api.py` dùng `with TestClient(app)`.

Trong môi trường repository hiện tại, test treo ngay tại
`TestClient.__enter__`. Các API tests hiện hữu dùng `httpx.ASGITransport` để
tránh vấn đề lifecycle/deadlock này.

API smoke bằng `ASGITransport` vẫn trả HTTP 200 và canonical path `[10, 20]`,
do đó lỗi nằm ở test harness, không phải route cơ bản.

**Yêu cầu trước merge:** chuyển integration test sang helper
`httpx.ASGITransport` của repository và chạy lại full suite.

### H01 — High: event retrieval chạy tuần tự và dùng mặc định KIS

Code gọi `retrieval.search(event, top_k)` trong loop. Điều này:

- không dùng `search_batch()`;
- có thể encode lặp events;
- không overlap retrieval giữa events;
- không truyền `query_type`, nên mặc định thành `TaskType.KIS`.

**Yêu cầu:** dùng batch API và truyền rõ `TaskType.TRAKE`.

### H02 — High: alternatives chưa phải global k-best

`align_video(..., paths=k)` chỉ lấy best path cho `k` endpoint cuối khác nhau.
Nó có thể bỏ qua path tốt thứ hai nếu path đó kết thúc cùng endpoint với best
path.

Ví dụ đã kiểm tra:

```text
global best:   score 20, path (0, 2)
global second: score 19, path (1, 2)
```

Implementation trả:

```text
score 20, path (0, 2)
score 10, path (0, 1)
```

**Yêu cầu:** triển khai true k-best, hoặc đổi tên/contract và không mô tả output
là global k-best.

### H03 — High: branch base cũ và xung đột shared infrastructure

Nhánh tách từ S2-T05 nên không có S2-T06/T07 trong ancestry. Merge mô phỏng
vào `main` hiện tại không có textual conflict, nhưng merge với
`feat/s2-observability` có conflict thực tế tại ít nhất:

- `src/hcmai/orchestration/pipelines/kis.py`;
- `src/hcmai/retriever/pipeline.py`.

Các file LLM, schemas và `SearchService` cũng được cả hai phía sửa và cần
semantic review.

**Yêu cầu:** merge S2-T08–T11 trước, sau đó rebase TRAKE lên `main` mới.

### H04 — High: candidate-video selection chưa thưởng complete coverage

Candidate videos là union của video xuất hiện trong Top-k của bất kỳ event
nào. Chưa có explicit score cho:

- số event được cover;
- weakest-event score;
- complete-sequence coverage.

Video mạnh ở một event vẫn có thể lọt shortlist dù yếu ở các event còn lại.

### M01 — Medium: gap penalty và resource limits hard-code

`lambda_gap=1e-5` chưa nằm trong config. Cũng chưa có config riêng cho:

- candidate video count;
- max frames/video;
- alignment deadline;
- k-best diversity;
- refinement depth.

### M02 — Medium: `ValueError` bị map quá rộng thành HTTP 422

`SearchService.search()` bắt mọi `ValueError` từ task pipeline và chuyển thành
`UnsupportedSearchTaskError`. Lỗi shape, mapping hoặc numerical nội bộ có thể
bị báo sai thành invalid user request.

**Yêu cầu:** dùng typed request-validation exception; để lỗi nội bộ giữ đúng
failure category.

### M03 — Medium: shared retrieval lộ concrete `DenseIndex`

`RetrievalService.visual_index` trả trực tiếp `DenseIndex`. TRAKE vì vậy phụ
thuộc vào `.mapping`, `.index` và `.search()` implementation cụ thể.

**Yêu cầu:** cân nhắc public candidate-local scoring contract thay vì expose
private index internals. Đây là shared-interface change và phải phối hợp với
KIS/VQA workstream.

### M04 — Medium: provider error detail có thể đi ra log/API

Parser lấy tối đa 160 ký tự từ `str(error)` và đưa vào `TrakeParserError`.
Provider response hoặc deployment detail có thể xuất hiện trong warning/API.

**Yêu cầu:** chỉ truyền bounded safe categories như `timeout`, `unavailable`,
`contract_error` hoặc `invalid_response`.

## 5. Phần chưa được triển khai

Nhánh hiện là keyframe baseline và chưa có:

- original-video frame refinement;
- shot-transition penalty;
- multimodal alignment score matrix;
- true global k-best paths;
- request-scoped stage trace theo S2-T11;
- request deadline propagation;
- degraded component warnings trong `TRAKEResponse`;
- benchmark corpus thực;
- official TRAKE metrics;
- duplicate-path measurement;
- reproducible experiment record với `runs/.../metrics.json`.

Các mục trên không nhất thiết đều phải nằm trong cùng merge, nhưng phải được
ghi rõ là limitation; không nên gọi pipeline hiện tại là competition-ready.

## 6. Validation đã thực hiện

| Kiểm tra | Kết quả |
|---|---:|
| Merge mô phỏng vào `main` | Auto-merge thành công, không textual conflict |
| Focused TRAKE/task-router unit tests | 18 passed |
| Broader suite, bỏ integration TRAKE bị treo | 64 passed |
| Pyright trên TRAKE và shared files liên quan | 0 errors, 0 warnings |
| Randomized Top-1 DP vs exhaustive oracle | 450/450 cases matched |
| API smoke qua `httpx.ASGITransport` | HTTP 200, path `[10, 20]` |
| `tests/integration/test_trake_api.py` | Treo tại test đầu |
| Full suite | Không hoàn tất do integration test treo |

Không chạy corpus thật, checkpoint thật hoặc remote provider thật. Không có
accuracy/latency claim cho competition corpus.

## 7. Shared-contract impact

Nhánh thay đổi các public/shared interfaces sau:

1. `/api/v1/search` nhận và trả `TaskRequest/TaskResponse` union.
2. `TaskPipeline.execute()` đổi từ KIS-only contract sang task union.
3. `SearchService.search()` đổi return type sang task union.
4. `RetrievalService` bổ sung `visual_index` và `_retriever_for()`.
5. `LLMService` bổ sung `parse_trake()`.
6. Inference server bổ sung `/v1/trake/parse`.
7. Common schemas bổ sung TRAKE parsing contracts.

Các thay đổi schema/parser chủ yếu additive. Tuy nhiên router response model,
pipeline protocol, `SearchService` return type và direct index exposure ảnh
hưởng KIS/VQA callers, OpenAPI và các branch S2-T08–T11; cần semantic merge,
không chỉ nhận auto-merge result.

## 8. Checklist bắt buộc trước merge

- [ ] Merge S2-T08–T11 vào `main`.
- [ ] Rebase TRAKE branch lên `main` mới.
- [ ] Resolve shared-contract changes với KIS/VQA owner.
- [ ] Wire structured parser tại composition root.
- [ ] Xác nhận và test official TRAKE delimiter/input format.
- [ ] Dùng `search_batch(..., query_type=TaskType.TRAKE)`.
- [ ] Bỏ full-corpus `index.search(..., len(mapping))`.
- [ ] Rescore candidate-local frames qua vectors/postings của S2-T09.
- [ ] Đưa gap penalty và resource limits vào config.
- [ ] Sửa true k-best hoặc thu hẹp claim/interface.
- [ ] Dùng typed input exceptions thay vì catch mọi `ValueError`.
- [ ] Đổi integration tests sang `httpx.ASGITransport`.
- [ ] Commit randomized exhaustive-oracle regression tests.
- [ ] Chạy full suite thành công.
- [ ] Chạy benchmark có corpus/config/version rõ ràng trước performance claim.

## 9. Merge recommendation

Thứ tự đề xuất:

```text
merge S2-T08 -> S2-T09 -> S2-T10 -> S2-T11
  -> teammate rebase feat/dense-retrieval
  -> sửa B01/B02/B03 và shared-contract conflicts
  -> full tests + API smoke + benchmark
  -> review lại
  -> merge TRAKE vào main
```

Trạng thái review hiện tại: **Request changes / not merge-ready**.
