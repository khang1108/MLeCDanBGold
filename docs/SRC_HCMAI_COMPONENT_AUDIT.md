# HCMAI 2026 — Đánh giá trạng thái các component trong `src/hcmai`

**Ngày đánh giá:** 2026-08-01  
**Phạm vi:** `src/hcmai`  
**Mục tiêu:** Xác định component nào đã hoàn chỉnh ở mức code, component nào
đã tích hợp end-to-end, component nào có artifact/benchmark thật, và thứ tự
công việc cần ưu tiên tiếp theo.

## 1. Kết luận điều hành

Dự án không còn ở mức demo rỗng. Nhiều component đã có implementation và test
tốt, nhưng hệ thống đang ở trạng thái:

> **Component đã làm khá nhiều, hệ thống tổng thể chưa khép kín.**

Vấn đề hiện tại không phải thiếu thêm model hoặc feature. Các blocker chính là:

1. Default pipeline không bootstrap được các artifact hiện có.
2. Chưa có evaluator bám đúng competition metric.
3. Chưa có `runs/<experiment>/metrics.json` để quyết định cần tối ưu retrieval,
   fusion hay reranking.
4. Public pipeline mới hỗ trợ KIS và working VKIS; VQA, TRAKE và KISC chưa có
   đường chạy end-to-end.
5. Latency, fallback và GPU resource usage chưa được chứng minh bằng benchmark.

Việc cần làm tiếp theo chưa phải tối ưu reranker. Ưu tiên đúng là:

> Khôi phục một baseline visual-only chạy end-to-end, hoàn thiện evaluator,
> tạo run có metric chính thức, rồi mới quyết định tối ưu component nào.

## 2. Bằng chứng tổng quát

Kết quả audit:

- Khoảng 86 file Python, tổng cộng 9.160 dòng.
- **149/149 test pass** trong 13,19 giây.
- Statement coverage toàn package: **80%**.
- Toàn bộ package compile thành công.
- `pyright` phát hiện 5 lỗi, tập trung ở transcripts.
- Đã sinh visual embedding cho **177.321 frame**.
- Đã caption đủ **177.321 frame**, manifest ghi nhận 0 failure.
- Visual FAISS index có 177.321 vector, dimension 768 và mapping hợp lệ.
- Không có thư mục `runs/` hoặc `metrics.json`.

Khi gọi trực tiếp `load_default_engine()` với config mặc định hiện tại:

```text
frame_store       = False
retriever         = False
reranker          = False
caption evidence  = True
query suggestions = True
```

Các startup message:

```text
Metadata not available at data/metadata/frames.parquet
Could not load artifacts/indexes/visual/dense.index
OCR artifact not available
ASR artifact not available
```

Điều này có nghĩa optional query suggestions được cấu hình, nhưng search cốt
lõi chưa sẵn sàng.

## 3. Ma trận trạng thái component

Quy ước:

- ✅ Có implementation và test tương đối tốt.
- 🟡 Có implementation nhưng chưa tích hợp hoặc chưa benchmark.
- 🔴 Thiếu đường chạy cần thiết cho cuộc thi.

| Component | Code/test | Artifact/integration | Đánh giá |
| --- | --- | --- | --- |
| Schemas và config | ✅ | Thiếu contract TRAKE/evaluation chính thức | 🟡 |
| Canonical `FrameStore` | ✅ | `frames.parquet` hiện không tồn tại | 🟡 |
| Frame preparation | ✅ | Chưa có artifact đầu vào hiện tại để tái chạy | 🟡 |
| Visual embedding | ✅ | Có đủ 177.321 vector | 🟡 |
| FAISS exact index | ✅ | Index hợp lệ nhưng sai filename runtime mong đợi | 🟡 |
| Caption enrichment | ✅ | Có artifact đầy đủ 177.321 frame | ✅/🟡 |
| OCR enrichment | ✅ với fake backend | Chưa có artifact OCR | 🟡 |
| ASR/transcripts | 🟡 | Chưa frame-align, chưa có artifact/index | 🔴 |
| Text index caption/OCR/ASR | ✅ implementation | Chưa có text index nào | 🔴 |
| Weighted RRF fusion | ✅ | Chưa chạy được trong default pipeline | 🟡 |
| Multimodal reranker | ✅ unit-level | Chưa có accuracy/latency run, thiếu fallback | 🟡 |
| Search orchestration | ✅ với fake components | Chỉ KIS/VKIS, chưa temporal refinement | 🟡 |
| Public API | ✅ với injected fake engine | Default startup không ready | 🟡/🔴 |
| KISC | Có agent, resolver, session | Router chưa mount | 🔴 |
| VQA | Có private inference endpoint | Chưa có retrieve → answer pipeline | 🔴 |
| TRAKE | Chỉ có enum | Chưa có schema/alignment/output | 🔴 |
| Evaluation | Có dense recall evaluator | 0% coverage, thiếu official score/MRR | 🔴 |
| Submission | Có submit một frame | Chưa có CSV/ZIP, VQA hay TRAKE export | 🔴 |
| Query suggestions | ✅ | Đã mount vào public API | ✅ nhưng P2 |
| Private LLM service | ✅ boundaries | Chưa chứng minh chạy đồng thời trên GPU | 🟡 |

## 4. Đánh giá chi tiết

### 4.1 Schemas và canonical mapping

[`src/hcmai/common/schemas`](../src/hcmai/common/schemas) là một trong những
phần ổn định nhất:

- Có `FrameRecord`, `RetrievalCandidate` và `SearchRequest/Response`.
- Giữ score riêng cho visual, caption, OCR, ASR, fusion và reranker.
- Bảo toàn canonical `frame_id → video_id → frame_idx`.
- Pydantic reject unknown fields.
- Feedback accepted/rejected được kiểm tra không trùng nhau.

[`FrameStore`](../src/hcmai/data/loader.py) cung cấp:

- Lookup O(1) theo `frame_id`.
- Temporal neighbors trong cùng video.
- Kiểm tra duplicate `frame_id`.
- Materialization bằng official mapping.
- Kiểm tra cặp submission `video_id,frame_idx`.

Đây là foundation nên giữ ổn định, chưa cần refactor hoặc tối ưu lại.

Khoảng trống:

- Chưa có authoritative TRAKE request/response.
- VQA schema mới mô tả hỏi một frame, chưa mô tả ranked competition row.
- Evaluation schema chưa biểu diễn đầy đủ interval `[s,e]`, TRAKE event
  sequence và Q&A answer comparison.

### 4.2 Frame preparation

[`data/prepare.py`](../src/hcmai/data/prepare.py) đọc official mapping, nối ảnh
và tạo canonical `frames.parquet`. Implementation có validation cho:

- Required mapping columns.
- Numeric and finite values.
- Non-negative `frame_idx` và timestamp.
- Duplicate keyframe order.
- Missing keyframe image.
- Duplicate canonical `frame_id`.
- Parquet schema và row count sau serialization.

Tuy nhiên, file mà config yêu cầu:

```text
data/metadata/frames.parquet
```

hiện không tồn tại. Đây là blocker vì:

- API không materialize được `video_id/frame_idx`.
- Reranker không tra được canonical `image_path`.
- Frame, submission và temporal-neighbor endpoints không hoạt động.
- Bootstrap không tạo được `FrameStore`.

Caption manifest chứng minh file này từng tồn tại khi caption được chạy. Cần
khôi phục nó từ official mapping; không được suy `frame_idx` từ artifact
embedding hoặc timestamp.

[`common/utils/data.py`](../src/hcmai/common/utils/data.py) là một
implementation mapping cũ, hiện không được import và có coverage 0%. Nó chứa
assumption khác với `data/prepare.py`, tạo hai nguồn sự thật. Nên đánh dấu
legacy và chỉ xóa sau khi baseline mới ổn định.

### 4.3 Visual embedding

[`EmbeddingPipeline`](../src/hcmai/embedding/embedding.py) đã tạo artifact thật:

- 177.321 embedding.
- Dimension 768.
- Model `google/siglip2-base-patch16-224`.
- Không có frame failure.
- Mapping không duplicate.
- Vector và mapping cùng dataset version.

Đây là tiến độ thực, không phải skeleton.

Nhưng chưa competition-ready vì:

- Không có labelled-query benchmark.
- Manifest không pin resolved checkpoint revision.
- Không có Recall, MRR hoặc Mean Top-k R-Score.
- PIL image không được đóng rõ ràng sau mỗi embedding batch, có nguy cơ gây
  memory/file-descriptor pressure khi tái chạy corpus lớn.

Không nên đổi embedding model trước khi đo baseline hiện tại.

### 4.4 FAISS exact index

Visual index hiện tại nằm tại:

```text
artifacts/indexes/visual/visual.index
```

Audit xác nhận:

```text
FAISS ntotal          = 177321
FAISS dimension       = 768
Mapping rows          = 177321
Metadata vector count = 177321
Metadata dimension    = 768
Embedding positions   = 0..177320
```

Tuy nhiên, [`DenseIndex.load()`](../src/hcmai/retriever/dense/index.py) yêu cầu:

```text
artifacts/indexes/visual/dense.index
```

Do đó default bootstrap không mở được index. Đây là artifact contract drift,
không phải lỗi chất lượng model. Cần reconcile filename/layout sau khi validation
thay vì build lại index một cách không cần thiết.

### 4.5 Caption enrichment

Caption là component hoàn chỉnh nhất sau canonical data:

- Resumable processing.
- Atomic manifest writes.
- Per-frame failure recording.
- Pinned model revision.
- Batched inference.
- Latency và throughput report.
- Full artifact 177.321 frame.

Manifest hiện có ghi nhận:

- 177.321 completed.
- 0 failed.
- Khoảng 4,4 ảnh/giây trên CPU.
- Florence-2 revision đã pin.

Đây là artifact có giá trị cao nhất để tích hợp tiếp. Sau visual baseline, bước
hợp lý là build caption BGE-M3 index.

Điểm cần sửa sau: [`caption/runner.py`](../src/hcmai/enrichment/caption/runner.py)
không đóng rõ ràng PIL image sau inference, trong khi OCR pipeline có làm việc
đó.

### 4.6 OCR enrichment

OCR pipeline đã có:

- Resume.
- Fake backend tests.
- Native Florence backend.
- Failure/evidence/report artifacts.
- Identity preservation.
- 98% coverage ở orchestration pipeline.

Tuy nhiên, native backend có coverage thấp và workspace chưa có OCR artifact.
Trạng thái đúng là:

> **Implementation-ready, experiment-not-run.**

Không nên ưu tiên OCR trước khi caption index và evaluator hoạt động. Sau đó
dùng failure analysis để xác định tỷ lệ query thực sự OCR-dependent.

### 4.7 ASR và transcripts

ASR có:

- Qwen ASR wrapper.
- Silero VAD.
- Batched speech segments.
- Pyannote diarization.
- TranscriptStore và temporal lookup.
- Resume theo từng video.

Các blocker:

- Venv hiện thiếu `av`, `silero_vad` và `pyannote.audio`.
- `pyright` báo 5 lỗi liên quan transcripts.
- Optional dependency xung đột:
  - reranking yêu cầu `transformers < 5`;
  - transcripts yêu cầu `transformers >= 5.13`.
- Chưa có bước nối transcript segment vào canonical frames để tạo
  `FrameEnrichment.asr_text`.
- Chưa có ASR frame-enrichment artifact.
- Chưa có ASR text index.

Vì vậy ASR chưa tham gia retrieval dù schemas và bootstrap đã có tên của nó.

### 4.8 Text retrieval và fusion

[`build_text_index()`](../src/hcmai/retriever/caption/retriever.py) đã làm đúng:

- Join bằng `frame_id`.
- Encode bằng BGE-M3.
- Normalize vector.
- Build exact FAISS index.
- Bảo toàn `video_id/frame_idx`.
- Dùng chung `RetrievalCandidate`.

[`RRFFusionRetriever`](../src/hcmai/retriever/fusion/rrf.py) cũng có
implementation và test tốt.

Khoảng trống:

- Chưa có caption/OCR/ASR text index.
- Tất cả fusion weights đang bằng 1 và chưa được tune.
- Bootstrap yêu cầu đủ cả ba text index; thiếu một index làm mất luôn visual
  retriever.
- Các retriever chạy tuần tự.
- Caption/OCR/ASR dùng cùng BGE encoder nhưng query bị encode lại ba lần.

Quick wins sau khi pipeline chạy:

1. Cho phép visual-only hoặc caption-only degraded mode.
2. Encode BGE query một lần rồi reuse cho các text index.
3. Chỉ bật source đã có artifact và benchmark.

### 4.9 Multimodal reranker

Reranker có:

- 99% coverage ở bounded reranker.
- Preserves candidate identity.
- Canonical image resolution.
- Deterministic ordering.
- Reject `NaN` và wrong score count.
- Native Qwen scorer có test.

Nhưng chưa competition-ready:

- Không có run chứng minh tăng điểm.
- `batch_size=1`.
- Rerank 100 candidate có thể tạo 100 HTTP request tuần tự.
- Dùng eager attention.
- `final_score` bị thay hoàn toàn bởi Qwen score.
- Thiếu timeout/fallback về RRF.
- Model revision đang `null`.

Chưa nên tối ưu reranker cho đến khi biết Candidate Recall@100. Nếu target không
nằm trong candidate pool thì reranker không thể cứu được.

### 4.10 Search orchestration

[`SearchEngine`](../src/hcmai/orchestration/search.py) đã làm được:

- Retrieve candidates.
- Optional RRF.
- Optional reranking.
- Canonical result materialization.
- Caption/OCR/ASR evidence.
- Stage logging.
- API response construction.

Nhưng còn thiếu:

- `temporal_window_ms` được config nhưng chưa được SearchEngine sử dụng.
- `query_encoding`, `fusion` và `temporal_refinement` latency luôn bằng 0.
- Không có near-duplicate suppression.
- Không có reranker fallback.
- Không có task-specific orchestration.
- TRAKE không thể dùng generic frame search hiện tại.

### 4.11 Public API và bootstrap

Public API hiện có:

```text
GET  /health
POST /api/v1/search
POST /api/v1/query-suggestions
GET  /api/v1/frames/{frame_id}
GET  /api/v1/frames/{frame_id}/image
GET  /api/v1/frames/{frame_id}/thumbnail
GET  /api/v1/frames/{frame_id}/neighbors
POST /api/v1/submit
```

`/api/v1/search` hỗ trợ:

| Task | Trạng thái |
| --- | --- |
| KIS | Có frame-search pipeline |
| VKIS description | Dùng cùng frame-search pipeline |
| VQA | Trả `501` |
| TRAKE | Trả `501` |
| KISC | Trả `422` |

149 test pass phần lớn nhờ injected fake `SearchEngine`. Bootstrap thật chỉ có
34% coverage, nên artifact mismatch chưa bị integration test phát hiện.

Default runtime hiện không có frame store hoặc retriever. Đây là blocker số một.

### 4.12 KISC

KISC có hai hướng song song:

1. Stateless `KISCAgent` với browser-owned history.
2. Stateful in-memory `KiscSessionManager`.

Resolver có validation tốt và feedback được giữ ngoài model. Tuy nhiên:

- `create_kisc_router()` không được export.
- Router không được mount trong [`app.py`](../src/hcmai/app.py).
- `default_kisc_agent()` không được gọi.
- Router KISC có coverage 0%.
- Test chủ động xác nhận KISC endpoint trả 404.
- Agent gọi resolver cho mọi turn, kể cả first-turn standalone hoặc
  feedback-only.

Đây là prototype có unit logic, chưa phải feature tích hợp.

### 4.13 VQA

Private LLM service có `/v1/vqa` và model trả answer tối đa 100 ký tự.

Competition Q&A cần:

```text
query
  → retrieve đúng video/frame
  → answer trên candidates
  → rank video,frame,answer rows
```

Hiện mới có:

```text
frame đã biết
  → hỏi model
  → trả answer
```

Thiếu:

- Candidate retrieval cho VQA.
- Answer cho nhiều candidate.
- Joint frame-answer ranking.
- Exact/normalized answer evaluation.
- Public API integration.
- Submission export.

### 4.14 TRAKE

TRAKE hiện chỉ xuất hiện trong `TaskType.TRAKE`; public dispatcher trả `501`.

Chưa có:

- Event-sequence schema.
- Per-event retrieval.
- Same-video constraint.
- Chronological ordering.
- Joint temporal alignment.
- Exactly-N output rows.
- TRAKE metric.
- Submission exporter.

Đây là khoảng trống lớn nhất nếu 2026 tiếp tục task semantics của 2025.

### 4.15 Evaluation và submission

[`RetrievalBenchmark`](../src/hcmai/retriever/evaluation/benchmark.py) có code
nhưng:

- Coverage 0%.
- Không có focused test.
- Chỉ đo Recall `{1,5,10,100}`.
- Không đo MRR.
- Không đo Mean Top-k R-Score `{1,5,20,50,100}`.
- Không đánh giá reranker hoặc full SearchEngine.
- Không hỗ trợ VQA hay TRAKE.
- Không có `runs/`.

Submission hiện chỉ format một dòng:

```text
video_id,frame_idx
```

Chưa có:

- Per-query CSV.
- UTF-8/no-header validation.
- Maximum 100 rows.
- VQA answer column.
- Exactly-N TRAKE columns.
- `submission/*.csv` ZIP.

Đây là lý do hệ thống chưa biết nên tối ưu gì: chưa có scoreboard nội bộ.

### 4.16 Query suggestions

Query suggestions có:

- Authoritative schemas.
- GPU inference và OpenAI-compatible provider.
- Configured timeout.
- Response parsing và identity validation.
- Public route.
- Tests.

Component này tương đối hoàn chỉnh, nhưng không thuộc đường găng competition.
Không nên đầu tư thêm cho tới khi search, evaluator, VQA và TRAKE có baseline.

### 4.17 Private LLM service

Private inference service có boundary rõ cho:

- Visual/text embedding.
- Caption.
- Reranking.
- Conversation resolution.
- Query suggestions.
- VQA.
- Readiness.

Các model được load một lần trong lifespan. Tuy nhiên mặc định service có thể
warm Florence, SigLIP, BGE-M3, Qwen reranker và GLM conversation model trong
cùng process. Chưa có run chứng minh tổ hợp này vừa VRAM và đạt latency budget
trên L40/A6000.

## 5. Các blocker ưu tiên

### P0.1 — Default pipeline không chạy

Nguyên nhân:

- Thiếu `data/metadata/frames.parquet`.
- Runtime cần `dense.index`, artifact hiện tên `visual.index`.
- Caption/OCR/ASR text indexes chưa tồn tại.
- Bootstrap đang load fusion theo kiểu all-or-nothing.

### P0.2 — Chưa có competition evaluator

Không có evaluator đúng nghĩa thì không thể biết:

- Candidate recall thấp do retrieval.
- Candidate có nhưng rank thấp do fusion/reranker.
- Accuracy tốt nhưng latency không dùng được.
- Component mới thực sự cải thiện hay chỉ làm pipeline phức tạp hơn.

### P0.3 — Chưa có reproducible run

Không có:

```text
runs/<experiment>/
├── config.yaml
├── metrics.json
├── predictions.csv
├── failures.csv
└── summary.md
```

Do đó chưa có baseline để so sánh.

### P0.4 — Competition tasks chưa hoàn chỉnh

- VQA mới có model call cho frame đã biết.
- TRAKE chưa có pipeline.
- KISC chưa mount và routing chưa đúng.

## 6. Roadmap đề xuất

### Giai đoạn 1 — Khôi phục visual-only baseline

1. Khôi phục canonical `data/metadata/frames.parquet` từ official mapping.
2. Reconcile `visual.index` với `DenseIndex` artifact contract.
3. Cho bootstrap chạy visual-only khi text indexes chưa có.
4. Thêm startup integration test bằng tiny on-disk artifacts.
5. Đạt `/health.ready = true`.
6. Chạy được một query KIS thật qua public API.

Exit condition:

```text
query → SigLIP → FAISS → canonical frame → API response
```

### Giai đoạn 2 — Hoàn thành evaluator

Evaluator tối thiểu cần:

- Mean Top-k R-Score tại `{1,5,20,50,100}`.
- Recall@1/5/100.
- MRR.
- Warm P50/P95.
- Query-to-first-useful-result.
- Predictions và failures.
- Effective config và checkpoint.
- Hardware description.

Tạo run đầu tiên:

```text
runs/visual_baseline/metrics.json
```

### Giai đoạn 3 — Caption retrieval

Caption artifact đã đầy đủ nên đây là bước mở rộng đầu tiên:

1. Build BGE-M3 caption index.
2. Chạy visual-only baseline.
3. Chạy visual+caption RRF trên cùng evaluation set.
4. Chỉ giữ caption nếu official score tăng với latency chấp nhận được.

### Giai đoạn 4 — Reranker

Chỉ bắt đầu khi Candidate Recall@100 đủ tốt:

- Depth: `10/20/50/100`.
- Batch size: `1/4/8`.
- Eager so với Flash Attention.
- RRF fallback khi timeout/OOM.
- Pure Qwen score so với rank fusion Qwen+RRF.
- Warm P50/P95 và peak GPU memory.

### Giai đoạn 5 — Các task còn thiếu

Thứ tự đề xuất:

1. VQA retrieve → answer → rank pipeline.
2. TRAKE joint temporal alignment.
3. OCR nếu failure analysis cho thấy OCR-dependent miss đủ lớn.
4. ASR sau khi giải quyết dependency và frame alignment.
5. Mount KISC sau khi thêm deterministic bypass.
6. Giữ query suggestions ở trạng thái hiện tại.

## 7. Quy tắc quyết định tối ưu

```text
Candidate Recall@100 thấp
    → tối ưu embedding, caption/OCR/ASR retrieval

Candidate Recall@100 cao nhưng Recall@1/MRR thấp
    → tối ưu fusion và reranker

Accuracy tốt nhưng P95 cao
    → batching, query embedding reuse, rerank depth, local inference

KIS tốt nhưng tổng điểm thấp
    → hoàn thiện VQA/TRAKE thay vì tiếp tục vắt thêm vài phần trăm KIS
```

## 8. Kết luận

Phần cần ưu tiên ngay không phải model mới hoặc thêm feature. Đường găng là:

```text
artifact contract
    → runnable visual baseline
    → official evaluator
    → visual baseline run
    → evidence-driven optimization
```

Sau chuỗi này, dự án sẽ chuyển từ “nhiều component chạy riêng lẻ” sang một hệ
thống có scoreboard. Khi đó mỗi quyết định tối ưu retrieval, fusion, reranker,
OCR, ASR, VQA hay TRAKE đều có bằng chứng rõ ràng.

## 9. Nguồn tham chiếu

- [AI Challenge HCMC 2026](https://aichallenge.hochiminhcity.gov.vn/)
- [HCMC AI Challenge 2025 Group A](https://www.codabench.org/competitions/10187/)
- [`AGENTS.md`](../AGENTS.md)
- [`configs/baseline.yaml`](../configs/baseline.yaml)
- [`llm/config.yaml`](../llm/config.yaml)

