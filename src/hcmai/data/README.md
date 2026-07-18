# Hệ thống dữ liệu

Thư viện này biến dataset AIC 2025 S1 thành một danh sách frame chuẩn để các
thành viên khác dùng ngay. Thư viện không tải, sửa hoặc xóa dataset nguồn.

## Đầu ra chung

Mỗi dòng là một `FrameRecord`:

| Trường | Ý nghĩa |
|---|---|
| `frame_id` | ID duy nhất của frame |
| `video_id` | Video chứa frame |
| `frame_idx` | Chỉ số frame chính thức để nộp kết quả |
| `timestamp_ms` | Thời điểm của frame trong video |
| `image_path` | Đường dẫn đến ảnh gốc |
| `thumbnail_path` | Đường dẫn đến ảnh xem trước |
| `width`, `height` | Kích thước ảnh gốc |

Các output còn lại:

```text
data/aic/thumbnails/                       Ảnh xem trước
data/aic/reports/corpus_inventory.json     Thống kê dataset
data/aic/reports/extraction_report.json    Kết quả ingest
data/aic/reports/mapping_collisions.csv    Các mapping bị trùng
data/aic/reports/validation_report.json    Kết quả kiểm tra cuối
data/aic/reports/audit_samples.csv         Mẫu để kiểm tra thủ công
data/aic/checksums.sha256                   Hash kiểm tra file
```

`metadata/shards/` chỉ dùng để resume ingestion. Các bên khác không cần đọc.
Ảnh gốc không được copy sang output; `image_path` trỏ thẳng tới dataset nguồn.

Theo file phân công, đầu ra bàn giao cho từng bên là:

| Bên nhận | Dữ liệu cần nhận |
|---|---|
| Fuvo — AI Engineer 1 | Fixture 100 frame, full `frames.parquet`, `frame_id` và `image_path` để tạo embedding/index |
| Khầy — AI Engineer 2 | Fixture, `FrameStore`, ảnh gốc và thumbnail để caption, OCR và rerank |
| Cr7 — Software Engineer | `FrameStore` để lấy metadata/đường dẫn ảnh cho API và `SearchEngine` |
| Tech Lead | Inventory, extraction report, validation report, audit và checksum |

## Thuật toán mapping

Quy tắc rất ngắn gọn:

1. `n` trong CSV trỏ tới tên ảnh dạng số, ví dụ `n=3` trỏ tới `003.jpg`.
2. `frame_idx` lấy nguyên từ CSV Kaggle. Không tính lại bằng timestamp hoặc FPS.
3. Nếu nhiều ảnh có cùng `frame_idx`, giữ ảnh có `n` nhỏ nhất. Các ảnh còn lại
   vẫn được giữ nguyên và được ghi vào `mapping_collisions.csv`.
4. Tạo ID ổn định theo công thức
   `frame_id = {video_id}_{frame_idx:08d}`.
5. `timestamp_ms = round(pts_time * 1000)` chỉ dùng để xem và tìm theo thời gian.

Ví dụ: video `L21_V001`, `frame_idx=90` sẽ có
`frame_id="L21_V001_00000090"`.

## Chuẩn bị dữ liệu

Chạy fixture 100 frame trước:

```python
from hcmai.data import prepare_dataset

frames_path = prepare_dataset(
    dataset_root="dataset",
    output_root="data/aic2025_fixture",
    dataset_version="aic2025_s1_v2",
    limit=100,
)
```

Khi fixture chạy ổn, bỏ `limit` để ingest toàn bộ dataset:

```python
frames_path = prepare_dataset(
    dataset_root="dataset",
    output_root="data/aic2025",
    dataset_version="aic2025_s1_v2",
)
```

Hàm trả về đường dẫn đến `frames.parquet`. Chạy lại cùng cấu hình sẽ resume các
video đã hoàn thành.

## Cách đọc dữ liệu chung

Khởi tạo `FrameStore` đúng một lần khi chương trình bắt đầu:

```python
from hcmai.data import FrameStore

store = FrameStore("data/aic2025/metadata/frames.parquet")
```

Lấy một frame:

```python
frame = store.get("L21_V001_00000090")
print(frame.video_id, frame.frame_idx, frame.image_path)
```

Lấy nhiều frame, giữ nguyên thứ tự đầu vào:

```python
frames = store.get_many([
    "L21_V001_00000090",
    "L21_V002_00000120",
])
```

Lấy các frame lân cận trong cùng video:

```python
neighbors = store.get_neighbors(
    "L21_V001_00000090",
    window_ms=5_000,
)
```

Lọc frame theo video hoặc thời gian:

```python
from hcmai.common.schemas.search import SearchFilters

frame_ids = store.filter_frame_ids(
    SearchFilters(
        video_ids=["L21_V001"],
        start_time_ms=10_000,
        end_time_ms=30_000,
    )
)
```

`min_score` không được xử lý tại đây vì score thuộc tầng retrieval.

## Fuvo — AI Engineer 1

Đầu vào cần dùng: toàn bộ `frame_id` và `image_path` để tạo embedding, sau đó
đối chiếu kết quả FAISS về đúng frame.

```python
from hcmai.data import FrameStore

store = FrameStore("data/aic2025/metadata/frames.parquet")
frame_ids = store.filter_frame_ids(None)

for start in range(0, len(frame_ids), 64):
    frames = store.get_many(frame_ids[start:start + 64])
    image_paths = [frame.image_path for frame in frames]
    batch_ids = [frame.frame_id for frame in frames]
    # Encoder xử lý image_paths và lưu mapping theo batch_ids.
```

Khi DenseRetriever nhận `SearchFilters`, dùng
`store.filter_frame_ids(filters)` để lấy tập ID được phép trả về.

## Khầy — AI Engineer 2

Caption/OCR cần toàn bộ ảnh thì dùng cách chia batch giống Fuvo. Reranker đã có
danh sách candidate thì chỉ cần lấy đúng các ảnh đó:

```python
candidate_ids = [candidate.frame_id for candidate in candidates]
frames = store.get_many(candidate_ids)
image_paths = [frame.image_path for frame in frames]
```

Nếu cần thêm ngữ cảnh trước và sau một candidate:

```python
context_frames = store.get_neighbors(
    candidate_ids[0],
    window_ms=5_000,
    include_self=True,
)
```

## Cr7 — Software Engineer

Backend khởi tạo một `FrameStore` khi startup và tái sử dụng cho mọi request.

```python
from hcmai.data import FrameStore

frame_store = FrameStore("data/aic2025/metadata/frames.parquet")
```

Với endpoint metadata, thumbnail hoặc ảnh gốc:

```python
try:
    frame = frame_store.get(frame_id)
except KeyError:
    # Trả HTTP 404.
    ...

metadata = frame.model_dump()
thumbnail_path = frame.thumbnail_path
image_path = frame.image_path
```

Truyền trực tiếp `frame_store` vào `SearchEngine`. Khi materialize kết quả,
`SearchEngine` hiện tại sẽ tự gọi `frame_store.get(candidate.frame_id)`.

## Tech Lead và kiểm tra bàn giao

Chạy validation mà không ingest lại:

```python
from hcmai.data import validate_dataset

report = validate_dataset(
    dataset_root="dataset",
    output_root="data/aic2025",
    dataset_version="aic2025_s1_v2",
    deep=True,
)

if not report["valid"]:
    print(report["errors"])
```

Chỉ bàn giao khi `report["valid"]` là `True`, không có collision chưa giải thích,
đường dẫn ảnh hợp lệ và số lượng frame khớp extraction report.

## Lệnh CLI

Có thể cấu hình bằng `HCMAI_DATASET_ROOT`, `HCMAI_DATA_ROOT` và
`HCMAI_DATASET_VERSION`:

```bash
PYTHONPATH=src python scripts/prepare_data.py --limit 100
PYTHONPATH=src python scripts/prepare_data.py
PYTHONPATH=src python scripts/prepare_data.py --validate-only
```
