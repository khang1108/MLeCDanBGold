# Tiền xử lý video thành FrameStore

Pipeline luôn chạy đầy đủ:

```text
Video → Optical Flow + TransNetV2 + EfficientGEBD
      → Dynamic Gap → DINOv2 Dedup → FrameStore
```

Video được decode hai lần: lần đầu dùng chung cho các tín hiệu chọn frame,
lần sau chỉ ghi candidate JPEG. Frame đầu tiên bắt đầu từ `0`:

```text
frame_idx = round(timestamp_ms * FPS / 1000)
frame_id  = <video_id>_frame_<frame_idx đủ 9 chữ số>
```

## Cài đặt

```bash
python -m pip install -e ".[preprocessing]"
```

TransNetV2 và EfficientGEBD dùng checkout/checkpoint chính thức đặt ngoài
repository. DINOv2 Small được tải từ Hugging Face public.

## Cấu hình

Tạo `preprocessing.yaml`:

```yaml
preprocessing:
  videos_root: /mounted/aic2026/videos
  output_root: artifacts/frame_store

  transnet_repo: /models/TransNetV2
  transnet_weights: /models/TransNetV2/inference/transnetv2-weights

  efficientgebd_repo: /models/EfficientGEBD
  efficientgebd_config: /models/EfficientGEBD/model_config.yaml
  efficientgebd_checkpoint: /models/EfficientGEBD/model_best.pth

  device: cuda
  dino_model: facebook/dinov2-small
  dino_dtype: float16
  dino_batch_size: 16

  efficientgebd_sample_fps: 10
  motion_threshold: 0.012
  shot_threshold: 0.5
  event_threshold: 0.5
  minimum_gap_ms: 500
  maximum_gap_ms: 2000
  dedup_similarity: 0.985
  image_quality: 92
```

Chỉ device và DINO dtype có thể override khi deploy:

```bash
export HCMAI_PREPROCESSING_DEVICE=cuda
export HCMAI_PREPROCESSING_DINO_DTYPE=float16
```

## Chạy

```bash
PYTHONPATH=src python scripts/preprocess_videos.py \
  --config preprocessing.yaml
```

Test ít video hoặc chạy lại từ đầu:

```bash
PYTHONPATH=src python scripts/preprocess_videos.py \
  --config preprocessing.yaml \
  --limit 1 \
  --no-resume
```

## Output

```text
artifacts/frame_store/
├── frames.parquet
└── images/<group>/<video_id>/<frame_idx>.jpg
```

Các bên chỉ cần:

```text
FrameStore root: artifacts/frame_store
Metadata:        artifacts/frame_store/frames.parquet
```

Checkpoint nằm tại `artifacts/.preprocessing_work/`, không phải output query.
`image_path` trong Parquet là đường dẫn tương đối từ FrameStore root.

## Khởi tạo

```python
from hcmai.data.pipeline import DataService

data = DataService.load(
    "artifacts/frame_store/frames.parquet",
    dataset_root="artifacts/frame_store",
)
```

Service được tạo một lần. Retrieval trả về `frame_id`; không tự tách hoặc tính
lại `video_id`, `frame_idx` từ chuỗi ID.

### KIS

```python
frames = data.get_frames(retrieved_frame_ids)
rows = [(frame.video_id, frame.frame_idx) for frame in frames]
```

### Q&A / VQA

```python
frame = data.get_frame(retrieved_frame_id)
neighbors = data.neighbors(frame.frame_id, window_ms=3_000)
image_path = data.resolve_frame_asset(frame)
```

Caption, OCR và ASR được join bằng đúng `frame_id`.

### TRAKE

```python
from hcmai.common.schemas.search import SearchFilters

ids = data.filter_frame_ids(SearchFilters(
    video_ids=["L21_V001"],
    start_time_ms=10_000,
    end_time_ms=20_000,
))
frames = data.get_frames(ids)
```

FrameStore chỉ cung cấp canonical frames và temporal filtering; thuật toán
temporal ranking thuộc pipeline TRAKE riêng.

## API

- `get_frame(frame_id)`: lấy một `FrameRecord`.
- `get_frames(frame_ids)`: giữ thứ tự và duplicate đầu vào.
- `neighbors(frame_id, window_ms=...)`: lấy frame lân cận cùng video.
- `filter_frame_ids(filters)`: lọc theo video và thời gian.
- `resolve_frame_asset(frame)`: lấy đường dẫn ảnh đầy đủ.
