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
frame_id  = <video_id>_frame_<decode_index đủ 9 chữ số>
```

`frame_idx` chỉ là mapping dùng khi submit. Pipeline dùng `decode_index` cho
identity, tên JPEG và join giữa hai decode pass, nên hai decoded frame có cùng
`frame_idx` vẫn không va chạm.

## Cài đặt

```bash
python -m pip install -e ".[preprocessing]"
```

Khi dùng S3, cài cả hai extra:

```bash
python -m pip install -e ".[preprocessing,s3]"
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
  # Nên pin immutable Hugging Face commit cho run chính thức.
  dino_revision: null
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

### S3 input và artifact publication

Thay `videos_root` bằng block `s3` (không cấu hình cả hai):

```yaml
preprocessing:
  s3:
    bucket: hcmai-dataset
    videos_prefix: videos
    artifacts_prefix: artifacts
    region: ap-southeast-1
    # endpoint_url: http://localhost:9000  # chỉ dùng cho S3-compatible storage
    # staging_root: /local-nvme/hcmai

  output_root: artifacts/frame_store
  # Các model path và selection settings giống cấu hình local ở trên.
```

Ví dụ đầy đủ nằm tại `configs/preprocessing.s3.example.yaml`. Credential không
được ghi trong YAML; boto3 dùng IAM role hoặc standard AWS credential chain.
Worker cần quyền list `videos_prefix`, get video objects, và put/head dưới
`artifacts_prefix`.

Pipeline list object đệ quy dưới `videos_prefix`, chỉ stage **một video tại một
thời điểm**, chạy đủ hai decode pass trên file local, rồi xóa file stage. Vì
vậy decoder/model hiện tại không cần seek trực tiếp qua S3 và local disk usage
không tăng theo toàn corpus.

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

`--limit N` luôn ghi vào FrameStore riêng
`<output_root>.limit-N`; nó không sửa hoặc truncate full-corpus FrameStore.
Với S3, limited run publish vào
`<artifacts_prefix>/limited/limit-N/`; nó không cập nhật production
`<artifacts_prefix>/latest.json`.

## Output

```text
artifacts/frame_store/
├── frames.parquet
├── manifest.json
└── images/<group>/<video_id>/<decode_index>.jpg
```

Các bên chỉ cần:

```text
FrameStore root: artifacts/frame_store
Metadata:        artifacts/frame_store/frames.parquet
```

Checkpoint nằm tại `artifacts/.frame_store_preprocessing_work/`, không phải
output query. Limited run có checkpoint root riêng tương ứng.
`image_path` trong Parquet là đường dẫn tương đối từ FrameStore root.
`manifest.json` lưu pipeline version, config hash, model/source fingerprints
và số video/frame để audit và invalidation khi resume.
Resume chỉ được bật khi `dino_revision` đã pin; nếu để `null`, pipeline luôn
xử lý lại để tránh tái dùng artifact sau khi remote model `main` thay đổi.
Với S3, checkpoint còn bind vào object key, ETag, size và LastModified; object
source thay đổi sẽ invalidate checkpoint tương ứng.

Sau khi local bundle và toàn bộ canonical image được validate, pipeline upload
vào prefix immutable:

```text
s3://<bucket>/<artifacts_prefix>/versions/<bundle_id>/
├── frames.parquet
├── manifest.json
├── source-manifest.json
├── images/...
└── _SUCCESS.json
```

Mỗi object được kiểm tra lại `ContentLength`. `_SUCCESS.json` chỉ được ghi sau
khi đủ artifact, và `<artifacts_prefix>/latest.json` được cập nhật cuối cùng.
Upload lỗi trước completion không thể chuyển consumer sang bundle chưa hoàn
chỉnh. Local FrameStore được giữ lại để audit/retry.

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
