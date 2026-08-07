# Tiền xử lý video thành FrameStore

Pipeline đọc video theo thời gian, dùng **TransNetV2**, **EfficientGEBD** và
optical flow để chọn candidate frame. Maximum gap bảo đảm video không có
khoảng thời gian dài bị bỏ trống. DINOv3 đang tắt trong baseline hiện tại.

Frame đầu tiên có `timestamp_ms = 0` và `frame_idx = 0`:

```text
frame_idx = round(timestamp_ms * FPS / 1000)
frame_id  = <video_id>_frame_<frame_idx đủ 9 chữ số>
```

## Cài đặt

```bash
python -m pip install -e ".[preprocessing]"
```

TransNetV2 và EfficientGEBD dùng checkout/checkpoint chính thức đặt bên ngoài
repo. Model được lazy-load một lần và dùng lại cho toàn bộ video.

## Cấu hình

```yaml
preprocessing:
  videos_root: /mounted/aic2026/videos
  output_root: artifacts/frame_store

  transnet_repo: /models/TransNetV2
  transnet_weights: /models/TransNetV2/inference/transnetv2-weights

  efficientgebd_repo: /models/EfficientGEBD
  efficientgebd_enabled: true
  efficientgebd_config: /models/EfficientGEBD/model_config.yaml
  efficientgebd_checkpoint: /models/EfficientGEBD/model_best.pth
  efficientgebd_device: cuda
  efficientgebd_sample_fps: 10
  efficientgebd_overlap_frames: 20

  dino_enabled: false
```

`model_config.yaml` phải đúng với checkpoint. Checkpoint ResNet50-L2L3L4 chính
thức dùng `BaseModel`, `HEAD_CHOICE: [3]`, `FPN_START_IDX: 1`,
`CAT_PREV: true`, `IS_BASIC: false` và `NUM_BLOCKS: 1`.

Nếu muốn chạy nhẹ chỉ với optical flow và maximum gap, đặt:

```yaml
transnet_enabled: false
efficientgebd_enabled: false
dino_enabled: false
```

Mọi trường có thể đặt bằng biến môi trường với prefix
`HCMAI_PREPROCESSING_`, ví dụ:

```bash
export HCMAI_PREPROCESSING_VIDEOS_ROOT=/mounted/aic2026/videos
export HCMAI_PREPROCESSING_OUTPUT_ROOT=artifacts/frame_store
```

## Chạy

```bash
PYTHONPATH=src python scripts/preprocess_videos.py --config /path/to/preprocessing.yaml
```

Chạy nhanh một video:

```bash
PYTHONPATH=src python scripts/preprocess_videos.py \
  --config /path/to/preprocessing.yaml \
  --limit 1
```

## Output

```text
artifacts/frame_store/
├── frames.parquet
└── images/<group>/<video_id>/<frame_idx>.jpg
```

Checkpoint resume nằm ở `artifacts/.preprocessing_work/`, không phải output
public. File `.partial` không bao giờ được tính là hoàn thành.

## Cách dùng

```python
from hcmai.data import FrameStore

store = FrameStore.load("artifacts/frame_store/frames.parquet")
frame = store.get("L21_V001_frame_000000090")
frames = store.get_many([frame.frame_id])
neighbors = store.get_neighbors(frame.frame_id, window_ms=3000)
frame_ids = store.filter_frame_ids(None)
```

- `get`: lấy một frame theo ID.
- `get_many`: lấy nhiều frame, giữ nguyên thứ tự đầu vào.
- `get_neighbors`: lấy các frame cùng video trong một khoảng thời gian.
- `filter_frame_ids`: lọc ID theo video và thời gian.

`image_path` là đường dẫn tương đối so với `artifacts/frame_store/`.
`frames.parquet` là metadata canonical được các pipeline đọc. Các file
`frame_mapping.parquet` trong embedding/index và `frame_enrichment.parquet`
là artifact downstream, không thay thế file này.
