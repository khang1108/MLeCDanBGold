# Canonical frame data

`hcmai.data` owns canonical frame preparation and lookup. Other components use
`DataService` from `pipeline.py`; store implementations remain internal.

Xem [preprocessing/README.md](preprocessing/README.md) để biết cây output chuẩn
và cách KIS, Q&A, TRAKE truy cập FrameStore.

## Cài toàn bộ model

Model source, revision và checksum được khóa trong
[`model_sources.yaml`](model_sources.yaml). Cài dependency, chấp nhận điều kiện
truy cập Pyannote Community-1 rồi chạy:

```bash
python -m pip install -e ".[preprocessing,transcripts]"
export HF_TOKEN="hf_..."
PYTHONPATH=src python -m hcmai.data.setup_models
```

Script tải đúng model đang được preprocessing và transcript sử dụng:

- TransNetV2 và ba TensorFlow weights từ GitHub chính thức.
- EfficientGEBD và checkpoint từ Google Drive chính thức.
- DINOv2 Small, Qwen3-ASR-1.7B và Pyannote Community-1 từ Hugging Face.
- Silero VAD được kiểm tra từ weight đi kèm package.

Weights nằm trong `artifacts/models/` hoặc Hugging Face cache, không nằm trong
Git. Link EfficientGEBD chứa archive khoảng 9.46 GB, nhưng script dùng HTTP
Range để chỉ tải checkpoint và config cần cho pipeline rồi kiểm tra SHA-256.

Kiểm tra model đã tải mà không gọi mạng:

```bash
PYTHONPATH=src python -m hcmai.data.setup_models --verify-only
```

Kết quả tạo thêm:

```text
artifacts/models/
├── TransNetV2/
├── EfficientGEBD/
└── preprocessing.yaml
```

Chạy FrameStore bằng cấu hình vừa sinh:

```bash
PYTHONPATH=src python scripts/preprocess_videos.py \
  --config artifacts/models/preprocessing.yaml
```

Có thể đổi nơi lưu bằng `--root`, `HCMAI_MODELS_ROOT` và `HF_HOME`.

```text
data/
├── pipeline.py              # DataService public facade
├── prepare.py               # Canonical frames.parquet builder
└── stores/
    ├── frame.py             # FrameStore
    └── evidence.py          # Caption/OCR/ASR evidence stores
```

## Purpose

This package creates the shared metadata boundary between the raw BTC dataset
and the retrieval system. AI pipelines use each record's stable `frame_id` and
relative `image_path` to build embeddings and indexes. At runtime, the backend
uses `FrameStore` to resolve a retrieved `frame_id` to the official
`video_id,frame_idx` submission pair. Keeping that mapping in one canonical
Parquet file prevents downstream components from inferring official
identifiers from filenames, timestamps, FPS, or internal IDs.

The runtime baseline reads the raw-video preprocessing output:

```text
artifacts/frame_store/frames.parquet
artifacts/frame_store/images/...
```

`preprocessing/` owns the confirmed zero-based
`round(timestamp_ms * FPS / 1000)` mapping. `prepare.py` is a separate builder
for downloaded keyframes that already provide authoritative mapping CSVs.

## Input layout

The builder follows the repository's downloaded AIC dataset layout:

```text
dataset-root/
├── features/
│   ├── map-keyframes/
│   │   └── L21_V001.csv
│   ├── media-info/
│   └── objects/
└── keyframes/
    └── L21_V001/
        └── 001.jpg
```

With the repository layout above, pass `data/` as `dataset-root`. Mapping CSVs
must contain `n`, `pts_time`, and official `frame_idx`; the numeric image stem
matches `n`. Older `map-keyframes/*.csv` and
`map-keyframes-aic25-b1/map-keyframes/*.csv` layouts remain accepted.

Every mapping CSV must have a matching `keyframes/<video_id>/` directory, and
every mapping row must resolve to its numbered image. The build fails instead
of silently dropping mappings when the downloaded dataset is incomplete.

## Alternative mapping-based build

Run from the repository root:

```bash
PYTHONPATH=src aic/bin/python scripts/prepare_data.py \
  --dataset-root data \
  --output data/metadata/frames.parquet
```

Successful output is intentionally small:

```text
Videos: 2
Frames: 6
Output: /absolute/path/data/metadata/frames.parquet
Status: PASSED
```

The command does not create reports, thumbnails, checksums, manifests, or
shards. It validates a temporary Parquet file before atomically replacing the
requested output.

The same implementation is available as a Python API:

```python
from pathlib import Path

from hcmai.data.pipeline import DataService

frames_path = DataService.prepare(
    dataset_root=Path("data"),
    output_path=Path("data/metadata/frames.parquet"),
)
```

## Parquet schema

| Column                | Meaning                                                     |
| --------------------- | ----------------------------------------------------------- |
| `frame_id`          | Stable internal key based on`video_id` and keyframe `n` |
| `video_id`          | Official mapping filename stem                              |
| `frame_idx`         | Official BTC submission index                               |
| `keyframe_order`    | Official mapping field`n`                                 |
| `timestamp_ms`      | `pts_time` converted to milliseconds for temporal search  |
| `image_path`        | POSIX relative path from`dataset_root`                    |
| `width`, `height` | Source dimensions required by`FrameRecord`                |

Code downstream of preprocessing must never rederive `frame_idx`. Every
submission must resolve through the canonical `FrameRecord`.
Distinct keyframes that share one `(video_id, frame_idx)` pair remain distinct
rows with separate `frame_id` values.

## Consumers

AI indexing can iterate records in deterministic Parquet order:

```python
from hcmai.data.pipeline import DataService

data = DataService.load(
    "artifacts/frame_store/frames.parquet",
    dataset_root="artifacts/frame_store",
)
for frame in data.iter_frames():
    image_path = data.resolve_frame_asset(frame)
    build_embedding(frame.frame_id, image_path)
```

Backend lookup preserves the official mapping:

```python
frame = data.get_frame(retrieved_frame_id)
submission = (frame.video_id, frame.frame_idx)
assert data.contains_submission(*submission)
```

Online composition also injects the configured dataset root into
`DataService`. Call `data.resolve_frame_asset(frame)` rather than joining
`image_path` independently; this enforces one path-containment policy for the
frame API, reranking, and VQA. `data.frame_asset_status()` provides a bounded,
deterministic availability sample for readiness checks.

Production code outside this package must not import `stores/` or
`prepare.py` directly. Focused unit tests may use those internals with tiny
fixtures.
