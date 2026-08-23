# Canonical transcript data

`hcmai.data.enrichment.transcripts` đọc audio từ video, chia các vùng có lời
nói và ghi riêng một file Parquet cho mỗi video. Các transcript segment là
**source of truth** của ASR và giữ nguyên timeline media. Caller dùng `TranscriptService` trong
`pipeline.py`; ASR và diarization cụ thể nằm sau các adapter nội bộ. Audio giữ
media-time offset từ PTS, vì vậy VAD/ASR timestamps không bị đặt lại về `0`.

```text
transcripts/
├── pipeline.py              # TranscriptService public facade
├── prepare.py               # Job implementation
├── manifest.py              # Source/config/model resume identity
├── materialize.py           # Derived frame-aligned compatibility view
├── publication.py           # Recoverable staged promotion
├── store.py                 # TranscriptStore
└── adapters/
    ├── asr.py
    └── diarization.py
```

## Công nghệ

- `Qwen/Qwen3-ASR-1.7B-hf`: nhận dạng giọng nói đa ngôn ngữ theo batch.
- Silero VAD 6.2: chia audio thành các segment có lời nói.
- `pyannote/speaker-diarization-community-1`: xác định ai nói khi nào.
- PyAV: đọc và chuyển audio về mono 16 kHz.
- pandas, PyArrow và Pydantic: ghi Parquet đúng schema.

## Input

```text
dataset/
├── Videos_L21_a/video/L21_V001.mp4
├── Videos_L22_a/video/L22_V001.mp4
└── Videos_*/video/*.mp4
```

Video được tìm đệ quy. Hỗ trợ MP4, MKV, AVI, MOV, WebM và M4V.
Các bản video trùng `video_id` và cùng kích thước chỉ được xử lý một lần.

## Build

Chạy từ thư mục gốc của repository:

```bash
python3.12 -m venv .venv-asr
source .venv-asr/bin/activate
sudo apt-get install ffmpeg
pip install torch==2.9.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[transcripts]'

export HF_TOKEN="hf_..."

PYTHONPATH=.:src python scripts/prepare_transcripts.py \
  --videos-root /path/to/dataset \
  --config configs/enrichment.yaml
```

Trước khi chạy, chấp nhận điều kiện của
[`community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1).
Token chỉ đọc từ `HF_TOKEN`, không lưu trong code hoặc output. Model revisions
được pin trong `configs/enrichment.yaml`. Có thể thêm `--limit 2`,
`--no-resume`, hoặc `--no-diarization`. Khi diarization tắt,
`speaker_id=None` vẫn là transcript hợp lệ.

Kết quả:

```text
Expected videos: <count>
Transcribed: <count>
No speech: <count>
Failed: 0
Segments: <count>
Output: artifacts/transcripts
Status: PASSED
```

## Output

```text
artifacts/
├── enrichment/transcripts/L25/
│   ├── L25_V001.parquet
│   └── L25_V001.manifest.json
└── enrichment/asr/frame_enrichment.parquet
```

Mỗi manifest ghi source SHA-256, configuration hash, resolved ASR/diarization
revisions, schema/pipeline versions, segment count và completion status. Resume
chỉ reuse khi tất cả trường identity khớp và Parquet validate lại thành công.
Source, config, revision hoặc version đổi sẽ tự động chạy lại.

Manifest cũng khai báo rõ:

```text
source_of_truth = transcript segments
frame_alignment = derived compatibility view
context_dependency = none
```

Parquet và manifest được ghi vào sibling staging paths, validate đầy đủ rồi mới
promote. Promotion lỗi sẽ restore artifact hợp lệ trước đó. Failure record chỉ
chứa exception category, không chứa raw provider error hoặc token.

Sau transcript preparation, script có thể materialize ASR text lên canonical
frames như một compatibility artifact dẫn xuất; đây không phải ASR source of
truth và không phải input của `FrameContext`.
Chỉ video có completed manifest được emit; video no-speech có completed row với
`asr_text=None`, còn video chưa evaluate không bị biến thành negative evidence.
Online `DataService` đọc file này qua existing `ASRStore`.

## Parquet schema

| Column | Meaning |
|---|---|
| `segment_id` | ID duy nhất của segment |
| `video_id` | ID lấy từ tên video |
| `segment_index` | Thứ tự segment trong video, bắt đầu từ `0` |
| `start_ms` | Media-time bắt đầu (đã cộng audio PTS offset) |
| `end_ms` | Media-time kết thúc (half-open interval) |
| `text` | Nội dung transcript |
| `language` | Nhãn Qwen như `vietnamese`, `english` hoặc `und` |
| `speaker_id` | Speaker có overlap lớn nhất, ví dụ `SPEAKER_00`; có thể rỗng |
| `confidence` | Điểm calibrated cấp segment nếu backend có; mặc định rỗng, không phải `0` |
| `status` | Trạng thái xử lý, mặc định `completed` |
| `model_name` | Model ASR tạo segment; có thể rỗng cho artifact legacy |
| `model_revision` | Revision đã resolve; có thể rỗng cho artifact legacy |
| `artifact_version` | Phiên bản contract segment, mặc định `asr-segment-v1` |
| `error_code`, `error_message` | Chi tiết lỗi có cấu trúc khi có |

`speaker_id` chỉ có ý nghĩa trong một video. Một segment có nhiều người nói
chỉ giữ speaker chiếm nhiều thời gian nhất.

## Đọc transcript

```python
from hcmai.data.enrichment.transcripts.pipeline import TranscriptService

store = TranscriptService.load_store("artifacts/transcripts")

segment = store.get("L21_V001_segment_000000")
segments = store.get_many(["L21_V001_segment_000000"])
video_segments = store.get_by_video("L21_V001")
at_time = store.get_at_time("L21_V001", 12_000)
in_range = store.get_in_range("L21_V001", 7_000, 17_000)
```

`TranscriptStore` nhận một file video hoặc cả thư mục `transcripts`, đọc
Parquet một lần và cung cấp các hàm:

| Hàm | Mục đích | Trả về |
|---|---|---|
| `get(segment_id)` | Lấy segment theo ID | Một `TranscriptSegment`; ID không tồn tại báo `KeyError` |
| `get_many(segment_ids)` | Lấy nhiều segment, giữ nguyên thứ tự và ID lặp | `list[TranscriptSegment]` |
| `get_by_video(video_id)` | Lấy toàn bộ lời nói của một video theo thứ tự | `list[TranscriptSegment]` |
| `get_at_time(video_id, timestamp_ms)` | Tìm segment chứa một thời điểm | `list[TranscriptSegment]` |
| `get_in_range(video_id, start_ms, end_ms)` | Tìm segment giao với một khoảng thời gian | `list[TranscriptSegment]` |

Video không tồn tại hoặc không có segment phù hợp trả về danh sách rỗng.

Code production bên ngoài component không import `adapters/`, `prepare.py`,
hay `store.py` trực tiếp. Script chuẩn gọi
`TranscriptService.from_job_config()` rồi `prepare()` để model chỉ được tạo một
lần cho cả job.
