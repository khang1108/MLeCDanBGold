# Shared Video Frame Extraction Baseline

## 1. Mục tiêu

Dataset video lớn và chứa nhiều frame gần giống nhau. Baseline cần tạo một
**Shared Informative Frame Bank** dùng chung cho KIS, Q&A và TRAKE:

```text
Raw Videos
→ phân tích timeline một lần offline
→ giữ frame mang thông tin mới
→ loại frame dư thừa
→ Frame Bank + Temporal Micro-Index
```

Mục tiêu không phải giữ ít frame nhất, mà là:

> **Giảm storage nhưng vẫn bảo toàn scene, event ngắn và khả năng quay lại đúng vùng video.**

Với TRAKE, Frame Bank dùng để tìm vùng thời gian. Exact frame được refine từ
raw video ở query time.

### Quy tắc bắt buộc về frame identity

- `frame_idx` chính thức chỉ lấy từ mapping của ban tổ chức.
- Không suy `frame_idx` từ FPS, timestamp, filename hoặc decode order.
- Frame chưa map được chỉ dùng nội bộ trong Temporal Micro-Index, không được
  xuất thẳng thành kết quả submission.

---

## 2. Luồng offline tổng thể

```text
Raw Video
   ↓
Sequential Low-Resolution Decode
   ↓
Cheap Change Signals + PTS/GOP Metadata
   ↓
TransNetV2 Shot Score + ESTimator Event Score
   ↓
Selective SEA-RAFT + Camera-Motion Compensation
   ↓
Coverage Floor + Event Burst
   ↓
Candidate Frames
   ↓
DINOv3 Local Dedup
   ↓
SigLIP2 Embedding
   ↓
Frame Bank + Temporal Micro-Index
```

Video được decode tuần tự một lần ở độ phân giải phân tích. Ảnh chất lượng đầy
đủ chỉ được lấy cho candidate cuối, tránh ghi toàn bộ frame ra ổ đĩa.

---

## 3. Shot và event boundary

### Shot structure — TransNetV2

- [TransNet V2: An Effective Deep Network Architecture for Fast Shot Transition Detection](https://arxiv.org/abs/2008.04838)

TransNetV2 phát hiện hard cut và gradual transition:

```text
shot boundary
→ giữ frame trước/sau boundary
→ không dedup xuyên shot
```

Nó chỉ phát hiện thay đổi shot, không phát hiện đầy đủ hành động xảy ra trong
cùng một góc quay.

### Event change — ESTimator

- [Online Generic Event Boundary Detection](https://openaccess.thecvf.com/content/ICCV2025/html/Jung_Online_Generic_Event_Boundary_Detection_ICCV_2025_paper.html)

ESTimator tạo `event_score` từ sai khác giữa trạng thái dự đoán và quan sát
thực tế của video:

```text
TransNetV2 → shot-level boundary
ESTimator  → event-level boundary
```

Hai model chỉ đề xuất candidate boundary. Chúng không tự quyết định
`frame_idx` và không được xem là ground truth của TRAKE.

---

## 4. Cheap multi-signal analysis

Frame Difference đơn lẻ dễ nhầm camera pan với action và dễ bỏ sót thay đổi
chữ nhỏ. Pass đầu chỉ tính các tín hiệu rẻ:

```text
global frame change
+ regional change
+ edge change
+ codec motion vector / residual
```

Mỗi signal được normalize độc lập theo video. Không cộng tất cả thành một điểm
khó giải thích; một signal vượt threshold là đủ mở candidate window.

OCR/text change chỉ chạy trong candidate window do các signal trên tạo ra hoặc
tại Maximum-Gap anchor. Semantic change được xử lý sau bằng DINOv3, không chạy
trên mọi native frame.

---

## 5. Motion — FlowGEBD và SEA-RAFT

- [What's in the Flow? Exploiting Temporal Motion Cues for Unsupervised Generic Event Boundary Detection](https://openaccess.thecvf.com/content/WACV2024/html/Gothe_Whats_in_the_Flow_Exploiting_Temporal_Motion_Cues_for_Unsupervised_WACV_2024_paper.html)
- [SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow](https://arxiv.org/abs/2405.14793)

FlowGEBD cho thấy motion cue hữu ích cho Generic Event Boundary Detection.
SEA-RAFT cung cấp dense optical flow với cân bằng accuracy–speed tốt; nó không
tự phát hiện event hoặc tự loại camera motion.

### Khi nào chạy SEA-RAFT?

Chỉ chạy trong temporal window khi có ít nhất một trigger:

```text
cheap change signal cao
OR ESTimator event score cao
OR codec motion peak
```

SEA-RAFT chỉ so sánh hai frame trong cùng shot. Shot boundary được bảo vệ trực
tiếp vì optical flow qua một camera cut không có ý nghĩa.

### Camera-motion compensation baseline

```text
Frame t-1 + Frame t
        ↓
Shi–Tomasi feature points
        ↓
Pyramidal Lucas–Kanade tracking
        ↓
RANSAC
        ↓
estimateAffinePartial2D
(translation + rotation + scale)
        ↓
Global Camera Flow
        ↓
Residual Flow
= SEA-RAFT Flow - Global Camera Flow
        ↓
FlowGEBD-style aggregation / motion score
```

Affine transform được đổi thành flow trên cùng pixel grid với SEA-RAFT. Phần
flow còn lại biểu diễn local/object motion tốt hơn raw flow.

Nếu không đủ feature point hoặc RANSAC có quá ít inlier, dùng raw SEA-RAFT flow
theo hướng giữ dư candidate và đánh dấu `camera_compensation_valid=false` để
audit.

---

## 6. Coverage Floor + Event Burst

- [KFS-Bench: Comprehensive Evaluation of Key Frame Sampling in Long Video Understanding](https://openaccess.thecvf.com/content/WACV2026/html/Li_KFS-Bench_Comprehensive_Evaluation_of_Key_Frame_Sampling_in_Long_Video_WACV_2026_paper.html)

KFS-Bench cho thấy precision thôi chưa đủ; scene coverage và sampling balance
cũng ảnh hưởng kết quả downstream. Baseline dùng Maximum Gap để bảo vệ coverage,
không coi đây là tham số do paper đề xuất.

### Frame retention rule

```text
KEEP frame if:

shot boundary
OR event boundary
OR residual-motion peak
OR OCR / visual change
OR maximum temporal gap reached
```

Frame quanh tín hiệu mạnh được lấy thành dense burst:

```text
[t - burst_radius, t + burst_radius]
→ sample theo burst_step
```

Maximum Gap chỉ là safety floor; Event Burst mới tăng khả năng giữ event ngắn.
`maximum_gap`, `burst_radius`, `burst_step` và mọi threshold đều nằm trong
config, không hardcode trong thuật toán.

Các frame sau luôn được đánh dấu `protected`:

```text
shot boundary
event boundary
motion peak
OCR change
maximum-gap anchor
```

---

## 7. Semantic dedup — DINOv3

- [LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding](https://proceedings.mlr.press/v267/shen25j.html)
- [DINOv3](https://arxiv.org/abs/2508.10104)

LongVU chứng minh hướng dùng DINOv2 similarity để giảm temporal redundancy.
Baseline dùng DINOv3 như candidate upgrade; hiệu quả phải được đối chiếu với
DINOv2 trên dữ liệu AIC.

Dedup chỉ thực hiện:

```text
cùng video
+ cùng shot
+ trong local dedup window
```

Các candidate gần nhau được xem là duplicate khi cosine similarity vượt
`dedup_threshold`. Trong mỗi duplicate group:

```text
mọi protected frame luôn được giữ
→ với phần còn lại: nhiều signal kích hoạt hơn
→ ảnh sắc nét hơn
→ timestamp sớm hơn để tie-break
```

Không dedup xuyên shot và không xóa protected frame. DINOv3 chỉ chạy trên
candidate frames, không chạy lại toàn bộ corpus ở native FPS.

---

## 8. Text–image retrieval — SigLIP2

- [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786)

SigLIP2 tạo embedding chung cho text và retained frame:

```text
query text ↔ retained frame
```

Phân vai cố định:

```text
DINOv3  → frame ↔ frame → local dedup
SigLIP2 → text ↔ frame  → retrieval embedding
```

SigLIP2 không quyết định frame nào được giữ. Model checkpoint, resolution và
batch size được chọn bằng config và benchmark.

---

## 9. Temporal Micro-Index

- [End-to-End Compressed Video Representation Learning for Generic Event Boundary Detection](https://openaccess.thecvf.com/content/CVPR2022/html/Li_End-to-End_Compressed_Video_Representation_Learning_for_Generic_Event_Boundary_Detection_CVPR_2022_paper.html)

Paper cho thấy motion vector, residual và GOP structure vẫn chứa temporal
information mà không cần lưu toàn bộ RGB frames. Baseline giữ một metadata
shard nhẹ cho mỗi video:

```text
video_id
internal_decode_index
pts
timestamp_ms
gop_seek_pts
shot_score
event_score
motion_score
change_score
ocr_score
camera_compensation_valid
protected_reason
```

Micro-Index không phải canonical FrameStore. Nó chỉ giúp tìm đúng GOP/window để
seek và decode lại raw video.

---

## 10. Output contract

```text
artifacts/
├── frames.parquet
├── frame_images/<group>/<video_id>/<frame_id>.jpg
├── siglip2_embeddings.npy
├── siglip2_mapping.parquet
└── temporal_micro_index/
    └── <video_id>.parquet
```

`frames.parquet` chứa retained frames đã canonicalize:

```text
frame_id
video_id
frame_idx
keyframe_order
timestamp_ms
image_path
width
height
```

Đây là contract `FrameRecord` hiện tại; `frame_idx` luôn đến từ official
mapping. PTS, shot/event score và protected reason nằm trong Micro-Index để
không tạo thêm một FrameStore schema song song.

`image_path` có thể trỏ tới official keyframe sẵn có; chỉ retained frame cần
materialize mới được ghi vào `frame_images`. Embedding và metadata join bằng
`frame_id`. Internal decode index chỉ nằm trong Micro-Index. Nếu mapping chính
thức không resolve được một decoded frame, frame đó chỉ được dùng cho temporal
refinement và không được đưa vào submission FrameStore.

---

## 11. TRAKE — Adaptive Temporal Search

- [Re-thinking Temporal Search for Long-Form Video Understanding](https://openaccess.thecvf.com/content/CVPR2025/html/Ye_Re-thinking_Temporal_Search_for_Long-Form_Video_Understanding_CVPR_2025_paper.html)

T* cho thấy sparse keyframe search dễ bỏ mất needle moment và adaptive temporal
zoom có thể tìm sâu hơn trong candidate region.

```text
TRAKE Query
→ Frame Bank coarse retrieval
→ candidate video + temporal region
→ Temporal Micro-Index
→ adaptive temporal zoom
→ seek nearest GOP anchor
→ native-FPS local decode
→ ordered-event matching
→ official mapping
→ exact canonical frame_idx sequence
```

Frame Bank trả lời “ở đâu”; raw-video refinement trả lời “frame nào”. Mọi event
trong một TRAKE row phải thuộc cùng video và giữ đúng thứ tự thời gian.

---

## 12. Cấu hình và benchmark

Các tham số vận hành bắt buộc nằm trong config:

```text
analysis_resolution / analysis_fps
TransNetV2 / ESTimator / SEA-RAFT checkpoint
shot_threshold / event_threshold
change_threshold / motion_threshold
maximum_gap_ms
burst_radius_ms / burst_step_ms
dedup_window_ms / dedup_threshold
DINOv3 checkpoint
SigLIP2 checkpoint / resolution / batch size
```

So sánh mọi sampler ở cùng retained-frame budget:

```text
Uniform sampling
→ TransNetV2 + Maximum Gap
→ thêm cheap change / residual motion
→ thêm ESTimator + Event Burst
→ thêm DINOv2 hoặc DINOv3 dedup
→ full hybrid baseline
```

Metric chính:

```text
frame reduction + storage
shot / event coverage
short-event recall
TRAKE per-event và full-sequence coverage
official Mean Top-k R-Score tại 1/5/20/50/100
offline throughput và query P50/P95 latency
```

## Core idea

> **Offline tạo một Frame Bank nhỏ nhưng có coverage; query time mới zoom vào raw video khi cần độ chính xác native-frame.**

