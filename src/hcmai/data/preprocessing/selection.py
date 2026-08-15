"""Frame Selection & Deduplication Module.

Module này cung cấp các thuật toán để chắt lọc (select) và loại bỏ trùng lặp (deduplicate)
các khung hình (frames) từ video, nhằm tối ưu hóa lượng dữ liệu đầu ra mà không làm mất đi các 
khoảnh khắc quan trọng.

Các tính năng chính:
1. Candidate Selection: Dựa vào điểm số của Shot (TransNet) và Event (GEBD), module sẽ chọn ra 
   các khung hình ứng viên, bao gồm các peak frames và các khung hình có chuyển động mạnh (dynamic coverage).
2. Semantic Deduplication: Sử dụng mô hình DINO để trích xuất đặc trưng hình ảnh, sau đó 
   loại bỏ các khung hình liền kề có sự tương đồng về mặt ngữ nghĩa (semantic similarity) quá cao.
3. Gap Restoration: Đảm bảo khoảng cách thời gian giữa các khung hình (gap) không vượt quá 
   giới hạn tối đa (maximum_gap_ms), tránh việc bỏ sót bối cảnh khi cảnh quay tĩnh kéo dài.
4. Burst Sampling: Hỗ trợ lấy mẫu dày đặc (burst) xung quanh các sự kiện quan trọng."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hcmai.data.preprocessing.config import PreprocessingConfig
from hcmai.data.preprocessing.video import (
    FrameMeta,
    add_dynamic_coverage,
    peak_indices,
)
import bisect
import cv2

BURST_RADIUS_MS = 500
BURST_STEP_MS = 200
DEDUP_WINDOW_MS = 1_000
DEDUP_MOTION_THRESHOLD = 0.008


@dataclass(slots=True)
class CandidateFrame:
    """One selected frame with its selection signals."""

    frame: FrameMeta
    shot_id: int
    event_id: int
    shot_score: float
    event_score: float
    reasons: tuple[str, ...]
    protected: bool


class DinoEncoder:
    """Lazy DINO encoder for local semantic deduplication."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self.processor: Any | None = None
        self.model: Any | None = None

    def encode(self, images: list[Any]) -> np.ndarray:
        """Return normalized global image embeddings."""
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if self.model is None:
            self.processor = AutoImageProcessor.from_pretrained(
                self.config.dino_model,
                revision=self.config.dino_revision,
            )
            self.model = AutoModel.from_pretrained(
                self.config.dino_model,
                revision=self.config.dino_revision,
                dtype=getattr(torch, self.config.dino_dtype),
            ).to(self.config.device).eval()

        assert self.processor is not None
        inputs = self.processor(images=images, return_tensors="pt").to(
            self.config.device
        )

        assert self.model is not None
        with torch.inference_mode():
            output = self.model(**inputs)

        # Normalize output vectors
        vectors = torch.nn.functional.normalize(output.pooler_output.float(), dim=1)
        return vectors.cpu().numpy()


def _expand_burst(
    timestamps: list[int],
    center: int,
    reason: str,
    reasons: list[set[str]],
) -> None:
    """Keep regularly spaced context around one trigger."""
    center_ms = timestamps[center]

    # Tìm khoảng frame để mở rộng burst xung quanh frame trigger
    # Dùng binary search để tìm index của frame đầu tiên và cuối cùng 
    # trong khoảng burst (tức là -500ms và +500ms so với frame trigger)

    # Ví dụ:
    # timestamps = [0, 100, 200, 1000, 1100, 1200, 2000, 2100]
    # center = 3 (1000ms)
    # left = bisect.bisect_left(timestamps, 500ms) -> 3
    # right = bisect.bisect_right(timestamps, 1500ms) -> 6
    # index chạy từ 3 đến 5
    left = bisect.bisect_left(timestamps, center_ms - BURST_RADIUS_MS)
    right = bisect.bisect_right(timestamps, center_ms + BURST_RADIUS_MS)

    last_ms = -BURST_STEP_MS
    for index in range(left, right):
        if index == center or timestamps[index] - last_ms >= BURST_STEP_MS:
            reasons[index].add(f"{reason}_context")
            last_ms = timestamps[index]


def select_candidates(
    frames: list[FrameMeta],
    shot_scores: np.ndarray,
    event_scores: np.ndarray,
    config: PreprocessingConfig,
) -> list[CandidateFrame]:
    """Combine boundary, motion, context, and temporal coverage signals."""
    if not frames:
        return []
    if len(shot_scores) != len(frames) or len(event_scores) != len(frames):
        raise ValueError("Boundary score count does not match decoded frames")

    # Khởi tạo danh sách reasons và protected frames
    # reasons: danh sách các lý do tại sao frame được chọn 
    # (shot_boundary, event_boundary, motion_peak, ...)
    # protected: tập hợp các index của các frame được bảo vệ 
    # (luôn được giữ lại)
    reasons = [set() for _ in frames]
    
    # Luôn bảo vệ frame đầu tiên và cuối cùng
    protected = {0, len(frames) - 1}
    
    # Tìm các peak frames dựa trên shot_threshold và event_threshold
    shot_peaks = peak_indices(shot_scores, config.shot_threshold)
    event_peaks = peak_indices(event_scores, config.event_threshold)
    
    # Tìm các frames có motion_score cao (local maxima)
    motion_peaks = {
        index
        for index, frame in enumerate(frames)
        if frame.motion_score >= config.motion_threshold
        and frame.motion_score == max(
            item.motion_score for item in frames[max(0, index - 1) : index + 2]
        )
    }

    # Thêm coverage anchors
    reasons[0].add("coverage_anchor")
    reasons[-1].add("coverage_anchor")
    timestamps = [frame.timestamp_ms for frame in frames]

    # Xử lý các loại triggers
    triggers = (
        ("shot_boundary", shot_peaks),
        ("event_boundary", event_peaks),
        ("motion_peak", motion_peaks),
    )

    # Xử lý các loại triggers
    for trigger, indices in triggers:
        # Với mỗi trigger, thêm reason vào reasons và protected set
        for index in indices:
            reasons[index].add(trigger)
            protected.add(index)
            # Mở rộng burst xung quanh trigger
            _expand_burst(
                timestamps, index, trigger.removesuffix("_boundary"), reasons
            )

    # Thêm dynamic coverage
    # Dynamic coverage là việc đảm bảo rằng không có khoảng trống thời gian nào 
    # giữa các frame được chọn vượt quá giới hạn tối đa (maximum_gap_ms)
    add_dynamic_coverage(frames, reasons, protected, config)
    
    selected = []
    shot_id = 0
    event_id = 0

    # Tạo danh sách các frame được chọn
    for index, (frame, frame_reasons) in enumerate(zip(frames, reasons)):
        # Cập nhật shot_id và event_id
        if index in shot_peaks and index > 0:
            shot_id += 1
        if index in event_peaks and index > 0:
            event_id += 1

        # Nếu có reasons thì thêm vào selected
        # Shot/event ID: cập nhật ID dựa trên các peak frames đã tìm thấy
        # Score: điểm shot và event tương ứng
        # Reasons: lý do frame được chọn (shot_boundary, event_boundary, motion_peak, ...)
        # Protected: frame có được bảo vệ không (luôn được giữ lại)
        if frame_reasons:
            selected.append(CandidateFrame(
                frame, shot_id, event_id, float(shot_scores[index]),
                float(event_scores[index]), tuple(sorted(frame_reasons)),
                index in protected,
            ))
    return selected


def _text_region_changed(path1: Any, path2: Any) -> bool:
    img1 = cv2.imread(str(path1), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(path2), cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        return True
    h, w = img1.shape
    roi_top = int(h * 0.7)
    roi1 = img1[roi_top:, :]
    roi2 = img2[roi_top:, :]
    mse = np.mean((roi1.astype("float") - roi2.astype("float")) ** 2)
    return mse > 200.0


def deduplicate(
    candidates: list[CandidateFrame],
    embeddings: np.ndarray,
    image_paths: list[Any],
    config: PreprocessingConfig,
) -> list[CandidateFrame]:
    """Drop only nearby, same-shot, unprotected semantic duplicates."""
    kept: list[int] = []
    for index, candidate in enumerate(candidates):
        if not kept or candidate.protected:
            kept.append(index)
            continue
        previous = candidates[kept[-1]]
        text_changed = _text_region_changed(image_paths[kept[-1]], image_paths[index])
        duplicate = (
            candidate.shot_id == previous.shot_id
            and candidate.frame.timestamp_ms - previous.frame.timestamp_ms
            <= DEDUP_WINDOW_MS
            and candidate.frame.motion_score <= DEDUP_MOTION_THRESHOLD
            and not text_changed
            and float(embeddings[index] @ embeddings[kept[-1]])
            >= config.dedup_similarity
        )
        if not duplicate:
            kept.append(index)
    return [candidates[index] for index in kept]


def restore_maximum_gap(
    candidates: list[CandidateFrame],
    retained: list[CandidateFrame],
    config: PreprocessingConfig,
) -> list[CandidateFrame]:
    """Reinsert the fewest available candidates needed for hard coverage.

    Semantic deduplication may remove an unprotected coverage frame.  This
    repair operates on decode identities and restores the configured temporal
    bound without changing submission frame indices.
    """

    if not retained:
        return []
    ordered = sorted(candidates, key=lambda item: item.frame.decode_index)
    retained_ids = {item.frame.decode_index for item in retained}
    repaired = [ordered[0]]
    for target in ordered[1:]:
        if target.frame.decode_index not in retained_ids:
            continue
        while (
            target.frame.timestamp_ms - repaired[-1].frame.timestamp_ms
            > config.maximum_gap_ms
        ):
            deadline = repaired[-1].frame.timestamp_ms + config.maximum_gap_ms
            available = [
                item
                for item in ordered
                if repaired[-1].frame.decode_index < item.frame.decode_index
                < target.frame.decode_index
                and item.frame.timestamp_ms <= deadline
            ]
            if not available:
                raise ValueError(
                    "Cannot restore maximum frame gap from selected candidates"
                )
            repaired.append(available[-1])
        repaired.append(target)
    return repaired
