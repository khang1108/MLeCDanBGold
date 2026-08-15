"""Module Video Preprocessing (Tiền xử lý Video).

Cung cấp các API công khai cho quá trình giải mã, phân tích và trích xuất khung hình từ video thô.

Các tính năng chính:
1. Cung cấp API Prepare: Trích xuất frame kết hợp Deduplication và Boundary Detection.
2. Hỗ trợ khôi phục (Resume): Chạy lại không bị lỗi (idempotent) khi pipeline bị gián đoạn.
3. Đóng gói kết quả: Trả về một `FrameStore` sẵn sàng để dùng cho giai đoạn Retrieval."""

from hcmai.data.preprocessing.config import (
    PreprocessingConfig,
    S3PreprocessingConfig,
)
from hcmai.data.preprocessing.prepare import (
    FramePreparationSession,
    prepare_frame_store,
)
from hcmai.data.preprocessing.s3 import prepare_frame_store_from_s3

__all__ = [
    "FramePreparationSession",
    "PreprocessingConfig",
    "S3PreprocessingConfig",
    "prepare_frame_store",
    "prepare_frame_store_from_s3",
]
