"""Module Data Stores.

Cung cấp các cấu trúc lưu trữ nội bộ (in-memory hoặc indexed) giúp tra cứu nhanh ở giai đoạn Online.

Các tính năng chính:
1. Quản lý FrameStore: Tra cứu siêu dữ liệu (metadata) của các khung hình đã lọc.
2. Quản lý EvidenceStore: Tra cứu toàn bộ văn bản mô tả (caption, ocr, audio) của khung hình.
3. Tối ưu tra cứu (Fast Retrieval): Tải dữ liệu vào cấu trúc Dict hoặc Hashmap để có độ trễ O(1)."""

from hcmai.data.stores.evidence import ASRStore, CaptionStore, OCRStore
from hcmai.data.stores.frame import FrameStore

__all__ = ["ASRStore", "CaptionStore", "FrameStore", "OCRStore"]
