"""Module lõi Data (Dữ liệu).

Chịu trách nhiệm quản lý toàn bộ vòng đời dữ liệu (từ thô đến tinh) của hệ thống.

Các tính năng chính:
1. Quản lý kho dữ liệu: Hỗ trợ tương tác với FrameStore và EvidenceStore.
2. API Gateway: Cung cấp `DataService` như một cổng giao tiếp duy nhất cho toàn hệ thống orchestration.
3. Định nghĩa Dataset: Quản lý metadata cốt lõi và các hằng số liên quan đến cuộc thi."""

# Compatibility alias for existing callers. New cross-component code should
# use DataService; the store implementation remains owned by this package.
from hcmai.data.stores.frame import FrameStore

__all__ = ["FrameStore"]
