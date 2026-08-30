"""Module lõi Data (Dữ liệu).

Chịu trách nhiệm quản lý toàn bộ vòng đời dữ liệu (từ thô đến tinh) của hệ thống.

Các tính năng chính:
1. Quản lý kho dữ liệu: Hỗ trợ tương tác với FrameStore và EvidenceStore.
2. API Gateway: Cung cấp `DataService` như một cổng giao tiếp duy nhất cho toàn hệ thống orchestration.
3. Định nghĩa Dataset: Quản lý metadata cốt lõi và các hằng số liên quan đến cuộc thi."""

# Artifact preparation remains here temporarily; runtime corpus reads live in
# ``hcmai.corpus`` and are not re-exported through this legacy package.
__all__: list[str] = []
