"""Module làm giàu dữ liệu: Transcripts.

Bao gồm toàn bộ quy trình trích xuất và xử lý lời thoại/phụ đề từ video.

Các tính năng chính:
1. Cung cấp API `pipeline`: Điểm truy cập chính để gọi hệ thống xử lý âm thanh.
2. Quản lý mô hình: Bao bọc (wrap) các adapter ASR (Whisper) và Diarization (PyAnnote).
3. Tương thích VQA: Xuất dữ liệu ở định dạng mà các hệ thống hỏi đáp video (VQA) có thể hiểu."""

__all__: list[str] = []
