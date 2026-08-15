"""Module làm giàu dữ liệu: OCR (Optical Character Recognition).

Trích xuất và lưu trữ văn bản xuất hiện bên trong khung hình (như biển hiệu, phụ đề cứng, ...).

Các tính năng chính:
1. Cung cấp API `generator`: Gọi chuỗi xử lý OCR trên thư mục ảnh đầu vào.
2. Tích hợp mô hình: Hỗ trợ cắm (plug) nhiều loại mô hình OCR khác nhau (như Florence-2, PaddleOCR).
3. Chuẩn hoá dữ liệu: Trả về cấu trúc OCR thống nhất (Text, Box) cho hệ thống tìm kiếm (Search)."""

__all__: list[str] = []
