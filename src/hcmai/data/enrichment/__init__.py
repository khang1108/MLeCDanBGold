"""Module Data Enrichment (Làm giàu dữ liệu).

Bao gồm các tính năng xử lý offline như sinh Caption, nhận diện chữ (OCR) và trích xuất lời thoại.

Các tính năng chính:
1. Phân phối nhiệm vụ: Gửi frame cho mô hình OCR (Florence-2) và Captioning.
2. Trích xuất âm thanh: Điều hướng video qua pipeline ASR (Whisper) để lấy phụ đề.
3. Đồng bộ dữ liệu: Tạo EvidenceStore hợp nhất các văn bản này lại với timestamp chính xác."""
