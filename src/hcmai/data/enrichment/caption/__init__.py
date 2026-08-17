"""Module làm giàu dữ liệu: Caption (Mô tả hình ảnh).

Chịu trách nhiệm sinh ra các đoạn văn bản mô tả nội dung trực quan của từng khung hình.

Các tính năng chính:
1. Gọi mô hình VLM: Tương tác với mô hình Vision-Language (BLIP, LLaVA) để lấy caption.
2. Phân tách ngữ cảnh: Hỗ trợ tạo caption tiếng Anh/Việt theo nhu cầu thi đấu (KIS/VQA).
3. Tích hợp Pipeline: Cung cấp API `generator` để gắn vào luồng Enrichment chung của toàn hệ thống."""

__all__: list[str] = []
