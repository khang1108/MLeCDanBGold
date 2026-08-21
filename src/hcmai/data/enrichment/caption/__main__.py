"""Điểm khởi chạy (Entry point) cho Caption Enrichment.

Chạy pipeline tạo caption bằng các thiết lập cấu hình mặc định (có thể gọi trực tiếp từ command line)."""

from hcmai.data.enrichment.pipeline import EnrichmentService

if __name__ == "__main__":
    raise SystemExit(EnrichmentService.run_caption_cli())
