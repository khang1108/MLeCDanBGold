"""Pipeline chính cho Data Enrichment (Làm giàu dữ liệu).

Điều phối các luồng xử lý offline (Captioning, OCR, Transcript) trên các frames đã tiền xử lý.

Các tính năng chính:
1. Quản lý luồng (Workflow): Chạy tuần tự hoặc song song các tác vụ AI enrichment (OCR, Caption, Audio).
2. Tổng hợp Artifacts: Gom nhóm kết quả text từ các mô hình thành dữ liệu chuẩn bị cho indexing.
3. Cập nhật trạng thái: Báo cáo tiến độ và duy trì manifest để đảm bảo dữ liệu không bị xử lý lặp."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hcmai.data.enrichment.caption.config import CaptionConfig
from hcmai.data.enrichment.caption.adapters.transformers import TransformersCaptionAdapter
from hcmai.data.enrichment.caption.generator import generate_captions
from hcmai.data.enrichment.caption.models.contracts import CaptionAdapter
from hcmai.data.enrichment.ocr.config import OCRConfig
from hcmai.data.enrichment.ocr.generator import generate_ocr
from hcmai.data.enrichment.ocr.models.contracts import OCRAdapter
from hcmai.data.enrichment.objects.config import ObjectConfig
from hcmai.data.enrichment.objects.importer import import_objects


class EnrichmentService:
    """Run caption, OCR, or BTC object enrichment through explicit boundaries."""

    @staticmethod
    def generate_captions(
        frames_path: str | Path,
        output_dir: str | Path,
        config: CaptionConfig,
        adapter: CaptionAdapter | None = None,
        *,
        dataset_root: str | Path = ".",
        frame_store_id: str | None = None,
    ) -> dict[str, Any]:
        return generate_captions(
            frames_path,
            output_dir,
            config,
            adapter,
            dataset_root=dataset_root,
            frame_store_id=frame_store_id,
        )

    @staticmethod
    def generate_ocr(
        frames_path: str | Path,
        output_dir: str | Path,
        config: OCRConfig,
        adapter: OCRAdapter | None = None,
        *,
        dataset_root: str | Path = ".",
        frame_store_id: str | None = None,
    ) -> dict[str, Any]:
        return generate_ocr(
            frames_path,
            output_dir,
            config,
            adapter,
            dataset_root=dataset_root,
            frame_store_id=frame_store_id,
        )

    @staticmethod
    def import_objects(
        frames_path: str | Path,
        objects_root: str | Path,
        output_dir: str | Path,
        config: ObjectConfig,
        *,
        frame_store_id: str | None = None,
    ) -> dict[str, Any]:
        """Import BTC-provided object JSON without running detection."""

        return import_objects(
            frames_path,
            objects_root,
            output_dir,
            config,
            frame_store_id=frame_store_id,
        )

    @staticmethod
    def run_caption_cli() -> int:
        from hcmai.data.enrichment.caption.generator import main

        return main()
    @staticmethod
    def create_caption_adapter(config: CaptionConfig) -> CaptionAdapter:
        return TransformersCaptionAdapter(config)

    @staticmethod
    def create_ocr_adapter(config: OCRConfig) -> OCRAdapter:
        if config.backend == "remote":
            raise NotImplementedError("Remote OCR adapter is not implemented.")
        from hcmai.data.enrichment.ocr.adapters.florence import FlorenceAdapter
        return FlorenceAdapter(config)
