"""Public service boundary for offline frame enrichment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hcmai.enrichment.caption.config import CaptionConfig
from hcmai.enrichment.caption.adapters.transformers import TransformersCaptionAdapter
from hcmai.enrichment.caption.generator import generate_captions
from hcmai.enrichment.caption.models.contracts import CaptionAdapter
from hcmai.enrichment.ocr.config import OCRConfig
from hcmai.enrichment.ocr.generator import generate_ocr
from hcmai.enrichment.ocr.models.contracts import OCRAdapter


class EnrichmentService:
    """Run caption or OCR enrichment through explicit model adapters."""

    @staticmethod
    def generate_captions(
        frames_path: str | Path,
        output_dir: str | Path,
        config: CaptionConfig,
        adapter: CaptionAdapter | None = None,
        *,
        dataset_root: str | Path = ".",
    ) -> dict[str, Any]:
        return generate_captions(
            frames_path,
            output_dir,
            config,
            adapter,
            dataset_root=dataset_root,
        )

    @staticmethod
    def generate_ocr(
        frames_path: str | Path,
        output_dir: str | Path,
        config: OCRConfig,
        adapter: OCRAdapter | None = None,
        *,
        dataset_root: str | Path = ".",
    ) -> dict[str, Any]:
        return generate_ocr(
            frames_path,
            output_dir,
            config,
            adapter,
            dataset_root=dataset_root,
        )

    @staticmethod
    def run_caption_cli() -> int:
        from hcmai.enrichment.caption.generator import main

        return main()
    @staticmethod
    def create_caption_adapter(config: CaptionConfig) -> CaptionAdapter:
        return TransformersCaptionAdapter(config)
