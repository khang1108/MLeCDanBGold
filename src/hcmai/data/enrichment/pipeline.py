"""Expose thin service boundaries for independent offline enrichment stages.

The service delegates Caption, OCR, BTC Object, and deterministic FrameContext
materialization. Specialist generation and context serialization remain owned
by their respective packages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hcmai.data.enrichment.caption.config import CaptionConfig
from hcmai.data.enrichment.caption.adapters.transformers import TransformersCaptionAdapter
from hcmai.data.enrichment.caption.generator import generate_captions
from hcmai.data.enrichment.caption.models.contracts import CaptionAdapter
from hcmai.data.enrichment.context.builder import build_frame_context
from hcmai.data.enrichment.context.config import FrameContextConfig
from hcmai.data.enrichment.ocr.config import OCRConfig
from hcmai.data.enrichment.ocr.generator import generate_ocr
from hcmai.data.enrichment.ocr.models.contracts import OCRAdapter
from hcmai.data.enrichment.objects.config import ObjectConfig
from hcmai.data.enrichment.objects.importer import import_objects


class EnrichmentService:
    """Run independent enrichment stages through explicit boundaries."""

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
    def build_frame_context(
        frames_path: str | Path,
        caption_path: str | Path,
        ocr_frames_path: str | Path,
        object_frames_path: str | Path,
        output_dir: str | Path,
        config: FrameContextConfig,
        *,
        frame_store_id: str | None = None,
    ) -> Path:
        """Build deterministic context from existing specialist artifacts."""

        return build_frame_context(
            frames_path,
            caption_path,
            ocr_frames_path,
            object_frames_path,
            output_dir,
            config,
            frame_store_id=frame_store_id,
        )

    @staticmethod
    def run_caption_cli() -> int:
        """Run the legacy caption CLI entry point."""

        from hcmai.data.enrichment.caption.generator import main

        return main()

    @staticmethod
    def create_caption_adapter(config: CaptionConfig) -> CaptionAdapter:
        """Create the configured local caption adapter."""

        return TransformersCaptionAdapter(config)

    @staticmethod
    def create_ocr_adapter(config: OCRConfig) -> OCRAdapter:
        """Create the configured local OCR adapter."""

        if config.backend == "remote":
            raise NotImplementedError("Remote OCR adapter is not implemented.")
        from hcmai.data.enrichment.ocr.adapters.florence import FlorenceAdapter
        return FlorenceAdapter(config)
