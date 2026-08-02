"""Load text evidence stores and their compatible dense indexes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hcmai.common.config import AppConfig
from hcmai.common.schemas import RetrievalSource
from hcmai.common.utils.logging import get_logger
from hcmai.data import ASRStore, CaptionStore, OCRStore
from hcmai.retriever.caption import ASRRetriever, CaptionRetriever, OCRRetriever
from hcmai.retriever.dense import DenseIndex

logger = get_logger(__name__)


def load_evidence_stores(
    settings: AppConfig,
    messages: list[str],
) -> dict[RetrievalSource, Any]:
    """Load configured caption, OCR, and ASR metadata stores."""
    configured = (
        (RetrievalSource.CAPTION, CaptionStore,
         settings.dataset.enrichment.caption_path),
        (RetrievalSource.OCR, OCRStore, settings.dataset.enrichment.ocr_path),
        (RetrievalSource.ASR, ASRStore, settings.dataset.enrichment.asr_path),
    )
    stores: dict[RetrievalSource, Any] = {}
    for source, store_type, path in configured:
        if path is None:
            continue
        if not path.is_file() or path.stat().st_size == 0:
            messages.append(f"{source.value.upper()} artifact not available at {path}")
            continue
        try:
            stores[source] = store_type(path)
            logger.info(
                "%sStore loaded path=%s frames=%d",
                source.value.upper(), path, len(stores[source]),
            )
        except Exception as error:
            messages.append(
                f"Could not load {source.value} artifact {path}: "
                f"{type(error).__name__}: {error}"
            )
    return stores


def _text_index_specs(
    settings: AppConfig,
) -> tuple[tuple[RetrievalSource, Any, Path], ...]:
    return (
        (RetrievalSource.CAPTION, CaptionRetriever, Path(os.getenv(
            "HCMAI_CAPTION_INDEX_PATH", str(settings.index.caption_path)
        ))),
        (RetrievalSource.OCR, OCRRetriever, Path(os.getenv(
            "HCMAI_OCR_INDEX_PATH", str(settings.index.ocr_path)
        ))),
        (RetrievalSource.ASR, ASRRetriever, Path(os.getenv(
            "HCMAI_ASR_INDEX_PATH", str(settings.index.asr_path)
        ))),
    )


def load_text_indexes(
    settings: AppConfig,
    visual_index: DenseIndex,
) -> list[tuple[RetrievalSource, Any, Path, DenseIndex]]:
    """Load all required text indexes and enforce one dataset version."""
    loaded = []
    for source, retriever_type, index_dir in _text_index_specs(settings):
        if not index_dir.is_dir():
            raise FileNotFoundError(
                f"{source.value.upper()} index directory not available at {index_dir}"
            )
        index = DenseIndex.load(index_dir)
        if index.metadata.dataset_version != visual_index.metadata.dataset_version:
            raise ValueError(
                f"visual and {source.value} index dataset versions differ"
            )
        loaded.append((source, retriever_type, index_dir, index))
    return loaded
