"""Load canonical frames and optional evidence artifacts at startup.

This module owns organizer frame metadata and the Caption, OCR, Object, ASR,
and media-info artifacts attached to it. It does not load retrieval indexes or
compose online request workflows.
"""

from __future__ import annotations

import os
from pathlib import Path
from hcmai.common.config import AppConfig, resolve_repository_path
from hcmai.common.utils.logging import get_logger
from hcmai.corpus import Corpus
from hcmai.corpus.corpus import _CorpusFrameLoadError
from hcmai.retrieval.models import RetrievalSource

logger = get_logger(__name__)


def load_corpus(
    settings: AppConfig,
    metadata_path: Path,
    dataset_root: Path,
    messages: list[str],
) -> Corpus | None:
    """Load the canonical corpus and append optional-artifact diagnostics."""
    if not metadata_path.is_file() or metadata_path.stat().st_size == 0:
        raise FileNotFoundError(f"Metadata not available at {metadata_path}")
    try:
        evidence_paths, object_path, transcript_path, video_metadata_path = (
            _configured_corpus_artifacts(settings, messages)
        )
        corpus = Corpus.open(
            metadata_path,
            evidence_paths=evidence_paths,
            dataset_root=dataset_root,
            object_counts_path=object_path,
            transcript_path=transcript_path,
            video_metadata_path=video_metadata_path,
        )
    except _CorpusFrameLoadError:
        # Canonical frames are required for identity-preserving retrieval.
        # Unlike optional evidence, malformed or unreadable frame metadata
        # must prevent startup rather than silently disabling search.
        raise
    except Exception as error:
        messages.append(
            f"Could not load metadata {metadata_path}: "
            f"{type(error).__name__}: {error}"
        )
        return None
    logger.info(
        "Corpus loaded path=%s dataset_root=%s frames=%d",
        metadata_path,
        dataset_root,
        len(corpus),
    )
    return corpus


def _configured_corpus_artifacts(
    settings: AppConfig,
    messages: list[str],
) -> tuple[
    dict[RetrievalSource, Path],
    Path | None,
    Path | None,
    Path | None,
]:
    """Select existing optional Corpus artifacts and retain startup diagnostics.

    ``Corpus.open`` remains the sole runtime loader. Context and raw-detection
    artifacts are retrieval inputs, not Corpus inputs, so they remain outside
    this boundary.
    """
    enrichment = settings.dataset.enrichment
    evidence_paths: dict[RetrievalSource, Path] = {}
    for source, configured_path in (
        (RetrievalSource.CAPTION, enrichment.caption_path),
        (RetrievalSource.OCR, enrichment.ocr_path),
    ):
        path = _runtime_optional_path(
            f"HCMAI_{source.value.upper()}_PATH",
            configured_path,
        )
        if not _typed_artifact_available(path, allow_directory=False):
            messages.append(f"{source.value.upper()} artifact not available at {path}")
            continue
        assert path is not None
        evidence_paths[source] = path

    object_path = _runtime_optional_path(
        "HCMAI_OBJECT_PATH",
        enrichment.object_path,
    )
    if not _typed_artifact_available(object_path, allow_directory=False):
        messages.append(f"OBJECTS artifact not available at {object_path}")
        object_path = None

    transcript_path = _runtime_optional_path(
        "HCMAI_TRANSCRIPTS_PATH",
        enrichment.transcripts_path,
    )
    if not _typed_artifact_available(transcript_path, allow_directory=True):
        messages.append(f"ASR transcript artifact not available at {transcript_path}")
        transcript_path = None

    video_metadata_path = _runtime_optional_path(
        "HCMAI_VIDEO_METADATA_PATH",
        settings.dataset.media_info_path,
    )
    if not _metadata_directory_available(video_metadata_path):
        messages.append(f"VIDEO metadata artifact not available at {video_metadata_path}")
        video_metadata_path = None

    return evidence_paths, object_path, transcript_path, video_metadata_path


def _runtime_optional_path(
    environment_name: str,
    default: str | Path | None,
) -> Path | None:
    """Resolve one optional artifact path with an explicit environment override."""
    configured = os.getenv(environment_name)
    if configured is not None:
        return resolve_repository_path(configured) if configured.strip() else None
    return resolve_repository_path(default) if default is not None else None


def _typed_artifact_available(
    path: Path | None,
    *,
    allow_directory: bool,
) -> bool:
    """Return whether one configured typed artifact contains readable input."""
    if path is None:
        return False
    if path.is_file():
        return path.stat().st_size > 0
    return (
        allow_directory
        and path.is_dir()
        and any(item.is_file() for item in path.rglob("*.parquet"))
    )


def _metadata_directory_available(path: Path | None) -> bool:
    """Return whether an organizer media-info directory has JSON records."""
    return path is not None and path.is_dir() and any(path.glob("*.json"))
