"""Freeze the configured published runtime artifact layout.

These assertions characterize paths only; compatibility fixtures are generated
under pytest's temporary directory so production artifacts are never copied.
"""

from hcmai.common.config import AppConfig


def test_default_artifact_paths_remain_current_layout() -> None:
    """Keep the current frame, evidence, transcript, and index paths stable."""

    settings = AppConfig()

    assert settings.dataset.frames_path.as_posix() == (
        "artifacts/frame_store/frames.parquet"
    )
    assert settings.dataset.enrichment.caption_path.as_posix() == (
        "artifacts/corpus/caption.parquet"
    )
    assert settings.dataset.enrichment.ocr_path.as_posix() == (
        "artifacts/corpus/ocr_frames.parquet"
    )
    assert settings.dataset.enrichment.object_path.as_posix() == (
        "artifacts/corpus/object_frames.parquet"
    )
    assert settings.dataset.enrichment.asr_path.as_posix() == (
        "artifacts/enrichment/asr/frame_enrichment.parquet"
    )
    assert settings.dataset.enrichment.context_path.as_posix() == (
        "artifacts/corpus/context.parquet"
    )
    assert settings.dataset.enrichment.transcripts_path.as_posix() == (
        "artifacts/enrichment/transcripts"
    )
    assert settings.dataset.media_info_path.as_posix() == "data/media-info"
    assert settings.index.path.as_posix() == "artifacts/indexes/visual"
    assert settings.index.context_path.as_posix() == "artifacts/indexes/context"
    assert settings.index.asr_segment_path.as_posix() == (
        "artifacts/indexes/asr_segments"
    )
