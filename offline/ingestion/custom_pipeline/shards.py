"""Split committed batch artifacts into per-video shards and build 3 indexes.

Operates only on already-computed batch-scoped tables (Caption, OCR frame/
region, Object frame/detection, FrameContext, and aligned visual/context
embedding vectors) plus an already-validated :class:`ASRReuseBundle`. This
module never runs model inference or re-embeds anything: the ASR
``SegmentDenseIndex`` is only ever subset from its persisted vectors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from hcmai.common.utils.io import write_parquet
from hcmai.common.utils.logging import get_logger
from offline.ingestion.custom_pipeline.asr import ASRReuseBundle
from hcmai.retrieval.retriever.artifacts import sha256_file
from hcmai.retrieval.retriever.dense.index import DenseIndex
from hcmai.retrieval.retriever.models.metadata import IndexMetadata
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex

logger = get_logger(__name__)

# Frame-native tables must cover every frame of their video exactly once.
FRAME_NATIVE_TABLE_NAMES = ("caption", "ocr_frames", "object_frames", "context")
# Child tables may legitimately have zero rows for a frame (no OCR region /
# no detected object); only foreign frame_id references are rejected.
CHILD_TABLE_NAMES = ("ocr_regions", "object_detections")


class VideoShardError(ValueError):
    """Raised when a per-video shard fails canonical coverage validation."""


@dataclass(frozen=True)
class VideoShard:
    """One video's row slice of every batch table, plus aligned vectors."""

    video_id: str
    frame_native: dict[str, pd.DataFrame]
    child: dict[str, pd.DataFrame]
    visual_vectors: np.ndarray
    visual_mapping: pd.DataFrame
    context_vectors: np.ndarray
    context_mapping: pd.DataFrame


@dataclass(frozen=True)
class IndexArtifactSummary:
    """Lineage and checksum summary for one published index directory."""

    path: str
    vector_count: int
    embedding_dim: int
    checksums: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchIndexInventory:
    """Ordered identity, counts, and checksums for a batch's three indexes."""

    batch_id: str
    video_ids: tuple[str, ...]
    visual: IndexArtifactSummary
    context: IndexArtifactSummary
    asr_segments: IndexArtifactSummary


def _slice_vectors(
    vectors: np.ndarray, mapping: pd.DataFrame, video_id: str
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return one video's vectors and mapping rows, ordered by embedding_index."""

    video_mapping = (
        mapping.loc[mapping["video_id"] == video_id]
        .sort_values("embedding_index")
        .reset_index(drop=True)
    )
    positions = video_mapping["embedding_index"].to_numpy()
    if positions.size and positions.max(initial=-1) >= len(vectors):
        raise VideoShardError(
            f"vector mapping for {video_id} references embedding_index beyond "
            f"the {len(vectors)}-row vector array"
        )
    return np.asarray(vectors[positions]), video_mapping


def split_batch_artifacts_by_video(
    video_ids: Sequence[str],
    frames_table: pd.DataFrame,
    frame_native_tables: Mapping[str, pd.DataFrame],
    child_tables: Mapping[str, pd.DataFrame],
    visual_vectors: np.ndarray,
    visual_mapping: pd.DataFrame,
    context_vectors: np.ndarray,
    context_mapping: pd.DataFrame,
) -> dict[str, VideoShard]:
    """Split every batch table into a validated per-video shard.

    Raises:
        VideoShardError: If any frame-native table or vector mapping does not
            exactly cover its video's canonical frames, or a child table
            references a frame outside that coverage.
    """

    shards: dict[str, VideoShard] = {}
    for video_id in tqdm(video_ids, desc="Splitting batch artifacts by video", unit="video"):
        expected_frame_ids = set(
            frames_table.loc[frames_table["video_id"] == video_id, "frame_id"]
        )
        if not expected_frame_ids:
            raise VideoShardError(f"no canonical frames found for video: {video_id}")

        frame_native = {
            name: table.loc[table["video_id"] == video_id].reset_index(drop=True)
            for name, table in frame_native_tables.items()
        }
        child = {
            name: table.loc[table["video_id"] == video_id].reset_index(drop=True)
            for name, table in child_tables.items()
        }
        visual_slice_vectors, visual_slice_mapping = _slice_vectors(
            visual_vectors, visual_mapping, video_id
        )
        context_slice_vectors, context_slice_mapping = _slice_vectors(
            context_vectors, context_mapping, video_id
        )

        shard = VideoShard(
            video_id=video_id,
            frame_native=frame_native,
            child=child,
            visual_vectors=visual_slice_vectors,
            visual_mapping=visual_slice_mapping,
            context_vectors=context_slice_vectors,
            context_mapping=context_slice_mapping,
        )
        validate_video_shard(shard, expected_frame_ids)
        shards[video_id] = shard

    logger.info("split batch artifacts into %d video shard(s)", len(shards))
    return shards


def validate_video_shard(shard: VideoShard, expected_frame_ids: set[str]) -> None:
    """Enforce exact frame coverage for one video's shard.

    Frame-native tables and vector mappings must cover ``expected_frame_ids``
    exactly once each; child tables may be empty but must not reference a
    frame outside that coverage.

    Raises:
        VideoShardError: On missing/foreign/duplicate frame coverage, a vector
            count that disagrees with its mapping, or non-finite vectors.
    """

    for name, table in shard.frame_native.items():
        table_ids = list(table["frame_id"])
        if len(set(table_ids)) != len(table_ids):
            raise VideoShardError(
                f"{name} shard for {shard.video_id} has duplicate frame_id rows"
            )
        table_id_set = set(table_ids)
        if table_id_set != expected_frame_ids:
            missing = sorted(expected_frame_ids - table_id_set)
            foreign = sorted(table_id_set - expected_frame_ids)
            raise VideoShardError(
                f"{name} shard for {shard.video_id} has incomplete frame coverage: "
                f"missing={missing} foreign={foreign}"
            )

    for name, table in shard.child.items():
        if table.empty:
            continue
        foreign = sorted(set(table["frame_id"]) - expected_frame_ids)
        if foreign:
            raise VideoShardError(
                f"{name} shard for {shard.video_id} references foreign frame_id: {foreign}"
            )

    for name, vectors, mapping in (
        ("visual", shard.visual_vectors, shard.visual_mapping),
        ("context", shard.context_vectors, shard.context_mapping),
    ):
        mapping_ids = set(mapping["frame_id"])
        if mapping_ids != expected_frame_ids:
            missing = sorted(expected_frame_ids - mapping_ids)
            foreign = sorted(mapping_ids - expected_frame_ids)
            raise VideoShardError(
                f"{name} vector mapping for {shard.video_id} has incomplete coverage: "
                f"missing={missing} foreign={foreign}"
            )
        if len(vectors) != len(mapping):
            raise VideoShardError(
                f"{name} vector count for {shard.video_id} ({len(vectors)}) disagrees "
                f"with mapping rows ({len(mapping)})"
            )
        if not np.all(np.isfinite(vectors)):
            raise VideoShardError(f"{name} vectors for {shard.video_id} contain non-finite values")


def write_video_shard(shard: VideoShard, output_root: str | Path) -> dict[str, Path]:
    """Persist one video shard's tables and vectors under ``output_root``.

    Returns:
        A mapping of logical artifact name to the written file path.
    """

    video_dir = Path(output_root) / "videos" / shard.video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for name, table in {**shard.frame_native, **shard.child}.items():
        path = video_dir / f"{name}.parquet"
        write_parquet(table, path)
        written[name] = path

    for name, vectors, mapping in (
        ("visual", shard.visual_vectors, shard.visual_mapping),
        ("context", shard.context_vectors, shard.context_mapping),
    ):
        vectors_path = video_dir / f"{name}_vectors.npy"
        mapping_path = video_dir / f"{name}_mapping.parquet"
        np.save(vectors_path, vectors)
        write_parquet(mapping, mapping_path)
        written[f"{name}_vectors"] = vectors_path
        written[f"{name}_mapping"] = mapping_path

    return written


def _concatenate_shard_vectors(
    video_ids: Sequence[str], video_shards: Mapping[str, VideoShard], kind: str
) -> tuple[np.ndarray, pd.DataFrame]:
    """Concatenate one vector kind across shards in canonical group order."""

    vector_parts: list[np.ndarray] = []
    mapping_parts: list[pd.DataFrame] = []
    for video_id in video_ids:
        shard = video_shards[video_id]
        vector_parts.append(shard.visual_vectors if kind == "visual" else shard.context_vectors)
        mapping_parts.append(shard.visual_mapping if kind == "visual" else shard.context_mapping)

    vectors = np.concatenate(vector_parts, axis=0)
    mapping = pd.concat(mapping_parts, ignore_index=True)
    mapping = mapping.assign(embedding_index=np.arange(len(mapping)))
    return vectors, mapping


def _subset_asr_segments(
    asr_bundle: ASRReuseBundle, video_ids: Sequence[str]
) -> tuple[np.ndarray, pd.DataFrame, IndexMetadata]:
    """Subset persisted ASR vectors/mapping for exactly ``video_ids``.

    Reuses the already-validated :class:`SegmentDenseIndex`; no ASR inference
    or text embedding is ever performed here. The source metadata travels
    with the subset so the derived batch index retains its true encoder
    lineage instead of assigning a synthetic model label.
    """

    index = SegmentDenseIndex.load(asr_bundle.index_root)
    vector_parts: list[np.ndarray] = []
    mapping_parts: list[pd.DataFrame] = []
    for video_id in video_ids:
        positions = index.video_positions(video_id)
        vector_parts.append(np.asarray(index.vectors[positions]))
        mapping_parts.append(index.mapping.iloc[positions].reset_index(drop=True))

    vectors = np.concatenate(vector_parts, axis=0)
    mapping = pd.concat(mapping_parts, ignore_index=True)
    mapping = mapping.assign(embedding_index=np.arange(len(mapping)))
    return vectors, mapping, index.metadata


def _summarize_index(output_dir: Path, checksum_filenames: Sequence[str], vector_count: int, embedding_dim: int) -> IndexArtifactSummary:
    """Hash every checksummed index file for the batch inventory record."""

    checksums = {
        filename: sha256_file(output_dir / filename) for filename in checksum_filenames
    }
    return IndexArtifactSummary(
        path=str(output_dir),
        vector_count=vector_count,
        embedding_dim=embedding_dim,
        checksums=checksums,
    )


def build_batch_index_bundle(
    batch_id: str,
    video_ids: Sequence[str],
    video_shards: Mapping[str, VideoShard],
    asr_bundle: ASRReuseBundle,
    output_root: str | Path,
    *,
    dataset_version: str,
    visual_model_name: str,
    visual_model_revision: str | None = None,
    context_model_name: str,
    context_model_revision: str | None = None,
) -> BatchIndexInventory:
    """Build, publish, and checksum-load a batch's visual/context/ASR indexes.

    Videos are concatenated in canonical ``video_ids`` order and every
    mapping's ``embedding_index`` is rewritten to a contiguous ``0..N-1``
    range local to this batch.

    Raises:
        KeyError: If ``video_shards`` is missing a requested video_id.
        IndexArtifactError: If any built index fails to checksum-load.
    """

    output_root = Path(output_root)
    missing_shards = [video_id for video_id in video_ids if video_id not in video_shards]
    if missing_shards:
        raise KeyError(f"missing video shard(s) for batch {batch_id}: {missing_shards}")

    from hcmai.retrieval.retriever.dense.index import CHECKSUM_FILENAMES as DENSE_CHECKSUM_FILENAMES
    from hcmai.retrieval.retriever.segment.index import (
        CHECKSUM_FILENAMES as SEGMENT_CHECKSUM_FILENAMES,
    )

    logger.info("building batch %s indexes for %d video(s)", batch_id, len(video_ids))

    visual_vectors, visual_mapping = _concatenate_shard_vectors(video_ids, video_shards, "visual")
    visual_dir = output_root / "visual"
    DenseIndex.build(
        visual_vectors,
        visual_mapping,
        dataset_version=dataset_version,
        model_name=visual_model_name,
        model_revision=visual_model_revision,
    ).save(visual_dir)
    loaded_visual = DenseIndex.load(visual_dir)

    context_vectors, context_mapping = _concatenate_shard_vectors(video_ids, video_shards, "context")
    context_dir = output_root / "context"
    DenseIndex.build(
        context_vectors,
        context_mapping,
        dataset_version=dataset_version,
        model_name=context_model_name,
        model_revision=context_model_revision,
    ).save(context_dir)
    loaded_context = DenseIndex.load(context_dir)

    asr_vectors, asr_mapping, asr_metadata = _subset_asr_segments(asr_bundle, video_ids)
    if asr_metadata.model_name != context_model_name:
        raise ValueError("reusable ASR source model differs from configured evidence encoder")
    if asr_metadata.model_revision != context_model_revision:
        raise ValueError("reusable ASR source revision differs from configured evidence encoder")

    asr_dir = output_root / "asr_segments"
    SegmentDenseIndex.build(
        asr_vectors,
        asr_mapping,
        dataset_version=dataset_version,
        model_name=asr_metadata.model_name,
        model_revision=asr_metadata.model_revision,
    ).save(asr_dir)
    loaded_asr = SegmentDenseIndex.load(asr_dir)

    inventory = BatchIndexInventory(
        batch_id=batch_id,
        video_ids=tuple(video_ids),
        visual=_summarize_index(
            visual_dir, DENSE_CHECKSUM_FILENAMES, loaded_visual.metadata.vector_count, loaded_visual.metadata.embedding_dim
        ),
        context=_summarize_index(
            context_dir, DENSE_CHECKSUM_FILENAMES, loaded_context.metadata.vector_count, loaded_context.metadata.embedding_dim
        ),
        asr_segments=_summarize_index(
            asr_dir, SEGMENT_CHECKSUM_FILENAMES, loaded_asr.metadata.vector_count, loaded_asr.metadata.embedding_dim
        ),
    )
    logger.info(
        "batch %s indexes built: visual=%d context=%d asr_segments=%d",
        batch_id,
        inventory.visual.vector_count,
        inventory.context.vector_count,
        inventory.asr_segments.vector_count,
    )
    return inventory


__all__ = [
    "BatchIndexInventory",
    "CHILD_TABLE_NAMES",
    "FRAME_NATIVE_TABLE_NAMES",
    "IndexArtifactSummary",
    "VideoShard",
    "VideoShardError",
    "build_batch_index_bundle",
    "split_batch_artifacts_by_video",
    "write_video_shard",
]
