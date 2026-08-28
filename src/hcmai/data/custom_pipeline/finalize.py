"""Compact committed local batches into deterministic global corpus artifacts.

Streams already-committed batch Parquet/vector artifacts in deterministic
``(video_id, frame_id)`` order into corpus-wide specialist tables and global
visual/context/ASR indexes. Never re-runs model inference, never re-embeds,
and never byte-merges FAISS index files directly; global indexes are always
rebuilt from precomputed, validated vectors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from hcmai.common.utils.io import atomic_write, read_json, write_json, write_parquet
from hcmai.common.utils.logging import get_logger
from hcmai.data.custom_pipeline.shards import CHILD_TABLE_NAMES, FRAME_NATIVE_TABLE_NAMES
from hcmai.data.custom_pipeline.state import ArchiveStage, PipelineStateStore
from hcmai.retrieval.retriever.dense.index import DenseIndex
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex

logger = get_logger(__name__)

_SPECIALIST_KINDS = (*FRAME_NATIVE_TABLE_NAMES, *CHILD_TABLE_NAMES)
_EMBEDDING_KINDS = ("visual", "context")
_FINALIZE_REPORT_FILENAME = "finalize_report.json"


class FinalizeError(RuntimeError):
    """Raised when local corpus finalization cannot proceed or is inconsistent."""


@dataclass(frozen=True)
class BatchManifest:
    """One committed batch's identity, as recorded by ``commit_local_batch``."""

    batch_id: str
    root: Path
    video_ids: tuple[str, ...]
    canonical_frame_digest: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_batch_manifest(batch_root: Path) -> BatchManifest:
    """Load and validate one committed batch's manifest and success marker."""

    manifest_path = batch_root / "manifest.json"
    success_path = batch_root / "_SUCCESS.json"
    if not manifest_path.is_file() or not success_path.is_file():
        raise FinalizeError(f"batch at {batch_root} is not committed (missing manifest/_SUCCESS)")

    data = read_json(manifest_path)
    return BatchManifest(
        batch_id=data["batch_id"],
        root=batch_root,
        video_ids=tuple(data["video_ids"]),
        canonical_frame_digest=data["canonical_frame_digest"],
    )


def discover_committed_batches(batches_root: str | Path) -> list[BatchManifest]:
    """Discover every committed batch under ``batches_root`` in deterministic order.

    Raises:
        FinalizeError: If two batches share a video_id, or a video_id is
            claimed twice within one plan.
    """

    root = Path(batches_root)
    manifests = sorted(
        (_load_batch_manifest(path.parent) for path in root.rglob("manifest.json")),
        key=lambda manifest: manifest.batch_id,
    )

    seen_video_ids: set[str] = set()
    for manifest in manifests:
        overlap = seen_video_ids & set(manifest.video_ids)
        if overlap:
            raise FinalizeError(
                f"batch {manifest.batch_id} claims video_id(s) already claimed by "
                f"another batch: {sorted(overlap)}"
            )
        seen_video_ids.update(manifest.video_ids)

    logger.info("discovered %d committed batch(es) under %s", len(manifests), root)
    return manifests


def require_full_plan_cleaned(state_store: PipelineStateStore, archive_ids: Sequence[str]) -> None:
    """Confirm every archive in the complete frozen plan is ``cleaned``.

    Raises:
        FinalizeError: If any archive is missing or not yet cleaned.
    """

    for archive_id in archive_ids:
        record = state_store.get_archive(archive_id)
        if record is None or record.stage != ArchiveStage.CLEANED:
            observed = record.stage.value if record is not None else "missing"
            raise FinalizeError(
                f"archive {archive_id} is {observed!r}; finalize requires the "
                "complete frozen plan to be cleaned"
            )


def _shard_paths_for_kind(batch_manifests: Sequence[BatchManifest], kind: str) -> list[Path]:
    """Return every per-video shard path for one specialist ``kind``, in order."""

    return [
        manifest.root / "videos" / video_id / f"{kind}.parquet"
        for manifest in batch_manifests
        for video_id in manifest.video_ids
    ]


def compact_specialist_shards(
    kind: str,
    shard_paths: Sequence[Path],
    output: str | Path,
) -> pd.DataFrame:
    """Stream-concatenate one specialist kind's per-video shards deterministically.

    Frame-native kinds (see ``FRAME_NATIVE_TABLE_NAMES``) are ordered by
    ``(video_id, frame_id)`` and must not contain a duplicate ``frame_id``.
    Child kinds may be empty and are concatenated in shard order.

    Raises:
        FinalizeError: If a shard path is missing, no shards are supplied, or
            a frame-native kind has a duplicate ``frame_id``.
    """

    tables: list[pd.DataFrame] = []
    for path in shard_paths:
        if not path.is_file():
            raise FinalizeError(f"missing {kind} shard: {path}")
        tables.append(pd.read_parquet(path))
    if not tables:
        raise FinalizeError(f"no {kind} shards supplied")

    table = pd.concat(tables, ignore_index=True)
    if kind in FRAME_NATIVE_TABLE_NAMES:
        if table["frame_id"].duplicated().any():
            raise FinalizeError(f"duplicate frame_id found while compacting {kind}")
        table = table.sort_values(["video_id", "frame_id"]).reset_index(drop=True)

    write_parquet(table, output)
    logger.info("compacted %s: %d row(s) -> %s", kind, len(table), output)
    return table


def compact_frame_metadata(
    batch_manifests: Sequence[BatchManifest],
    dataset_root: str | Path,
    output: str | Path,
) -> pd.DataFrame:
    """Compact retained per-video ``frames`` shards without decoding images.

    Every row's ``image_path`` is checked for existence on disk; image bytes
    are never loaded.

    Raises:
        FinalizeError: If any declared image path does not exist.
    """

    root = Path(dataset_root)
    table = compact_specialist_shards(
        "frames", _shard_paths_for_kind(batch_manifests, "frames"), output
    )
    missing = [
        str(value)
        for value in table["image_path"]
        if not (Path(str(value)) if Path(str(value)).is_absolute() else root / str(value)).is_file()
    ]
    if missing:
        raise FinalizeError(
            f"{len(missing)} retained image path(s) are missing, e.g. {missing[:5]}"
        )
    return table


def compact_batch_embeddings(
    batch_manifests: Sequence[BatchManifest], kind: str
) -> tuple[np.ndarray, pd.DataFrame, str]:
    """Concatenate one embedding ``kind`` ('visual', 'context', 'asr_segments')
    across every committed batch's already-validated index.

    Returns:
        A ``(vectors, mapping, model_name)`` tuple, where ``model_name`` is the
        single encoder name shared by every compacted batch.

    Raises:
        FinalizeError: If a batch's video_ids overlap another batch's, its
            embedding dimension/model lineage disagrees, or any vector is
            non-finite.
    """

    is_segment_kind = kind == "asr_segments"
    loader = SegmentDenseIndex.load if is_segment_kind else DenseIndex.load

    seen_video_ids: set[str] = set()
    vector_parts: list[np.ndarray] = []
    mapping_parts: list[pd.DataFrame] = []
    expected_dim: int | None = None
    expected_model: str | None = None

    for manifest in batch_manifests:
        index = loader(manifest.root / kind)
        overlap = seen_video_ids & set(index.mapping["video_id"])
        if overlap:
            raise FinalizeError(
                f"batch {manifest.batch_id} {kind} overlaps video_id(s) already "
                f"compacted: {sorted(overlap)}"
            )
        seen_video_ids.update(index.mapping["video_id"])

        if expected_dim is None:
            expected_dim = index.metadata.embedding_dim
            expected_model = index.metadata.model_name
        elif index.metadata.embedding_dim != expected_dim:
            raise FinalizeError(
                f"batch {manifest.batch_id} {kind} embedding_dim "
                f"{index.metadata.embedding_dim} disagrees with {expected_dim}"
            )
        elif index.metadata.model_name != expected_model:
            raise FinalizeError(
                f"batch {manifest.batch_id} {kind} model_name "
                f"{index.metadata.model_name!r} disagrees with {expected_model!r}"
            )

        vector_parts.append(np.asarray(index.vectors))
        mapping_parts.append(index.mapping)

    vectors = np.concatenate(vector_parts, axis=0)
    mapping = pd.concat(mapping_parts, ignore_index=True)
    mapping = mapping.assign(embedding_index=np.arange(len(mapping)))
    if not np.all(np.isfinite(vectors)):
        raise FinalizeError(f"compacted {kind} vectors contain non-finite values")

    logger.info("compacted %s embeddings: %d vector(s), dim=%s", kind, len(vectors), expected_dim)
    assert expected_model is not None
    return vectors, mapping, expected_model


def build_dense_index_from_precomputed(
    vectors: np.ndarray,
    mapping: pd.DataFrame,
    output_dir: str | Path,
    *,
    dataset_version: str,
    model_name: str,
) -> DenseIndex:
    """Build, publish, and checksum-load a global frame-native dense index."""

    DenseIndex.build(
        vectors, mapping, dataset_version=dataset_version, model_name=model_name
    ).save(output_dir)
    return DenseIndex.load(output_dir)


def build_segment_index_from_precomputed(
    vectors: np.ndarray,
    mapping: pd.DataFrame,
    output_dir: str | Path,
    *,
    dataset_version: str,
    model_name: str,
) -> SegmentDenseIndex:
    """Build, publish, and checksum-load a global ASR segment dense index."""

    SegmentDenseIndex.build(
        vectors, mapping, dataset_version=dataset_version, model_name=model_name
    ).save(output_dir)
    return SegmentDenseIndex.load(output_dir)


def finalize_corpus(
    state_store: PipelineStateStore,
    archive_ids: Sequence[str],
    batches_root: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    dataset_version: str,
) -> dict[str, object]:
    """Compact every committed local batch into the final corpus and indexes.

    Requires every archive in ``archive_ids`` to be ``cleaned``. Writes corpus
    Parquet tables, global visual/context/ASR indexes, and an atomic
    ``finalize_report.json``.

    Raises:
        FinalizeError: If any archive is not cleaned, batches overlap, or a
            compaction/index step is inconsistent.
    """

    require_full_plan_cleaned(state_store, archive_ids)
    batch_manifests = discover_committed_batches(batches_root)
    if not batch_manifests:
        raise FinalizeError("no committed batches found; nothing to finalize")

    output_root = Path(output_root)
    corpus_dir = output_root / "corpus"
    indexes_dir = output_root / "indexes"

    frame_counts: dict[str, int] = {}
    for kind in FRAME_NATIVE_TABLE_NAMES:
        table = compact_specialist_shards(
            kind, _shard_paths_for_kind(batch_manifests, kind), corpus_dir / f"{kind}.parquet"
        )
        frame_counts[kind] = len(table)
    for kind in CHILD_TABLE_NAMES:
        table = compact_specialist_shards(
            kind, _shard_paths_for_kind(batch_manifests, kind), corpus_dir / f"{kind}.parquet"
        )
        frame_counts[kind] = len(table)

    vector_counts: dict[str, int] = {}
    for kind in _EMBEDDING_KINDS:
        vectors, mapping, model_name = compact_batch_embeddings(batch_manifests, kind)
        index = build_dense_index_from_precomputed(
            vectors,
            mapping,
            indexes_dir / kind,
            dataset_version=dataset_version,
            model_name=model_name,
        )
        vector_counts[kind] = index.metadata.vector_count

    asr_vectors, asr_mapping, asr_model_name = compact_batch_embeddings(batch_manifests, "asr_segments")
    asr_index = build_segment_index_from_precomputed(
        asr_vectors,
        asr_mapping,
        indexes_dir / "asr_segments",
        dataset_version=dataset_version,
        model_name=asr_model_name,
    )
    vector_counts["asr_segments"] = asr_index.metadata.vector_count

    report = {
        "dataset_version": dataset_version,
        "archive_count": len(archive_ids),
        "batch_count": len(batch_manifests),
        "video_count": sum(len(manifest.video_ids) for manifest in batch_manifests),
        "frame_counts": frame_counts,
        "vector_counts": vector_counts,
        "generated_at": _now(),
    }
    atomic_write(output_root / "reports" / _FINALIZE_REPORT_FILENAME, lambda p: write_json(report, p))
    logger.info(
        "finalized corpus %s: %d batches, %d videos", dataset_version, report["batch_count"], report["video_count"]
    )
    return report


__all__ = [
    "BatchManifest",
    "FinalizeError",
    "build_dense_index_from_precomputed",
    "build_segment_index_from_precomputed",
    "compact_batch_embeddings",
    "compact_frame_metadata",
    "compact_specialist_shards",
    "discover_committed_batches",
    "finalize_corpus",
    "require_full_plan_cleaned",
]
