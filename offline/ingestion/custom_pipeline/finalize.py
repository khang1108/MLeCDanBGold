"""Compact committed local batches into deterministic global corpus artifacts.

Streams already-committed batch Parquet/vector artifacts in deterministic
``(video_id, frame_id)`` order into corpus-wide specialist tables and global
visual/context/ASR indexes. Never re-runs model inference, never re-embeds,
and never byte-merges FAISS index files directly; global indexes are always
rebuilt from precomputed, validated vectors.
"""

from __future__ import annotations

import gc
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import faiss
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from hcmai.common.utils.io import atomic_write, read_json, write_json, write_parquet
from hcmai.common.utils.logging import get_logger
from offline.ingestion.custom_pipeline.shards import CHILD_TABLE_NAMES, FRAME_NATIVE_TABLE_NAMES
from offline.ingestion.custom_pipeline.state import ArchiveStage, PipelineStateStore
from hcmai.retrieval.retriever.dense.index import DenseIndex
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex
from hcmai.retrieval.retriever.models.metadata import IndexMetadata

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


@dataclass(frozen=True)
class ChunkedEmbeddings:
    """Memory-mapped global vectors plus their canonical mapping and chunk spans.

    ``vectors`` remains backed by a temporary ``.npy`` file.  Only the batch
    chunk currently being copied and the compact mapping table need resident
    memory; the complete vector matrix is never materialized as a RAM array.
    """

    vectors: np.memmap
    mapping: pd.DataFrame
    model_name: str
    model_revision: str | None
    batch_spans: tuple[tuple[int, int], ...]


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


def _batch_chunks(
    batch_manifests: Sequence[BatchManifest], batch_chunk_size: int
) -> list[Sequence[BatchManifest]]:
    """Split committed batches into deterministic, non-empty bounded chunks."""

    if batch_chunk_size < 1:
        raise ValueError("batch_chunk_size must be positive")
    return [
        batch_manifests[start : start + batch_chunk_size]
        for start in range(0, len(batch_manifests), batch_chunk_size)
    ]


def compact_specialist_batches(
    kind: str,
    batch_manifests: Sequence[BatchManifest],
    output: str | Path,
    *,
    batch_chunk_size: int,
) -> int:
    """Write specialist shards incrementally with at most N batches in memory.

    Frame-native chunks retain deterministic ``(video_id, frame_id)`` order
    and are checked for duplicate identities across chunk boundaries. Child
    tables preserve committed shard order and may contain zero rows.

    Returns:
        Total number of rows written to the final Parquet artifact.
    """

    chunks = _batch_chunks(batch_manifests, batch_chunk_size)
    shard_paths = _shard_paths_for_kind(batch_manifests, kind)
    if not shard_paths:
        raise FinalizeError(f"no {kind} shards supplied")
    for path in shard_paths:
        if not path.is_file():
            raise FinalizeError(f"missing {kind} shard: {path}")

    # Unifying lightweight Parquet schemas first avoids letting an all-empty
    # child-table chunk establish ``null`` columns that reject later rows.
    schema = pa.unify_schemas([pq.read_schema(path) for path in shard_paths])
    seen_frame_ids: set[str] = set()
    previous_key: tuple[str, str] | None = None
    row_count = 0

    def write_chunks(temporary: Path) -> None:
        nonlocal previous_key, row_count

        with pq.ParquetWriter(temporary, schema=schema) as writer:
            for chunk_number, manifest_chunk in enumerate(chunks, start=1):
                paths = _shard_paths_for_kind(manifest_chunk, kind)
                table = pd.concat(
                    [pd.read_parquet(path) for path in paths],
                    ignore_index=True,
                )
                if kind in FRAME_NATIVE_TABLE_NAMES:
                    if table["frame_id"].duplicated().any():
                        raise FinalizeError(
                            f"duplicate frame_id found while compacting {kind}"
                        )
                    frame_ids = set(table["frame_id"].astype(str))
                    overlap = seen_frame_ids & frame_ids
                    if overlap:
                        raise FinalizeError(
                            f"duplicate frame_id found while compacting {kind}: "
                            f"{sorted(overlap)[:5]}"
                        )
                    seen_frame_ids.update(frame_ids)
                    table = table.sort_values(["video_id", "frame_id"]).reset_index(
                        drop=True
                    )
                    if not table.empty:
                        first_key = (
                            str(table.iloc[0]["video_id"]),
                            str(table.iloc[0]["frame_id"]),
                        )
                        last_key = (
                            str(table.iloc[-1]["video_id"]),
                            str(table.iloc[-1]["frame_id"]),
                        )
                        if previous_key is not None and first_key < previous_key:
                            raise FinalizeError(
                                f"batch chunks are not globally ordered for {kind}"
                            )
                        previous_key = last_key

                arrow_table = pa.Table.from_pandas(
                    table,
                    schema=schema,
                    preserve_index=False,
                    safe=True,
                )
                writer.write_table(arrow_table)
                row_count += len(table)
                logger.info(
                    "compacting %s chunk=%d/%d batches=%d rows=%d total_rows=%d",
                    kind,
                    chunk_number,
                    len(chunks),
                    len(manifest_chunk),
                    len(table),
                    row_count,
                )

    atomic_write(output, write_chunks)
    logger.info("compacted %s: %d row(s) -> %s", kind, row_count, output)
    return row_count


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


def compact_batch_embeddings_to_memmap(
    batch_manifests: Sequence[BatchManifest],
    kind: str,
    output: str | Path,
    *,
    batch_chunk_size: int,
) -> ChunkedEmbeddings:
    """Copy validated batch vectors into one global memory-mapped ``.npy``.

    Metadata is scanned first to allocate the exact final shape. Each
    committed index is then checksum-loaded and released within its configured
    batch chunk. Global ``embedding_index`` values are rewritten contiguously
    without concatenating vector arrays in RAM.
    """

    chunks = _batch_chunks(batch_manifests, batch_chunk_size)
    is_segment_kind = kind == "asr_segments"
    loader = SegmentDenseIndex.load if is_segment_kind else DenseIndex.load
    identity_column = "segment_id" if is_segment_kind else "frame_id"

    total_vectors = 0
    expected_dim: int | None = None
    expected_model: str | None = None
    expected_revision: str | None = None
    expected_batch_version: str | None = None
    for manifest in batch_manifests:
        metadata_path = manifest.root / kind / "metadata.json"
        if not metadata_path.is_file():
            raise FinalizeError(f"missing {kind} metadata: {metadata_path}")
        metadata = IndexMetadata.from_dict(read_json(metadata_path))
        if expected_batch_version is None:
            expected_batch_version = metadata.dataset_version
        elif metadata.dataset_version != expected_batch_version:
            raise FinalizeError(
                f"batch {manifest.batch_id} {kind} dataset_version "
                f"{metadata.dataset_version!r} disagrees with "
                f"{expected_batch_version!r}"
            )
        if expected_dim is None:
            expected_dim = metadata.embedding_dim
            expected_model = metadata.model_name
            expected_revision = metadata.model_revision
        elif metadata.embedding_dim != expected_dim:
            raise FinalizeError(
                f"batch {manifest.batch_id} {kind} embedding_dim "
                f"{metadata.embedding_dim} disagrees with {expected_dim}"
            )
        elif metadata.model_name != expected_model:
            raise FinalizeError(
                f"batch {manifest.batch_id} {kind} model_name "
                f"{metadata.model_name!r} disagrees with {expected_model!r}"
            )
        elif metadata.model_revision != expected_revision:
            raise FinalizeError(
                f"batch {manifest.batch_id} {kind} model_revision differs"
            )
        total_vectors += metadata.vector_count

    if total_vectors < 1 or expected_dim is None or expected_model is None:
        raise FinalizeError(f"no {kind} vectors supplied")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vectors = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_vectors, expected_dim),
    )
    mapping_chunks: list[pd.DataFrame] = []
    batch_spans: list[tuple[int, int]] = []
    seen_video_ids: set[str] = set()
    seen_identities: set[str] = set()
    cursor = 0

    for chunk_number, manifest_chunk in enumerate(chunks, start=1):
        chunk_start = cursor
        chunk_mappings: list[pd.DataFrame] = []
        for manifest in manifest_chunk:
            index = loader(manifest.root / kind)
            metadata = index.metadata
            if (
                metadata.embedding_dim != expected_dim
                or metadata.model_name != expected_model
                or metadata.model_revision != expected_revision
            ):
                raise FinalizeError(
                    f"batch {manifest.batch_id} {kind} changed after metadata scan"
                )

            ordered_mapping = index.mapping.sort_values(
                "embedding_index"
            ).reset_index(drop=True)
            batch_vectors = index.vectors
            if len(batch_vectors) != len(ordered_mapping):
                raise FinalizeError(
                    f"batch {manifest.batch_id} {kind} vector/mapping count differs"
                )
            if not np.isfinite(batch_vectors).all():
                raise FinalizeError(
                    f"batch {manifest.batch_id} {kind} vectors contain non-finite values"
                )

            video_ids = set(ordered_mapping["video_id"].astype(str))
            overlap = seen_video_ids & video_ids
            if overlap:
                raise FinalizeError(
                    f"batch {manifest.batch_id} {kind} overlaps video_id(s) already "
                    f"compacted: {sorted(overlap)}"
                )
            seen_video_ids.update(video_ids)

            identities = set(ordered_mapping[identity_column].astype(str))
            identity_overlap = seen_identities & identities
            if identity_overlap:
                raise FinalizeError(
                    f"batch {manifest.batch_id} {kind} duplicates {identity_column}: "
                    f"{sorted(identity_overlap)[:5]}"
                )
            seen_identities.update(identities)

            end = cursor + len(batch_vectors)
            vectors[cursor:end] = batch_vectors
            global_mapping = ordered_mapping.copy()
            global_mapping["embedding_index"] = np.arange(
                cursor, end, dtype=np.int64
            )
            chunk_mappings.append(global_mapping)
            cursor = end

            # FAISS batch indexes and their mmap handles must not accumulate
            # across a chunk; only the copied disk-backed global slice remains.
            del global_mapping, ordered_mapping, batch_vectors, index

        mapping_chunks.append(pd.concat(chunk_mappings, ignore_index=True))
        batch_spans.append((chunk_start, cursor))
        vectors.flush()
        gc.collect()
        logger.info(
            "compacted %s embeddings chunk=%d/%d batches=%d vectors=%d total_vectors=%d",
            kind,
            chunk_number,
            len(chunks),
            len(manifest_chunk),
            cursor - chunk_start,
            cursor,
        )

    if cursor != total_vectors:
        raise FinalizeError(
            f"compacted {kind} count {cursor} disagrees with metadata total "
            f"{total_vectors}"
        )
    mapping = pd.concat(mapping_chunks, ignore_index=True)
    vectors.flush()
    logger.info(
        "compacted %s embeddings to memmap: %d vector(s), dim=%d",
        kind,
        total_vectors,
        expected_dim,
    )
    return ChunkedEmbeddings(
        vectors=vectors,
        mapping=mapping,
        model_name=expected_model,
        model_revision=expected_revision,
        batch_spans=tuple(batch_spans),
    )


def build_index_from_chunked_embeddings(
    compacted: ChunkedEmbeddings,
    output_dir: str | Path,
    *,
    dataset_version: str,
    retrieval_source: str,
) -> int:
    """Build and checksum-load one exact FAISS index by sequential chunk adds."""

    mapping = compacted.mapping.sort_values("embedding_index").reset_index(drop=True)
    vector_count, embedding_dim = compacted.vectors.shape
    if len(mapping) != vector_count:
        raise FinalizeError("global vector/mapping count differs")
    positions = mapping["embedding_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(positions, np.arange(vector_count, dtype=np.int64)):
        raise FinalizeError("global embedding_index must be contiguous from zero")

    is_segment = retrieval_source == "asr"
    identity_column = "segment_id" if is_segment else "frame_id"
    if mapping[identity_column].duplicated().any():
        raise FinalizeError(f"global mapping contains duplicate {identity_column}")

    started = monotonic()
    faiss_index = faiss.IndexFlatIP(embedding_dim)
    for chunk_number, (start, end) in enumerate(compacted.batch_spans, start=1):
        block = np.asarray(compacted.vectors[start:end], dtype=np.float32)
        if not block.flags.c_contiguous:
            block = np.ascontiguousarray(block)
        faiss_index.add(block)
        logger.info(
            "building %s FAISS chunk=%d/%d vectors=%d ntotal=%d",
            retrieval_source,
            chunk_number,
            len(compacted.batch_spans),
            end - start,
            faiss_index.ntotal,
        )
        del block

    metadata = IndexMetadata(
        dataset_version=dataset_version,
        model_name=compacted.model_name,
        model_revision=compacted.model_revision,
        index_type="flat_ip",
        metric="inner_product",
        normalization="l2",
        embedding_dim=embedding_dim,
        vector_count=vector_count,
        build_time_sec=monotonic() - started,
        index_size_bytes=0,
        generated_at=_now(),
        schema_version="dense-index-v2",
        entity_kind="segment" if is_segment else "frame",
        retrieval_source=retrieval_source,
    )
    index_type = SegmentDenseIndex if is_segment else DenseIndex
    bundle = index_type(
        faiss_index,
        mapping,
        metadata,
        vectors=compacted.vectors,
    )
    bundle.save(output_dir)

    # Release the build copy before checksum-loading the published FAISS file;
    # holding both complete IndexFlatIP instances caused the prior OOM crash.
    del bundle, faiss_index, mapping
    gc.collect()
    validated = index_type.load(output_dir)
    validated_count = validated.metadata.vector_count
    del validated
    gc.collect()
    return validated_count


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
    batch_chunk_size: int = 16,
) -> dict[str, object]:
    """Compact every committed local batch into the final corpus and indexes.

    Requires every archive in ``archive_ids`` to be ``cleaned``. Writes corpus
    Parquet tables, global visual/context/ASR indexes, and an atomic
    ``finalize_report.json``. At most ``batch_chunk_size`` committed batches
    are read for each Parquet/vector compaction chunk.

    Raises:
        FinalizeError: If any archive is not cleaned, batches overlap, or a
            compaction/index step is inconsistent.
    """

    require_full_plan_cleaned(state_store, archive_ids)
    batch_manifests = discover_committed_batches(batches_root)
    if not batch_manifests:
        raise FinalizeError("no committed batches found; nothing to finalize")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    corpus_dir = output_root / "corpus"
    indexes_dir = output_root / "indexes"

    frame_counts: dict[str, int] = {}
    for kind in FRAME_NATIVE_TABLE_NAMES:
        frame_counts[kind] = compact_specialist_batches(
            kind,
            batch_manifests,
            corpus_dir / f"{kind}.parquet",
            batch_chunk_size=batch_chunk_size,
        )
    for kind in CHILD_TABLE_NAMES:
        frame_counts[kind] = compact_specialist_batches(
            kind,
            batch_manifests,
            corpus_dir / f"{kind}.parquet",
            batch_chunk_size=batch_chunk_size,
        )

    vector_counts: dict[str, int] = {}
    with TemporaryDirectory(prefix=".finalize-work-", dir=output_root) as work:
        work_root = Path(work)
        for kind in _EMBEDDING_KINDS:
            compacted = compact_batch_embeddings_to_memmap(
                batch_manifests,
                kind,
                work_root / f"{kind}_vectors.npy",
                batch_chunk_size=batch_chunk_size,
            )
            vector_counts[kind] = build_index_from_chunked_embeddings(
                compacted,
                indexes_dir / kind,
                dataset_version=dataset_version,
                retrieval_source=kind,
            )
            del compacted
            gc.collect()

        asr_compacted = compact_batch_embeddings_to_memmap(
            batch_manifests,
            "asr_segments",
            work_root / "asr_segment_vectors.npy",
            batch_chunk_size=batch_chunk_size,
        )
        vector_counts["asr_segments"] = build_index_from_chunked_embeddings(
            asr_compacted,
            indexes_dir / "asr_segments",
            dataset_version=dataset_version,
            retrieval_source="asr",
        )
        del asr_compacted
        gc.collect()

    report = {
        "dataset_version": dataset_version,
        "archive_count": len(archive_ids),
        "batch_count": len(batch_manifests),
        "video_count": sum(len(manifest.video_ids) for manifest in batch_manifests),
        "frame_counts": frame_counts,
        "vector_counts": vector_counts,
        "batch_chunk_size": batch_chunk_size,
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
    "build_index_from_chunked_embeddings",
    "build_segment_index_from_precomputed",
    "compact_batch_embeddings",
    "compact_batch_embeddings_to_memmap",
    "compact_frame_metadata",
    "compact_specialist_batches",
    "compact_specialist_shards",
    "discover_committed_batches",
    "finalize_corpus",
    "require_full_plan_cleaned",
]
