"""Modality-neutral exact FAISS index over normalized frame embeddings."""

from __future__ import annotations

import shutil
from tempfile import mkdtemp

import numpy as np
import faiss
import pandas as pd
from tqdm.auto import tqdm

from functools import cached_property
from pathlib import Path
from typing import Any

from hcmai.common.utils.io import read_json, write_json
from hcmai.common.utils.logging import get_logger
from hcmai.common.utils.timing import Timer
from hcmai.retrieval.retriever.artifacts import publish_directory, sha256_file
from hcmai.retrieval.retriever.models.metadata import IndexMetadata

logger = get_logger(__name__)

# Artifact filenames written under an index directory, kept together so the
# builder, loader, and downstream retriever agree on the on-disk layout.
INDEX_FILENAME = "dense.index"
MAPPING_FILENAME = "frame_mapping.parquet"
METADATA_FILENAME = "metadata.json"
VECTORS_FILENAME = "vectors.npy"
POSTING_VIDEO_IDS_FILENAME = "posting_video_ids.json"
POSTING_OFFSETS_FILENAME = "posting_offsets.npy"
POSTING_POSITIONS_FILENAME = "posting_positions.npy"
TIMESTAMPS_FILENAME = "timestamps.npy"

REQUIRED_INDEX_FILENAMES = (
    INDEX_FILENAME,
    MAPPING_FILENAME,
    METADATA_FILENAME,
    VECTORS_FILENAME,
    POSTING_VIDEO_IDS_FILENAME,
    POSTING_OFFSETS_FILENAME,
    POSTING_POSITIONS_FILENAME,
    TIMESTAMPS_FILENAME,
)

# Metadata is committed last and describes the complete contents of these
# frame-native artifacts. Its own checksum would be self-referential.
CHECKSUM_FILENAMES = tuple(
    filename for filename in REQUIRED_INDEX_FILENAMES if filename != METADATA_FILENAME
)


class IndexArtifactError(RuntimeError):
    """A persisted retrieval index bundle is incomplete or inconsistent."""


class DenseIndex:
    """Build, persist, load, and search an exact inner-product frame index.

    Vectors are assumed to be L2-normalized, so inner-product scores are
    equivalent to cosine similarity. Only ``IndexFlatIP`` is used: the exact
    baseline must be measured before any IVF/PQ approximation is introduced.
    """

    def __init__(
        self,
        index: Any,
        mapping: pd.DataFrame,
        metadata: IndexMetadata,
        vectors: np.ndarray | None = None,
        posting_video_ids: list[str] | None = None,
        posting_offsets: np.ndarray | None = None,
        posting_positions: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
    ) -> None:
        """Wrap a live FAISS index with its frame mapping and metadata.

        The mapping is sorted by ``embedding_index`` so that FAISS position
        ``i`` always resolves to row ``i`` of :attr:`mapping`.
        """
        self.index = index
        self.mapping = mapping.sort_values("embedding_index").reset_index(drop=True)
        self.metadata = metadata
        self.vectors = vectors if vectors is not None else _reconstruct(index)
        if posting_video_ids is None:
            posting_video_ids, posting_offsets, posting_positions = _postings(
                self.mapping
            )
        self.posting_video_ids = posting_video_ids
        self.posting_offsets = _int64_array(posting_offsets)
        self.posting_positions = _int64_array(posting_positions)
        self.timestamps = (
            _int64_array(timestamps)
            if timestamps is not None
            else self.mapping["timestamp_ms"].to_numpy(dtype=np.int64)
        )
        self._video_slices = {
            video_id: slice(
                int(self.posting_offsets[index]),
                int(self.posting_offsets[index + 1]),
            )
            for index, video_id in enumerate(self.posting_video_ids)
        }

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        mapping: pd.DataFrame,
        *,
        dataset_version: str,
        model_name: str,
        model_revision: str | None = None,
        index_type: str = "flat_ip",
        show_progress: bool = False,
    ) -> DenseIndex:
        """Build an exact ``IndexFlatIP`` from normalized embeddings.

        Args:
            embeddings: Array of shape (N, dim) with L2-normalized rows.
            mapping: Frame mapping with an ``embedding_index`` column of
                positions ``0..N-1`` and one row per embedding.
            dataset_version: Dataset version to couple to the index artifact.
            model_name: Encoder checkpoint that produced the embeddings.
            model_revision: Optional immutable revision of that encoder.
            index_type: Index family; only ``flat_ip`` is supported.

        Returns:
            A ready-to-search :class:`DenseIndex`.
        """

        if index_type != "flat_ip":
            raise ValueError(f"Unsupported index_type {index_type!r}; only 'flat_ip' is supported")

        # Validate that embeddings and mapping describe the same corpus. These
        # checks are the contract for "no duplicate frame IDs" and
        # "vector_position is 0..N-1"; they run once here at build time.
        vector_count = int(embeddings.shape[0])
        if vector_count != len(mapping):
            raise ValueError(f"embedding count ({vector_count}) does not match mapping rows ({len(mapping)})")
        positions = mapping["embedding_index"].to_numpy()
        if sorted(positions.tolist()) != list(range(vector_count)):
            raise ValueError("mapping embedding_index must be a permutation of 0..N-1")
        if mapping["frame_id"].duplicated().any():
            raise ValueError("mapping contains duplicate frame_id values")

        # FAISS requires C-contiguous float32 input; adding rows in
        # embedding_index order keeps position i aligned with mapping row i.
        ordered = mapping.sort_values("embedding_index").reset_index(drop=True)
        vectors = np.ascontiguousarray(embeddings[ordered["embedding_index"].to_numpy()], dtype=np.float32)
        embedding_dim = int(vectors.shape[1])

        logger.info(f"Building IndexFlatIP: {vector_count} vectors, dim={embedding_dim}")
        timer = Timer()
        index = faiss.IndexFlatIP(embedding_dim)
        with tqdm(
            total=vector_count,
            desc="Building FAISS index",
            unit="vector",
            dynamic_ncols=True,
            disable=not show_progress,
        ) as progress:
            for start in range(0, vector_count, 50_000):
                batch = vectors[start : start + 50_000]
                index.add(batch)
                progress.update(len(batch))
        build_time_sec = timer.stop() / 1000.0

        metadata = IndexMetadata(
            dataset_version=dataset_version,
            model_name=model_name,
            model_revision=model_revision,
            index_type=index_type,
            metric="inner_product",
            normalization="l2",
            embedding_dim=embedding_dim,
            vector_count=vector_count,
            build_time_sec=build_time_sec,
            index_size_bytes=0,  # Filled in by save() once the file exists.
            generated_at=pd.Timestamp.now().isoformat(),
            schema_version="dense-index-v2",
            entity_kind="frame",
        )
        logger.info(f"Index built in {build_time_sec:.3f}s")
        return cls(index, ordered, metadata, vectors=vectors)

    def save(self, output_dir: Path | str) -> Path:
        """Stage and atomically publish the complete index bundle.

        All artifact files are written to a private sibling directory first.
        The destination only changes after metadata has recorded checksums for
        every non-metadata file, preventing readers from seeing a mixed bundle.

        Returns:
            The directory where the complete bundle was published.
        """
        output_dir = Path(output_dir).resolve()
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
        )

        try:
            self._write_bundle(staging_dir)
            publish_directory(staging_dir, output_dir)
        except Exception:
            # A write failure must leave the existing published directory
            # untouched; publication owns rollback once its rename begins.
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            raise

        logger.info(
            f"Saved index ({self.metadata.index_size_bytes} bytes), mapping, "
            f"and metadata to {output_dir}"
        )
        return output_dir

    def _write_bundle(self, output_dir: Path) -> None:
        """Write one complete bundle into an unpublished staging directory.

        Metadata is deliberately written last because it checksums all other
        files. This helper does not publish or clean up its staging directory.
        """

        index_path = output_dir / INDEX_FILENAME
        faiss.write_index(self.index, str(index_path))
        self.mapping.to_parquet(output_dir / MAPPING_FILENAME)
        np.save(output_dir / VECTORS_FILENAME, np.asarray(self.vectors, dtype=np.float32))
        write_json(self.posting_video_ids, output_dir / POSTING_VIDEO_IDS_FILENAME)
        np.save(output_dir / POSTING_OFFSETS_FILENAME, self.posting_offsets)
        np.save(output_dir / POSTING_POSITIONS_FILENAME, self.posting_positions)
        np.save(output_dir / TIMESTAMPS_FILENAME, self.timestamps)

        # Record the on-disk index size now that the file exists so the
        # metadata reports the real artifact size.
        self.metadata.index_size_bytes = index_path.stat().st_size
        self.metadata.checksums = {
            filename: sha256_file(output_dir / filename)
            for filename in CHECKSUM_FILENAMES
        }
        write_json(self.metadata.to_dict(), output_dir / METADATA_FILENAME)

    @classmethod
    def load(
        cls,
        index_dir: Path | str,
    ) -> DenseIndex:
        """Load an index directory and reject mismatched artifacts.

        Args:
            index_dir: Directory containing the complete immutable index
                bundle written by :meth:`save` in the offline GPU pipeline.

        Raises:
            IndexArtifactError: If required files are missing or persisted
                index, mapping, vectors, postings, and metadata disagree.
        """

        index_dir = Path(index_dir)
        missing = [
            filename
            for filename in REQUIRED_INDEX_FILENAMES
            if not (index_dir / filename).is_file()
        ]
        if missing:
            names = ", ".join(missing)
            raise IndexArtifactError(
                f"Incomplete index bundle at {index_dir}: missing {names}. "
                "Rebuild and synchronize the complete artifact from the "
                "offline GPU pipeline."
            )

        index = faiss.read_index(str(index_dir / INDEX_FILENAME))
        mapping = pd.read_parquet(index_dir / MAPPING_FILENAME)
        metadata = IndexMetadata.from_dict(read_json(index_dir / METADATA_FILENAME))

        # Cross-check the three artifacts so a stale or mispaired index is
        # rejected with a clear error instead of returning wrong frames.
        if not (index.ntotal == len(mapping) == metadata.vector_count):
            raise IndexArtifactError(
                "Mismatched index artifacts: "
                f"index.ntotal={index.ntotal}, mapping_rows={len(mapping)}, "
                f"metadata.vector_count={metadata.vector_count}"
            )
        if index.d != metadata.embedding_dim:
            raise IndexArtifactError(
                "Mismatched index dimensions: "
                f"index.d={index.d}, metadata.embedding_dim="
                f"{metadata.embedding_dim}"
            )
        positions = mapping["embedding_index"].to_numpy()
        if sorted(positions.tolist()) != list(range(len(mapping))):
            raise IndexArtifactError(
                "Loaded mapping embedding_index must be a permutation of 0..N-1"
            )

        vectors = np.load(index_dir / VECTORS_FILENAME, mmap_mode="r")
        posting_video_ids = list(
            read_json(index_dir / POSTING_VIDEO_IDS_FILENAME)
        )
        posting_offsets = np.load(
            index_dir / POSTING_OFFSETS_FILENAME, mmap_mode="r"
        )
        posting_positions = np.load(
            index_dir / POSTING_POSITIONS_FILENAME, mmap_mode="r"
        )
        timestamps = np.load(index_dir / TIMESTAMPS_FILENAME, mmap_mode="r")

        if vectors.shape != (metadata.vector_count, metadata.embedding_dim):
            raise IndexArtifactError("Persisted vectors do not match index metadata")
        if vectors.dtype != np.float32:
            raise IndexArtifactError("Persisted vectors must use float32 dtype")
        if timestamps.shape != (metadata.vector_count,):
            raise IndexArtifactError("Persisted timestamps do not match index metadata")
        if posting_positions.shape != (metadata.vector_count,):
            raise IndexArtifactError(
                "Persisted posting positions do not match index metadata"
            )
        if posting_offsets.shape != (len(posting_video_ids) + 1,):
            raise IndexArtifactError(
                "Persisted posting offsets do not match posting video IDs"
            )
        if (
            not len(posting_offsets)
            or int(posting_offsets[0]) != 0
            or int(posting_offsets[-1]) != len(posting_positions)
            or np.any(np.diff(posting_offsets) < 0)
        ):
            raise IndexArtifactError("Persisted posting offsets are invalid")
        if len(posting_positions) and (
            int(posting_positions.min()) < 0
            or int(posting_positions.max()) >= metadata.vector_count
        ):
            raise IndexArtifactError("Persisted posting positions are out of bounds")

        logger.info(
            f"Loaded index from {index_dir}: {index.ntotal} vectors, "
            f"model={metadata.model_name}, version={metadata.dataset_version}"
        )
        return cls(
            index,
            mapping,
            metadata,
            vectors=vectors,
            posting_video_ids=posting_video_ids,
            posting_offsets=posting_offsets,
            posting_positions=posting_positions,
            timestamps=timestamps,
        )

    def search(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search the index for the nearest frames to each query vector.

        Args:
            query_vectors: Array of shape (Q, dim) with L2-normalized rows.
            top_k: Number of neighbours to return per query.

        Returns:
            A ``(scores, positions)`` pair, each of shape (Q, top_k). Positions
            index into :attr:`mapping`; FAISS pads with ``-1`` when fewer than
            ``top_k`` vectors exist.
        """
        queries = np.ascontiguousarray(query_vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        scores, positions = self.index.search(queries, min(top_k, self.index.ntotal))
        return scores, positions

    def score_subset(
        self,
        query_vectors: np.ndarray,
        positions: np.ndarray,
        chunk_size: int = 65_536,
    ) -> np.ndarray:
        """Score every query against a subset of indexed vectors, exactly.

        Args:
            query_vectors: Array of shape (Q, dim) with L2-normalized rows.
            positions: Index positions to score, as held in ``mapping``.
            chunk_size: Vectors materialized at a time, bounding peak memory.

        Returns:
            Array of shape (Q, len(positions)), column ``j`` for ``positions[j]``.
        """
        queries = np.ascontiguousarray(query_vectors, dtype=np.float32).reshape(-1, self.index.d)
        positions = np.ascontiguousarray(positions, dtype=np.int64)
        scores = np.empty((len(queries), len(positions)), dtype=np.float32)
        for start in range(0, len(positions), chunk_size):
            chunk = positions[start : start + chunk_size]
            vectors = np.asarray(self.vectors[chunk], dtype=np.float32)
            scores[:, start : start + len(chunk)] = queries @ vectors.T
        return scores

    def video_positions(self, video_id: str) -> np.ndarray:
        """Index positions of one video's frames, in canonical frame order.

        Args:
            video_id: Video to look up; an unknown id raises ``KeyError``.

        Returns:
            The video's positions sorted by ``frame_idx``, which the posting
            table does not guarantee since it is ordered by embedding index.
        """
        positions = self.posting_positions[self._video_slices[video_id]]
        return positions[np.argsort(self.frame_idx[positions], kind="stable")]

    @cached_property
    def video_ids(self) -> np.ndarray:
        """Video of each index position, sharing the mapping's string objects."""
        return self.mapping["video_id"].to_numpy()

    @cached_property
    def frame_ids(self) -> np.ndarray:
        """Frame id of each index position."""
        return self.mapping["frame_id"].to_numpy()

    @cached_property
    def frame_idx(self) -> np.ndarray:
        """In-video frame index of each index position."""
        return self.mapping["frame_idx"].to_numpy()

def _postings(
    mapping: pd.DataFrame,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    ordered = mapping.sort_values("embedding_index")
    video_ids = sorted(str(value) for value in ordered["video_id"].unique())
    groups = [
        np.sort(
            ordered.loc[
                ordered["video_id"].astype(str) == video_id,
                "embedding_index",
            ].to_numpy(dtype=np.int64)
        )
        for video_id in video_ids
    ]
    offsets = np.zeros(len(groups) + 1, dtype=np.int64)
    if groups:
        offsets[1:] = np.cumsum([len(group) for group in groups])
    positions = (
        np.concatenate(groups)
        if groups
        else np.empty(0, dtype=np.int64)
    )
    return video_ids, offsets, positions


def _reconstruct(index: Any) -> np.ndarray:
    vectors = np.empty((index.ntotal, index.d), dtype=np.float32)
    index.reconstruct_n(0, index.ntotal, vectors)
    return vectors


def _int64_array(values: Any) -> np.ndarray:
    if isinstance(values, np.ndarray) and values.dtype == np.int64:
        return values
    return np.asarray(values, dtype=np.int64)
