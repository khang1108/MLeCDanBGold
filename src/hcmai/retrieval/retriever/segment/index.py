"""Persist and search exact dense vectors for timestamped ASR segments.

This module owns a segment-native ``faiss.IndexFlatIP`` bundle.  It deliberately
does not accept or materialize ``frame_id``: ASR evidence remains identified by
``segment_id`` until an explicit downstream segment-to-frame projection.
"""

from __future__ import annotations

import shutil
from functools import cached_property
from numbers import Integral, Real
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import faiss
import numpy as np
import pandas as pd

from hcmai.common.schemas.search import SearchFilters
from hcmai.common.utils.io import read_json, write_json
from hcmai.retrieval.retriever.artifacts import publish_directory, sha256_file
from hcmai.retrieval.retriever.dense.index import IndexArtifactError
from hcmai.retrieval.retriever.filtered import exact_subset_search
from hcmai.retrieval.retriever.models.metadata import IndexMetadata

INDEX_FILENAME = "dense.index"
MAPPING_FILENAME = "segment_mapping.parquet"
METADATA_FILENAME = "metadata.json"
VECTORS_FILENAME = "vectors.npy"
POSTING_VIDEO_IDS_FILENAME = "posting_video_ids.json"
POSTING_OFFSETS_FILENAME = "posting_offsets.npy"
POSTING_POSITIONS_FILENAME = "posting_positions.npy"
START_MS_FILENAME = "start_ms.npy"
END_MS_FILENAME = "end_ms.npy"

REQUIRED_MAPPING_COLUMNS = frozenset(
    {
        "embedding_index",
        "segment_id",
        "video_id",
        "segment_index",
        "start_ms",
        "end_ms",
    }
)
REQUIRED_INDEX_FILENAMES = (
    INDEX_FILENAME,
    MAPPING_FILENAME,
    METADATA_FILENAME,
    VECTORS_FILENAME,
    POSTING_VIDEO_IDS_FILENAME,
    POSTING_OFFSETS_FILENAME,
    POSTING_POSITIONS_FILENAME,
    START_MS_FILENAME,
    END_MS_FILENAME,
)
# Metadata is written last and therefore cannot checksum itself.
CHECKSUM_FILENAMES = tuple(
    filename for filename in REQUIRED_INDEX_FILENAMES if filename != METADATA_FILENAME
)


class SegmentDenseIndex:
    """Build, persist, load, and search an exact ASR segment index.

    The index contains L2-normalized ``float32`` vectors, so its inner-product
    score is cosine similarity.  The mapping is segment-native and preserves
    half-open intervals ``[start_ms, end_ms)`` independently of frame data.
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
        start_ms: np.ndarray | None = None,
        end_ms: np.ndarray | None = None,
        subset_search_threshold: int = 100_000,
    ) -> None:
        """Wrap a validated live index and its segment-only support arrays.

        ``mapping`` is ordered by ``embedding_index`` so index position ``i``
        always resolves to mapping row ``i``.  Callers should use
        :meth:`build` or :meth:`load`; this initializer only wires an already
        validated bundle into fast filtering structures.
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
        self.start_ms = (
            _int64_array(start_ms)
            if start_ms is not None
            else self.mapping["start_ms"].to_numpy(dtype=np.int64)
        )
        self.end_ms = (
            _int64_array(end_ms)
            if end_ms is not None
            else self.mapping["end_ms"].to_numpy(dtype=np.int64)
        )
        self.subset_search_threshold = subset_search_threshold
        self._video_slices = {
            video_id: slice(
                int(self.posting_offsets[position]),
                int(self.posting_offsets[position + 1]),
            )
            for position, video_id in enumerate(self.posting_video_ids)
        }

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        mapping: pd.DataFrame,
        *,
        dataset_version: str,
        model_name: str,
        index_type: str = "flat_ip",
    ) -> SegmentDenseIndex:
        """Build an exact index from normalized ASR segment embeddings.

        Args:
            embeddings: Non-empty two-dimensional, L2-normalized ``float32``
                array with one row for each mapping row.
            mapping: Segment mapping containing the required identity and
                half-open interval columns.  Supplemental segment provenance
                columns are retained, but ``frame_id`` is forbidden.
            dataset_version: Dataset version paired with this immutable bundle.
            model_name: Name of the text encoder that produced ``embeddings``.
            index_type: Only ``flat_ip`` is supported for the exact baseline.
        """

        if index_type != "flat_ip":
            raise ValueError(
                f"Unsupported index_type {index_type!r}; only 'flat_ip' is supported"
            )
        _validate_build_inputs(embeddings, mapping)

        ordered = mapping.sort_values("embedding_index").reset_index(drop=True)
        positions = ordered["embedding_index"].to_numpy(dtype=np.int64)
        vectors = np.ascontiguousarray(embeddings[positions], dtype=np.float32)
        index = faiss.IndexFlatIP(int(vectors.shape[1]))
        index.add(vectors)

        metadata = IndexMetadata(
            dataset_version=dataset_version,
            model_name=model_name,
            index_type=index_type,
            metric="inner_product",
            normalization="l2",
            embedding_dim=int(vectors.shape[1]),
            vector_count=int(vectors.shape[0]),
            build_time_sec=0.0,
            index_size_bytes=0,
            generated_at=pd.Timestamp.now().isoformat(),
            schema_version="dense-index-v2",
            entity_kind="segment",
            retrieval_source="asr",
        )
        return cls(index, ordered, metadata, vectors=vectors)

    def save(self, output_dir: Path | str) -> Path:
        """Atomically publish the complete, checksummed segment index bundle."""

        output_dir = Path(output_dir).resolve()
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
        )
        try:
            self._write_bundle(staging_dir)
            publish_directory(staging_dir, output_dir)
        except Exception:
            # Staging failures must not alter the currently published bundle.
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        return output_dir

    def _write_bundle(self, output_dir: Path) -> None:
        """Write all artifact files into an unpublished staging directory."""

        index_path = output_dir / INDEX_FILENAME
        faiss.write_index(self.index, str(index_path))
        self.mapping.to_parquet(output_dir / MAPPING_FILENAME, index=False)
        np.save(output_dir / VECTORS_FILENAME, np.asarray(self.vectors, dtype=np.float32))
        write_json(self.posting_video_ids, output_dir / POSTING_VIDEO_IDS_FILENAME)
        np.save(output_dir / POSTING_OFFSETS_FILENAME, self.posting_offsets)
        np.save(output_dir / POSTING_POSITIONS_FILENAME, self.posting_positions)
        np.save(output_dir / START_MS_FILENAME, self.start_ms)
        np.save(output_dir / END_MS_FILENAME, self.end_ms)

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
        *,
        subset_search_threshold: int = 100_000,
    ) -> SegmentDenseIndex:
        """Load and fully validate a published segment index bundle.

        The loader verifies layout, v2 provenance, checksums, mapping identity,
        array shapes, posting bounds, intervals, and index dimensions before it
        makes retrieval available.
        """

        index_dir = Path(index_dir)
        missing = [
            filename
            for filename in REQUIRED_INDEX_FILENAMES
            if not (index_dir / filename).is_file()
        ]
        if missing:
            raise IndexArtifactError(
                f"Incomplete segment index bundle at {index_dir}: missing {', '.join(missing)}"
            )

        metadata = IndexMetadata.from_dict(read_json(index_dir / METADATA_FILENAME))
        _validate_metadata(metadata)
        _validate_checksums(index_dir, metadata)

        index = faiss.read_index(str(index_dir / INDEX_FILENAME))
        mapping = pd.read_parquet(index_dir / MAPPING_FILENAME)
        _validate_loaded_mapping(mapping, metadata.vector_count)
        vectors = np.load(index_dir / VECTORS_FILENAME, mmap_mode="r")
        posting_video_ids = _validate_posting_video_ids(
            read_json(index_dir / POSTING_VIDEO_IDS_FILENAME)
        )
        posting_offsets = np.load(index_dir / POSTING_OFFSETS_FILENAME, mmap_mode="r")
        posting_positions = np.load(index_dir / POSTING_POSITIONS_FILENAME, mmap_mode="r")
        start_ms = np.load(index_dir / START_MS_FILENAME, mmap_mode="r")
        end_ms = np.load(index_dir / END_MS_FILENAME, mmap_mode="r")

        _validate_loaded_arrays(
            index,
            metadata,
            mapping.sort_values("embedding_index").reset_index(drop=True),
            vectors,
            posting_video_ids,
            posting_offsets,
            posting_positions,
            start_ms,
            end_ms,
        )
        if not np.array_equal(
            np.asarray(start_ms, dtype=np.int64),
            mapping.sort_values("embedding_index")["start_ms"].to_numpy(dtype=np.int64),
        ) or not np.array_equal(
            np.asarray(end_ms, dtype=np.int64),
            mapping.sort_values("embedding_index")["end_ms"].to_numpy(dtype=np.int64),
        ):
            raise IndexArtifactError("Persisted segment interval arrays disagree with mapping")

        return cls(
            index,
            mapping,
            metadata,
            vectors=vectors,
            posting_video_ids=posting_video_ids,
            posting_offsets=posting_offsets,
            posting_positions=posting_positions,
            start_ms=start_ms,
            end_ms=end_ms,
            subset_search_threshold=subset_search_threshold,
        )

    def search(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return exact global top-k segment positions for each query vector."""

        queries = np.ascontiguousarray(query_vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        return self.index.search(queries, min(top_k, self.index.ntotal))

    def video_positions(self, video_id: str) -> np.ndarray:
        """Return one video's positions in stable ``segment_index`` order.

        Raises:
            KeyError: If ``video_id`` is not present in the segment bundle.
        """

        positions = self.posting_positions[self._video_slices[video_id]]
        return positions[np.argsort(self.segment_index[positions], kind="stable")]

    @cached_property
    def video_ids(self) -> np.ndarray:
        """Return the source video identifier for each index position."""

        return self.mapping["video_id"].to_numpy()

    @cached_property
    def segment_ids(self) -> np.ndarray:
        """Return the unique ASR segment identifier for each index position."""

        return self.mapping["segment_id"].to_numpy()

    @cached_property
    def segment_index(self) -> np.ndarray:
        """Return each segment's timeline order within its source video."""

        return self.mapping["segment_index"].to_numpy(dtype=np.int64)

    def search_filtered(
        self,
        query_vectors: np.ndarray,
        top_k: int,
        filters: SearchFilters | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search globally or exactly score only positions allowed by filters."""

        allowed = self.filtered_positions(filters)
        if allowed is None:
            return self.search(query_vectors, top_k)
        return exact_subset_search(
            query_vectors,
            self.vectors,
            allowed,
            top_k,
            chunk_size=self.subset_search_threshold,
        )

    def filtered_positions(self, filters: SearchFilters | None) -> np.ndarray | None:
        """Return positions overlapping a half-open requested time interval.

        A segment ``[a, b)`` overlaps a requested range ``[start, end)`` when
        ``b > start`` and ``a < end``.  Therefore merely touching a requested
        endpoint is excluded, and an explicit zero-width requested range is
        empty by definition.
        """

        if filters is None or not (
            filters.video_ids
            or filters.start_time_ms is not None
            or filters.end_time_ms is not None
        ):
            return None
        if (
            filters.start_time_ms is not None
            and filters.end_time_ms is not None
            and filters.start_time_ms == filters.end_time_ms
        ):
            return np.empty(0, dtype=np.int64)
        if filters.video_ids:
            groups = [
                self.posting_positions[bounds]
                for video_id in filters.video_ids
                if (bounds := self._video_slices.get(video_id)) is not None
            ]
            positions = (
                np.unique(np.concatenate(groups))
                if groups
                else np.empty(0, dtype=np.int64)
            )
        else:
            positions = np.arange(self.metadata.vector_count, dtype=np.int64)
        if filters.start_time_ms is not None:
            positions = positions[self.end_ms[positions] > filters.start_time_ms]
        if filters.end_time_ms is not None:
            positions = positions[self.start_ms[positions] < filters.end_time_ms]
        return positions


def _validate_build_inputs(embeddings: np.ndarray, mapping: pd.DataFrame) -> None:
    """Validate the immutable segment identity and normalized-vector contract."""

    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2 or not len(embeddings):
        raise ValueError("embeddings must be a non-empty two-dimensional array")
    if embeddings.dtype != np.float32:
        raise ValueError("segment embeddings must use float32 dtype")
    if not np.isfinite(embeddings).all():
        raise ValueError("segment embeddings must be finite")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError("segment embeddings must be L2-normalized")
    _validate_mapping(mapping, int(embeddings.shape[0]), error_type=ValueError)


def _validate_loaded_mapping(mapping: pd.DataFrame, vector_count: int) -> None:
    """Convert mapping-contract failures into artifact errors during loading."""

    _validate_mapping(mapping, vector_count, error_type=IndexArtifactError)


def _validate_mapping(
    mapping: pd.DataFrame, vector_count: int, *, error_type: type[Exception]
) -> None:
    """Check identity, index alignment, and interval invariants for mapping rows."""

    missing = sorted(REQUIRED_MAPPING_COLUMNS - set(mapping.columns))
    if missing:
        raise error_type(f"segment mapping is missing required columns: {', '.join(missing)}")
    if "frame_id" in mapping.columns:
        raise error_type("segment mapping must not contain frame_id")
    if len(mapping) != vector_count:
        raise error_type(
            f"embedding count ({vector_count}) does not match mapping rows ({len(mapping)})"
        )
    _validate_non_empty_strings(mapping, "segment_id", error_type)
    _validate_non_empty_strings(mapping, "video_id", error_type)
    for column in ("embedding_index", "segment_index", "start_ms", "end_ms"):
        _validate_integral_column(mapping, column, error_type)
    if (mapping["segment_index"] < 0).any():
        raise error_type("segment_index values must be non-negative")
    if (mapping["start_ms"] < 0).any() or (mapping["end_ms"] < 0).any():
        raise error_type("segment interval values must be non-negative")
    positions = mapping["embedding_index"].to_numpy()
    if sorted(positions.tolist()) != list(range(vector_count)):
        raise error_type("mapping embedding_index must be a permutation of 0..N-1")
    if mapping["segment_id"].duplicated().any():
        raise error_type("mapping contains duplicate segment_id values")
    if (mapping["end_ms"] <= mapping["start_ms"]).any():
        raise error_type("segment mapping requires positive duration")


def _validate_non_empty_strings(
    mapping: pd.DataFrame, column: str, error_type: type[Exception]
) -> None:
    """Ensure identity fields cannot be null, blank, or coerced to strings."""

    values = mapping[column].tolist()
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise error_type(f"segment mapping {column} values must be non-empty strings")


def _validate_integral_column(
    mapping: pd.DataFrame, column: str, error_type: type[Exception]
) -> None:
    """Reject non-integral values before any positional or interval cast occurs."""

    minimum, maximum = np.iinfo(np.int64).min, np.iinfo(np.int64).max
    for value in mapping[column].tolist():
        if isinstance(value, bool) or not isinstance(value, (Integral, Real)):
            raise error_type(f"segment mapping {column} values must be finite integers")
        if isinstance(value, Integral):
            integer_value = int(value)
        elif not np.isfinite(float(value)) or not float(value).is_integer():
            raise error_type(f"segment mapping {column} values must be finite integers")
        else:
            integer_value = int(float(value))
        if integer_value < minimum or integer_value > maximum:
            raise error_type(f"segment mapping {column} values must fit int64")


def _validate_metadata(metadata: IndexMetadata) -> None:
    """Reject a frame or legacy bundle presented as a segment ASR artifact."""

    if metadata.schema_version != "dense-index-v2":
        raise IndexArtifactError("Segment index requires dense-index-v2 metadata")
    if metadata.entity_kind != "segment":
        raise IndexArtifactError("Segment index metadata entity_kind must be 'segment'")
    if metadata.retrieval_source != "asr":
        raise IndexArtifactError("Segment index metadata retrieval_source must be 'asr'")
    if metadata.index_type != "flat_ip" or metadata.metric != "inner_product":
        raise IndexArtifactError("Segment index metadata must describe exact flat inner-product search")
    if metadata.normalization != "l2":
        raise IndexArtifactError("Segment index metadata normalization must be 'l2'")


def _validate_checksums(index_dir: Path, metadata: IndexMetadata) -> None:
    """Ensure every non-metadata file exactly matches the v2 manifest."""

    if not isinstance(metadata.checksums, dict):
        raise IndexArtifactError("Invalid checksum manifest: expected filename digests")
    expected_names = set(CHECKSUM_FILENAMES)
    actual_names = set(metadata.checksums)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected {', '.join(unexpected)}")
        raise IndexArtifactError("Invalid checksum manifest: " + "; ".join(details))
    for filename in CHECKSUM_FILENAMES:
        if sha256_file(index_dir / filename) != metadata.checksums[filename]:
            raise IndexArtifactError(f"Index artifact checksum mismatch for {filename}")


def _validate_loaded_arrays(
    index: Any,
    metadata: IndexMetadata,
    mapping: pd.DataFrame,
    vectors: np.ndarray,
    posting_video_ids: list[str],
    posting_offsets: np.ndarray,
    posting_positions: np.ndarray,
    start_ms: np.ndarray,
    end_ms: np.ndarray,
) -> None:
    """Cross-check the FAISS index and persisted support arrays before serving."""

    count = metadata.vector_count
    if not (index.ntotal == count):
        raise IndexArtifactError(
            f"Mismatched segment artifacts: index.ntotal={index.ntotal}, metadata.vector_count={count}"
        )
    if index.d != metadata.embedding_dim:
        raise IndexArtifactError("Mismatched segment index dimensions")
    if not isinstance(index, faiss.IndexFlatIP):
        raise IndexArtifactError("Segment FAISS index must be concretely IndexFlatIP")
    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise IndexArtifactError("Segment FAISS index must use inner-product metric")
    if vectors.shape != (count, metadata.embedding_dim) or vectors.dtype != np.float32:
        raise IndexArtifactError("Persisted segment vectors do not match metadata")
    if not np.isfinite(vectors).all() or not np.allclose(
        np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-4, atol=1e-5
    ):
        raise IndexArtifactError("Persisted segment vectors must be finite L2-normalized float32")
    if start_ms.shape != (count,) or end_ms.shape != (count,):
        raise IndexArtifactError("Persisted segment intervals do not match metadata")
    if start_ms.dtype != np.int64 or end_ms.dtype != np.int64:
        raise IndexArtifactError("Persisted segment interval arrays must use int64 dtype")
    if np.any(end_ms <= start_ms):
        raise IndexArtifactError("Persisted segment intervals require positive duration")
    if posting_positions.shape != (count,) or posting_positions.dtype != np.int64:
        raise IndexArtifactError("Persisted segment posting positions are invalid")
    if posting_offsets.shape != (len(posting_video_ids) + 1,) or posting_offsets.dtype != np.int64:
        raise IndexArtifactError("Persisted segment posting offsets are invalid")
    if (
        not len(posting_offsets)
        or int(posting_offsets[0]) != 0
        or int(posting_offsets[-1]) != len(posting_positions)
        or np.any(np.diff(posting_offsets) < 0)
    ):
        raise IndexArtifactError("Persisted segment posting offsets are invalid")
    if len(posting_positions) and (
        int(posting_positions.min()) < 0 or int(posting_positions.max()) >= count
    ):
        raise IndexArtifactError("Persisted segment posting positions are out of bounds")
    if len(np.unique(posting_positions)) != count:
        raise IndexArtifactError("Persisted segment posting positions must cover each vector once")
    expected_video_ids = sorted(mapping["video_id"].unique().tolist())
    if sorted(posting_video_ids) != expected_video_ids:
        raise IndexArtifactError("Persisted segment posting video IDs disagree with mapping")
    for position, video_id in enumerate(posting_video_ids):
        start = int(posting_offsets[position])
        end = int(posting_offsets[position + 1])
        posted_video_ids = mapping.iloc[posting_positions[start:end]]["video_id"]
        if not posted_video_ids.eq(video_id).all():
            raise IndexArtifactError(
                "Persisted segment posting positions disagree with their posting video IDs"
            )


def _validate_posting_video_ids(raw_value: Any) -> list[str]:
    """Return valid posting keys without accepting JSON values that coerce to IDs."""

    if not isinstance(raw_value, list):
        raise IndexArtifactError("Persisted segment posting video IDs must be a list")
    if any(not isinstance(value, str) or not value.strip() for value in raw_value):
        raise IndexArtifactError(
            "Persisted segment posting video IDs must be non-empty strings"
        )
    if len(set(raw_value)) != len(raw_value):
        raise IndexArtifactError("Persisted segment posting video IDs must be unique")
    return raw_value


def _postings(mapping: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Build deterministic video postings without changing vector positions."""

    ordered = mapping.sort_values("embedding_index")
    video_ids = sorted(str(value) for value in ordered["video_id"].unique())
    groups = [
        np.sort(
            ordered.loc[
                ordered["video_id"].astype(str) == video_id, "embedding_index"
            ].to_numpy(dtype=np.int64)
        )
        for video_id in video_ids
    ]
    offsets = np.zeros(len(groups) + 1, dtype=np.int64)
    if groups:
        offsets[1:] = np.cumsum([len(group) for group in groups])
    positions = np.concatenate(groups) if groups else np.empty(0, dtype=np.int64)
    return video_ids, offsets, positions


def _reconstruct(index: Any) -> np.ndarray:
    """Recover vectors only for in-memory construction without persisted arrays."""

    vectors = np.empty((index.ntotal, index.d), dtype=np.float32)
    index.reconstruct_n(0, index.ntotal, vectors)
    return vectors


def _int64_array(values: Any) -> np.ndarray:
    """Return a positional support array with the stable ``int64`` contract."""

    if isinstance(values, np.ndarray) and values.dtype == np.int64:
        return values
    return np.asarray(values, dtype=np.int64)


__all__ = [
    "CHECKSUM_FILENAMES",
    "IndexArtifactError",
    "SegmentDenseIndex",
]
