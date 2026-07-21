"""Exact FAISS ``IndexFlatIP`` over normalized visual frame embeddings."""

from __future__ import annotations

import numpy as np
import faiss
import pandas as pd

from pathlib import Path
from typing import Any

from hcmai.common.utils.io import read_json, write_json
from hcmai.common.utils.logging import get_logger
from hcmai.common.utils.timing import Timer
from hcmai.retriever.models import IndexMetadata

logger = get_logger(__name__)

# Artifact filenames written under an index directory, kept together so the
# builder, loader, and downstream retriever agree on the on-disk layout.
INDEX_FILENAME      = "visual.index"
MAPPING_FILENAME    = "frame_mapping.parquet"
METADATA_FILENAME   = "metadata.json"


class VisualIndex:
    """Build, persist, load, and search an exact inner-product frame index.

    Vectors are assumed to be L2-normalized, so inner-product scores are
    equivalent to cosine similarity. Only ``IndexFlatIP`` is used: the exact
    baseline must be measured before any IVF/PQ approximation is introduced.
    """

    def __init__(self, index: Any, mapping: pd.DataFrame, metadata: IndexMetadata) -> None:
        """Wrap a live FAISS index with its frame mapping and metadata.

        The mapping is sorted by ``embedding_index`` so that FAISS position
        ``i`` always resolves to row ``i`` of :attr:`mapping`.
        """
        self.index = index
        self.mapping = mapping.sort_values("embedding_index").reset_index(drop=True)
        self.metadata = metadata

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        mapping: pd.DataFrame,
        *,
        dataset_version: str,
        model_name: str,
        index_type: str = "flat_ip",
    ) -> VisualIndex:
        """Build an exact ``IndexFlatIP`` from normalized embeddings.

        Args:
            embeddings: Array of shape (N, dim) with L2-normalized rows.
            mapping: Frame mapping with an ``embedding_index`` column of
                positions ``0..N-1`` and one row per embedding.
            dataset_version: Dataset version to couple to the index artifact.
            model_name: Encoder checkpoint that produced the embeddings.
            index_type: Index family; only ``flat_ip`` is supported.

        Returns:
            A ready-to-search :class:`VisualIndex`.
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
        index.add(vectors)
        build_time_sec = timer.stop() / 1000.0

        metadata = IndexMetadata(
            dataset_version=dataset_version,
            model_name=model_name,
            index_type=index_type,
            metric="inner_product",
            normalization="l2",
            embedding_dim=embedding_dim,
            vector_count=vector_count,
            build_time_sec=build_time_sec,
            index_size_bytes=0,  # Filled in by save() once the file exists.
            generated_at=pd.Timestamp.now().isoformat(),
        )
        logger.info(f"Index built in {build_time_sec:.3f}s")
        return cls(index, ordered, metadata)

    def save(self, output_dir: Path | str) -> Path:
        """Serialize the index, mapping, and metadata to ``output_dir``.

        Returns:
            The directory the artifacts were written to.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        index_path = output_dir / INDEX_FILENAME
        faiss.write_index(self.index, str(index_path))
        self.mapping.to_parquet(output_dir / MAPPING_FILENAME)

        # Record the on-disk index size now that the file exists so the
        # metadata reports the real artifact size.
        self.metadata.index_size_bytes = index_path.stat().st_size
        write_json(self.metadata.to_dict(), output_dir / METADATA_FILENAME)

        logger.info(f"Saved index ({self.metadata.index_size_bytes} bytes), mapping, and metadata to {output_dir}")
        return output_dir

    @classmethod
    def load(cls, index_dir: Path | str) -> VisualIndex:
        """Load an index directory and reject mismatched artifacts.

        Args:
            index_dir: Directory containing ``visual.index``,
                ``frame_mapping.parquet``, and ``metadata.json``.

        Raises:
            ValueError: If the index vector count, mapping length, and metadata
                ``vector_count`` disagree, or positions are not ``0..N-1``.
        """

        index_dir = Path(index_dir)
        index = faiss.read_index(str(index_dir / INDEX_FILENAME))
        mapping = pd.read_parquet(index_dir / MAPPING_FILENAME)
        metadata = IndexMetadata.from_dict(read_json(index_dir / METADATA_FILENAME))

        # Cross-check the three artifacts so a stale or mispaired index is
        # rejected with a clear error instead of returning wrong frames.
        if not (index.ntotal == len(mapping) == metadata.vector_count):
            raise ValueError(
                "Mismatched index artifacts: "
                f"index.ntotal={index.ntotal}, mapping_rows={len(mapping)}, "
                f"metadata.vector_count={metadata.vector_count}"
            )
        positions = mapping["embedding_index"].to_numpy()
        if sorted(positions.tolist()) != list(range(len(mapping))):
            raise ValueError("Loaded mapping embedding_index must be a permutation of 0..N-1")

        logger.info(
            f"Loaded index from {index_dir}: {index.ntotal} vectors, "
            f"model={metadata.model_name}, version={metadata.dataset_version}"
        )
        return cls(index, mapping, metadata)

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
