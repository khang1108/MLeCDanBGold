"""Build strict, resumable visual embeddings from canonical BTC keyframes.

This module owns offline visual artifact construction. It preserves the
organizer-provided frame coordinates and does not extract or reinterpret video
frames; the canonical ``frames.parquet`` table remains the identity authority.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hcmai.common.config import EncoderConfig
from hcmai.common.utils.image import load_image
from hcmai.common.utils.io import atomic_write, read_parquet, write_parquet, write_yaml
from hcmai.common.utils.logging import get_logger
from hcmai.common.utils.timing import Timer
from hcmai.retrieval.embedding.models.contracts import ImageEmbeddingAdapter
from hcmai.retrieval.embedding.models.metadata import EmbeddingMetadata
from hcmai.retrieval.embedding.models.stats import EncodingStats
from hcmai.retrieval.retriever.artifacts import fingerprint_files

logger = get_logger(__name__)

_REQUIRED_FRAME_COLUMNS = (
    "frame_id",
    "video_id",
    "frame_idx",
    "timestamp_ms",
    "keyframe_order",
    "image_path",
)


class EmbeddingArtifactBuilder:
    """Encode canonical BTC images into aligned, resumable visual artifacts.

    Shard boundaries are fixed canonical row slices. A completed shard can be
    reused only when both its frame identities and vector shape exactly match
    the current slice, preventing an interrupted or stale build from changing
    the frame-to-vector relationship.
    """

    def __init__(
        self,
        frames_path: Path | str,
        dataset_root: Path | str,
        output_dir: Path | str,
        encoder_config: EncoderConfig,
        dataset_version: str = "hcmai2026",
        encoder: ImageEmbeddingAdapter | None = None,
        *,
        strict: bool = True,
        resume: bool = True,
        shard_size: int = 2_048,
        checkpoint_dir: Path | str | None = None,
    ) -> None:
        """Configure canonical inputs, checkpoints, and encoder.

        Strict mode refuses final artifacts unless every canonical frame is
        readable and represented exactly once. Resume reuses only valid
        completed canonical-row shards. ``checkpoint_dir`` may place those
        shards outside output publication staging so failed runs remain
        resumable.
        """
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.frames_path = Path(frames_path)
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.output_dir = Path(output_dir)
        self.encoder_config = encoder_config
        self.dataset_version = dataset_version
        self.strict = strict
        self.resume = resume
        self.shard_size = shard_size
        self.embeddings_dir = self.output_dir / "embeddings"
        self.shards_dir = (
            Path(checkpoint_dir)
            if checkpoint_dir is not None
            else self.embeddings_dir / "shards"
        )
        self.embeddings_file = self.embeddings_dir / "visual_embeddings.npy"
        self.mapping_file = self.embeddings_dir / "frame_mapping.parquet"
        self.metadata_file = self.embeddings_dir / "metadata.yaml"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        if encoder is None:
            from hcmai.retrieval.embedding.adapters.siglip import SigLIPAdapter

            encoder = SigLIPAdapter(encoder_config)
        self.encoder = encoder
        self.embeddings_list: list[np.ndarray] = []
        self.frame_mapping: list[dict[str, Any]] = []
        self.failed_frames: list[dict[str, Any]] = []

    def _resolve_image(self, value: object) -> Path:
        """Resolve one canonical image path without changing frame metadata."""
        path = Path(str(value))
        resolved = path if path.is_absolute() else self.dataset_root / path
        return resolved.resolve()

    def _canonical_records(self, table: pd.DataFrame) -> list[dict[str, Any]]:
        """Validate identity-bearing columns while retaining Parquet row order."""
        missing = [name for name in _REQUIRED_FRAME_COLUMNS if name not in table]
        if missing:
            raise RuntimeError(
                "Canonical frame mapping is missing required columns: "
                + ", ".join(missing)
            )
        if table.empty:
            raise RuntimeError("Canonical frame mapping must contain at least one frame")
        frame_ids = table.loc[:, "frame_id"].to_numpy()
        if bool(pd.isna(frame_ids).any()) or bool(pd.Index(frame_ids).duplicated().any()):
            raise RuntimeError("Canonical frame mapping has missing or duplicate frame_id")
        identity_values = table.loc[:, list(_REQUIRED_FRAME_COLUMNS)].to_numpy()
        if bool(pd.isna(identity_values).any()):
            raise RuntimeError("Canonical frame mapping has missing identity values")
        records = table.to_dict(orient="records")
        for record in records:
            if not isinstance(record["frame_id"], str) or not record["frame_id"]:
                raise RuntimeError("Canonical frame mapping has invalid frame_id")
            if not isinstance(record["video_id"], str) or not record["video_id"]:
                raise RuntimeError("Canonical frame mapping has invalid video_id")
        return records

    def _shard_path(self, start: int, end: int) -> Path:
        """Return the stable checkpoint path for one canonical row interval."""
        return self.shards_dir / f"visual-{start:09d}-{end:09d}.npz"

    def _load_valid_shard(
        self,
        path: Path,
        expected_ids: list[str],
    ) -> np.ndarray | None:
        """Return vectors only when a checkpoint exactly matches its slice."""
        if not self.resume or not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as checkpoint:
                frame_ids = checkpoint["frame_ids"].tolist()
                vectors = checkpoint["vectors"]
        except (EOFError, OSError, ValueError, KeyError, zipfile.BadZipFile):
            return None
        expected_shape = (len(expected_ids), int(self.encoder.embedding_dim))
        if (
            frame_ids != expected_ids
            or vectors.shape != expected_shape
            or vectors.dtype != np.float32
            or not np.isfinite(vectors).all()
        ):
            return None
        return np.asarray(vectors, dtype=np.float32)

    def _write_shard(
        self,
        path: Path,
        frame_ids: list[str],
        vectors: np.ndarray,
    ) -> None:
        """Atomically checkpoint one complete canonical row slice."""
        atomic_write(
            path,
            lambda temporary: self._save_npz(temporary, frame_ids, vectors),
        )

    @staticmethod
    def _save_npz(path: Path, frame_ids: list[str], vectors: np.ndarray) -> None:
        """Write an NPZ through an open handle so atomic temp suffixes work."""
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                frame_ids=np.asarray(frame_ids, dtype=str),
                vectors=np.asarray(vectors, dtype=np.float32),
            )

    def _load_images(
        self,
        records: list[dict[str, Any]],
        stats: EncodingStats,
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """Load readable images and record failures without rewriting identity."""
        images: list[Any] = []
        readable_records: list[dict[str, Any]] = []
        for record in records:
            try:
                images.append(load_image(self._resolve_image(record["image_path"])))
                readable_records.append(record)
            except (OSError, ValueError) as error:
                stats.num_failed += 1
                self.failed_frames.append(
                    {"frame_id": record["frame_id"], "error": str(error)}
                )
        return images, readable_records

    def _encode_images(self, images: list[Any], stats: EncodingStats) -> np.ndarray:
        """Encode images in configured batches and validate adapter alignment."""
        batches: list[np.ndarray] = []
        for start in range(0, len(images), self.encoder_config.batch_size):
            batch = images[start : start + self.encoder_config.batch_size]
            vectors = np.asarray(self.encoder.encode_images(batch, stats), dtype=np.float32)
            expected_shape = (len(batch), int(self.encoder.embedding_dim))
            if vectors.shape != expected_shape or not np.isfinite(vectors).all():
                raise RuntimeError(
                    "Visual encoder returned vectors that do not match the canonical batch"
                )
            batches.append(vectors)
        return np.vstack(batches) if batches else np.empty((0, self.encoder.embedding_dim), dtype=np.float32)

    def _append_vectors(
        self,
        vectors: np.ndarray,
        records: list[dict[str, Any]],
    ) -> None:
        """Append vectors and exact organizer coordinates in aligned order."""
        for vector, record in zip(vectors, records, strict=True):
            position = len(self.embeddings_list)
            self.embeddings_list.append(vector[None, :])
            self.frame_mapping.append(
                {
                    "frame_id": record["frame_id"],
                    "video_id": record["video_id"],
                    "frame_idx": int(record["frame_idx"]),
                    "embedding_index": position,
                    "timestamp_ms": int(record["timestamp_ms"]),
                    "keyframe_order": int(record["keyframe_order"]),
                }
            )

    def _process_shard(
        self,
        records: list[dict[str, Any]],
        start: int,
        end: int,
        stats: EncodingStats,
    ) -> None:
        """Reuse or rebuild one fixed canonical interval without partial reuse."""
        expected_ids = [record["frame_id"] for record in records]
        shard_path = self._shard_path(start, end)
        cached = self._load_valid_shard(shard_path, expected_ids)
        if cached is not None and not self.strict:
            self._append_vectors(cached, records)
            return

        images, readable_records = self._load_images(records, stats)
        if len(readable_records) != len(records):
            if not self.strict:
                self._append_vectors(self._encode_images(images, stats), readable_records)
            return
        if cached is not None:
            self._append_vectors(cached, records)
            return

        vectors = self._encode_images(images, stats)
        self._write_shard(shard_path, expected_ids, vectors)
        self._append_vectors(vectors, records)

    def _metadata(
        self,
        total_frames: int,
        processing_time_sec: float,
    ) -> EmbeddingMetadata:
        """Build provenance for the generated corpus and its source table."""
        return EmbeddingMetadata(
            dataset_version=self.dataset_version,
            model_name=self.encoder_config.model_name,
            model_checkpoint=None,
            preprocessing_size=self.encoder_config.image_size,
            dtype=self.encoder_config.dtype,
            embedding_dimension=self.encoder.embedding_dim,
            total_frames=total_frames,
            successful_frames=len(self.frame_mapping),
            failed_frames=len(self.failed_frames),
            normalization="l2",
            generated_at=pd.Timestamp.now().isoformat(),
            device=self.encoder_config.device,
            batch_size=self.encoder_config.batch_size,
            processing_time_sec=processing_time_sec,
            model_revision=self.encoder_config.revision,
            source_fingerprint=fingerprint_files([self.frames_path]),
        )

    def _save(self, metadata: EmbeddingMetadata) -> None:
        """Atomically persist compact vectors, exact mapping, and metadata."""
        vectors = np.vstack(self.embeddings_list).astype(np.float32, copy=False)
        mapping = pd.DataFrame(self.frame_mapping)
        atomic_write(
            self.embeddings_file,
            lambda path: self._save_npy(path, vectors),
        )
        atomic_write(
            self.mapping_file,
            lambda path: write_parquet(mapping, path, index=False),
        )
        atomic_write(
            self.metadata_file,
            lambda path: write_yaml(metadata.to_dict(), path),
        )

    @staticmethod
    def _save_npy(path: Path, vectors: np.ndarray) -> None:
        """Write a NumPy array through an open handle for atomic temp names."""
        with path.open("wb") as handle:
            np.save(handle, vectors)

    def run(self) -> EmbeddingMetadata:
        """Build complete visual artifacts in canonical Parquet row order.

        Strict mode leaves prior compact artifacts untouched when a frame cannot
        be read or coverage is incomplete. Checkpoints remain so a repaired
        source can resume from already completed canonical slices.
        """
        timer = Timer()
        records = self._canonical_records(read_parquet(self.frames_path))
        stats = EncodingStats()
        for start in range(0, len(records), self.shard_size):
            end = min(start + self.shard_size, len(records))
            self._process_shard(records[start:end], start, end, stats)

        metadata = self._metadata(len(records), timer.stop() / 1000.0)
        complete = (
            not self.failed_frames
            and len(self.frame_mapping) == len(records)
            and len({row["frame_id"] for row in self.frame_mapping}) == len(records)
        )
        if self.strict and not complete:
            raise RuntimeError("Visual build does not have complete visual coverage")
        if self.frame_mapping:
            self._save(metadata)
        elif not self.strict:
            atomic_write(
                self.metadata_file,
                lambda path: write_yaml(metadata.to_dict(), path),
            )
        logger.info(
            "Embedding run: total=%d successful=%d failed=%d",
            metadata.total_frames,
            metadata.successful_frames,
            metadata.failed_frames,
        )
        return metadata
