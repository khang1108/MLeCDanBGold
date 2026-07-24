"""Generate canonical visual embedding artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hcmai.common.config import EncoderConfig
from hcmai.common.utils.image import load_image
from hcmai.common.utils.io import read_parquet, write_parquet, write_yaml
from hcmai.common.utils.logging import get_logger
from hcmai.common.utils.timing import Timer
from hcmai.embedding.models import EmbeddingMetadata
from hcmai.retriever.encoder import DenseEncoder, EncodingStats

logger = get_logger(__name__)


class EmbeddingPipeline:
    """Encode canonical frame images and persist aligned artifacts."""

    def __init__(
        self,
        frames_path: Path | str,
        dataset_root: Path | str,
        output_dir: Path | str,
        encoder_config: EncoderConfig,
        dataset_version: str = "hcmai2026",
    ) -> None:
        """Configure paths and a lazy dense encoder."""
        self.frames_path = Path(frames_path)
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.output_dir = Path(output_dir)
        self.encoder_config = encoder_config
        self.dataset_version = dataset_version
        self.embeddings_dir = self.output_dir / "embeddings"
        self.embeddings_file = self.embeddings_dir / "visual_embeddings.npy"
        self.mapping_file = self.embeddings_dir / "frame_mapping.parquet"
        self.metadata_file = self.embeddings_dir / "metadata.yaml"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.encoder = DenseEncoder(encoder_config)
        self.embeddings_list: list[np.ndarray] = []
        self.frame_mapping: list[dict[str, Any]] = []
        self.failed_frames: list[dict[str, Any]] = []

    def _resolve_image(self, value: object) -> Path:
        """Resolve one canonical image path without changing the Parquet."""
        path = Path(str(value))
        resolved = path if path.is_absolute() else self.dataset_root / path
        return resolved.resolve()

    def _append_batch(
        self,
        images: list[Any],
        records: list[dict[str, Any]],
        stats: EncodingStats,
    ) -> None:
        """Encode a batch and append position-aligned mapping rows."""
        embeddings = self.encoder.encode_images(images, stats)
        for embedding, record in zip(embeddings, records):
            position = len(self.embeddings_list)
            self.embeddings_list.append(embedding[None, :])
            self.frame_mapping.append(
                {
                    "frame_id": record["frame_id"],
                    "video_id": record["video_id"],
                    "frame_idx": int(record["frame_idx"]),
                    "embedding_index": position,
                    "timestamp_ms": int(record["timestamp_ms"]),
                }
            )

    def _process_records(
        self,
        records: list[dict[str, Any]],
        stats: EncodingStats,
    ) -> None:
        """Load valid images and encode them in configured batches."""
        images: list[Any] = []
        batch: list[dict[str, Any]] = []
        for record in records:
            try:
                images.append(load_image(self._resolve_image(record["image_path"])))
                batch.append(record)
            except (OSError, ValueError) as error:
                stats.num_failed += 1
                self.failed_frames.append(
                    {"frame_id": record["frame_id"], "error": str(error)}
                )
                continue
            if len(images) == self.encoder_config.batch_size:
                self._append_batch(images, batch, stats)
                images, batch = [], []
        if images:
            self._append_batch(images, batch, stats)

    def _metadata(
        self,
        total_frames: int,
        processing_time_sec: float,
    ) -> EmbeddingMetadata:
        """Build provenance for the generated corpus."""
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
        )

    def _save(self, metadata: EmbeddingMetadata) -> None:
        """Persist embeddings, their exact frame mapping, and provenance."""
        if self.embeddings_list:
            np.save(self.embeddings_file, np.vstack(self.embeddings_list))
            write_parquet(pd.DataFrame(self.frame_mapping), self.mapping_file)
        write_yaml(metadata.to_dict(), self.metadata_file)

    def run(self) -> EmbeddingMetadata:
        """Generate embedding artifacts for every readable canonical frame."""
        timer = Timer()
        table = read_parquet(self.frames_path)
        records = table.to_dict(orient="records")
        stats = EncodingStats()
        self._process_records(records, stats)
        metadata = self._metadata(len(records), timer.stop() / 1000.0)
        self._save(metadata)
        logger.info(
            "Embedding run: total=%d successful=%d failed=%d",
            metadata.total_frames,
            metadata.successful_frames,
            metadata.failed_frames,
        )
        return metadata
