"""Generate and manage versioned, resumable frame embeddings using dense encoder."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Optional

from hcmai.common.utils.logging import get_logger
from hcmai.common.utils.io import read_parquet, write_parquet, write_yaml
from hcmai.common.utils.timing import Timer
from hcmai.common.utils.image import load_image
from hcmai.common.config import EncoderConfig
from hcmai.embedding.models import EmbeddingMetadata
from hcmai.retriever.encoder import DenseEncoder, EncodingStats

logger = get_logger(__name__)


class EmbeddingPipeline:
    """Generate versioned, resumable frame embeddings from a corpus."""

    def __init__(
        self,
        frames_path: Path | str,
        output_dir: Path | str,
        encoder_config: EncoderConfig,
        dataset_version: str = "hcmai2026",
        checkpoint_file: Optional[Path | str] = None,
    ) -> None:
        """Initialize the embedding pipeline.

        Args:
            frames_path: Path to frames.parquet containing FrameRecord objects.
            output_dir: Directory where embeddings and metadata will be saved.
            encoder_config: Configuration for the DenseEncoder.
            dataset_version: Version identifier for the dataset.
            checkpoint_file: Optional path to resume from a checkpoint.
        """
        self.frames_path = Path(frames_path)
        self.output_dir = Path(output_dir)
        self.encoder_config = encoder_config
        self.dataset_version = dataset_version
        self.checkpoint_file = Path(checkpoint_file) if checkpoint_file else None

        self.embeddings_dir = self.output_dir / "embeddings"
        self.embeddings_file = self.embeddings_dir / "visual_embeddings.npy"
        self.mapping_file = self.embeddings_dir / "frame_mapping.parquet"
        self.metadata_file = self.embeddings_dir / "metadata.yaml"

        # Ensure directories exist
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        self.encoder = DenseEncoder(encoder_config)
        self.processed_frame_ids: set[str] = set()
        self.embeddings_list: list[np.ndarray] = []
        self.frame_mapping: list[dict[str, Any]] = []
        self.failed_frames: list[dict[str, Any]] = []

    def run(self) -> EmbeddingMetadata:
        """Execute the embedding pipeline end-to-end.

        Returns:
            EmbeddingMetadata with statistics about the run.
        """
        logger.info(f"Starting embedding pipeline")
        logger.info(f"Frames path: {self.frames_path}")
        logger.info(f"Output dir: {self.output_dir}")

        timer = Timer()

        # Load frames
        logger.info("Loading frames...")
        frames_df = read_parquet(self.frames_path)
        total_frames = len(frames_df)
        logger.info(f"Loaded {total_frames} frames")

        # Resume from checkpoint if available
        if self.checkpoint_file and self.checkpoint_file.exists():
            try:
                checkpoint = json.loads(self.checkpoint_file.read_text())
                self.processed_frame_ids = set(checkpoint.get("processed", []))
                logger.info(f"Loaded checkpoint with {len(self.processed_frame_ids)} frames")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

        # Process frames in batches
        remaining_frames = [
            (idx, row)
            for idx, row in frames_df.iterrows()
            if row["frame_id"] not in self.processed_frame_ids
        ]

        logger.info(f"Processing {len(remaining_frames)} new frames")

        # Collect images for batch encoding
        batch_images = []
        batch_frame_ids = []
        batch_data = []
        encoding_stats = EncodingStats()

        for idx, row in remaining_frames:
            frame_id = row["frame_id"]
            
            try:
                image_path = row["image_path"]
                image = load_image(image_path)
                batch_images.append(image)
                batch_frame_ids.append(frame_id)
                batch_data.append(row.to_dict())

                # Encode batch when full or at end
                if len(batch_images) == self.encoder_config.batch_size or (idx == remaining_frames[-1][0]):
                    embeddings = self.encoder.encode_images(batch_images, encoding_stats)
                    for i, (frame_id, row) in enumerate(zip(batch_frame_ids, batch_data)):
                        self.embeddings_list.append(embeddings[i : i + 1])
                        mapping_entry = {
                            "frame_id": frame_id,
                            "video_id": row.get("video_id", ""),
                            "frame_idx": row.get("frame_idx", 0),
                            "embedding_index": len(self.embeddings_list) - 1,
                            "timestamp_ms": row.get("timestamp_ms", 0),
                        }
                        self.frame_mapping.append(mapping_entry)
                        self.processed_frame_ids.add(frame_id)
                    batch_images = []
                    batch_frame_ids = []
                    batch_data = []

            except Exception as e:
                logger.error(f"Failed to process frame {frame_id}: {e}")
                self.failed_frames.append(
                    {
                        "frame_id": frame_id,
                        "error": str(e),
                        "image_path": row.get("image_path"),
                    }
                )

        timer.stop()
        processing_time_sec = timer.elapsed_ms / 1000.0

        # Save artifacts
        logger.info("Saving embeddings and artifacts...")
        if self.embeddings_list:
            all_embeddings = np.vstack(self.embeddings_list)
            np.save(self.embeddings_file, all_embeddings)
            logger.info(
                f"Saved {len(all_embeddings)} embeddings to {self.embeddings_file}"
            )
        else:
            logger.warning("No embeddings to save")

        if self.frame_mapping:
            mapping_df = pd.DataFrame(self.frame_mapping)
            write_parquet(mapping_df, self.mapping_file)
            logger.info(
                f"Saved {len(mapping_df)} mappings to {self.mapping_file}"
            )
        else:
            logger.warning("No mapping to save")

        # Create and save metadata
        metadata = EmbeddingMetadata(
            dataset_version=self.dataset_version,
            model_name=self.encoder_config.model_name,
            model_checkpoint=None,
            preprocessing_size=self.encoder_config.image_size,
            dtype=self.encoder_config.dtype,
            embedding_dimension=self.encoder.embedding_dim,
            total_frames=total_frames,
            successful_frames=len(self.processed_frame_ids) + len(self.frame_mapping),
            failed_frames=len(self.failed_frames),
            normalization="l2",
            generated_at=pd.Timestamp.now().isoformat(),
            device=self.encoder_config.device,
            batch_size=self.encoder_config.batch_size,
            processing_time_sec=processing_time_sec,
        )

        write_yaml(metadata.to_dict(), self.metadata_file)
        logger.info(f"Metadata saved to {self.metadata_file}")

        # Log statistics
        logger.info("EMBEDDING PIPELINE COMPLETED")
        logger.info(encoding_stats.report())
        logger.info(
            f"Total frames: {total_frames}, "
            f"Successful: {metadata.successful_frames}, "
            f"Failed: {metadata.failed_frames}"
        )
        logger.info(f"Processing time: {processing_time_sec:.2f}s")
        logger.info(f"Embeddings saved to: {self.embeddings_file}")
        logger.info(f"Mapping saved to: {self.mapping_file}")

        return metadata

