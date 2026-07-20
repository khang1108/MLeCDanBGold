"""Dense image-text encoder using SigLIP2 or other vision-language models."""

from __future__ import annotations

import numpy as np
from PIL import Image

from hcmai.common.utils.logging import get_logger
from hcmai.common.utils.timing import Timer
from hcmai.retriever.config import EncoderConfig
from hcmai.retriever.stats import EncodingStats

logger = get_logger(__name__)


class DenseEncoder:
    """Encode images and text to dense vectors using SigLIP2 or similar model."""

    def __init__(self, config: EncoderConfig) -> None:
        """Initialize the encoder with configuration.

        Args:
            config: EncoderConfig with model name, device, batch size, etc.
        """
        self.config = config
        self.model = None
        self.processor = None
        self.embedding_dim = 0

        # Load the model and processor
        try:
            import torch
            from transformers import AutoProcessor, AutoModel

            logger.info(f"Loading model: {self.config.model_name}")
            self.processor = AutoProcessor.from_pretrained(self.config.model_name)
            self.model     = AutoModel.from_pretrained(self.config.model_name).to(self.config.device)
            self.model.eval()

            with torch.no_grad():
                dummy_image = Image.new("RGB", (224, 224))
                inputs = self.processor(images=[dummy_image], return_tensors="pt").to(self.config.device)
                outputs = self.model.get_image_features(**inputs)
                self.embedding_dim = outputs.shape[-1]

            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        except ImportError as e:
            raise ImportError(
                f"Required dependencies not installed: {e}. "
                "Install with: pip install transformers torch pillow"
            )

    def encode_images(self, images: list[Image.Image], stats: EncodingStats | None = None) -> np.ndarray:
        """Encode a batch of images to dense vectors.

        Args:
            images: List of PIL images to encode.
            stats: Optional EncodingStats to accumulate timing/stats.

        Returns:
            Array of shape (len(images), embedding_dim).
        """
        import torch

        embeddings_list = []
        total_time = 0.0
        
        # Process images in batches
        for i in range(0, len(images), self.config.batch_size):
            batch = images[i : i + self.config.batch_size]

            # Use Timer context manager to measure batch processing time
            with Timer() as batch_timer:
                with torch.no_grad():
                    inputs = self.processor(images=batch, return_tensors="pt").to(self.config.device)
                    outputs = self.model.get_image_features(**inputs)
                    embeddings = outputs.cpu().numpy()

                # Normalize embeddings to unit norm for IP similarity
                embeddings = embeddings / (
                    np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
                )
                embeddings_list.append(embeddings)

            batch_time = batch_timer.elapsed_ms
            total_time += batch_time

            if stats is not None:
                stats.batch_times_ms.append(batch_time)

        all_embeddings = np.vstack(embeddings_list).astype(self.config.dtype)

        if stats is not None:
            stats.num_encoded += len(images)
            stats.total_time_ms += total_time
            stats.embedding_dim = self.embedding_dim

        return all_embeddings

    def encode_text(self, texts: list[str], stats: EncodingStats | None = None) -> np.ndarray:
        """Encode a batch of text strings to dense vectors.

        Args:
            texts: List of text strings to encode.
            stats: Optional EncodingStats to accumulate timing/stats.

        Returns:
            Array of shape (len(texts), embedding_dim).
        """
        import torch

        embeddings_list = []
        total_time = 0.0

        # Process text in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            
            # Use Timer context manager to measure batch processing time
            with Timer() as batch_timer:
                with torch.no_grad():
                    inputs = self.processor(text=batch, return_tensors="pt").to(self.config.device)
                    outputs = self.model.get_text_features(**inputs)
                    embeddings = outputs.cpu().numpy()

                # Normalize embeddings to unit norm for IP similarity
                embeddings = embeddings / (
                    np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
                )
                embeddings_list.append(embeddings)

            batch_time = batch_timer.elapsed_ms
            total_time += batch_time

            if stats is not None:
                stats.batch_times_ms.append(batch_time)

        all_embeddings = np.vstack(embeddings_list).astype(self.config.dtype)

        if stats is not None:
            stats.num_encoded += len(texts)
            stats.total_time_ms += total_time
            stats.embedding_dim = self.embedding_dim

        return all_embeddings
