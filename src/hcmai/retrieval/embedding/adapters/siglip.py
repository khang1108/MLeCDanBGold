"""Dense image-text encoder using SigLIP2-style models."""

from __future__ import annotations

from typing import Any

import numpy as np

from PIL import Image
from tqdm.auto import tqdm

from transformers import AutoModel, AutoProcessor
from hcmai.common.config import EncoderConfig
from hcmai.common.utils.logging import get_logger
from hcmai.common.utils.timing import Timer
from hcmai.retrieval.embedding.models.stats import EncodingStats

logger = get_logger(__name__)


class SigLIPAdapter:
    """Encode images and text into one normalized embedding space."""

    def __init__(self, config: EncoderConfig) -> None:
        """Store configuration; load the model on the first encode call."""
        self.config = config
        self.model: Any | None = None
        self.processor: Any | None = None
        self.embedding_dim = 0

    def _load_model(self) -> None:
        """Load the processor and model exactly once."""
        if self.model is not None and self.processor is not None:
            return

        logger.info(f"Loading model: {self.config.model_name}")

        revision = (
            {"revision": self.config.revision}
            if self.config.revision is not None
            else {}
        )
        processor = AutoProcessor.from_pretrained(
            self.config.model_name, **revision
        )
        
        model = AutoModel.from_pretrained(self.config.model_name, **revision)
        model = model.to(self.config.device)
        model.eval()

        self.processor = processor
        self.model = model
        self.embedding_dim = int(getattr(model.config, "projection_dim", 0))

        dimension = self.embedding_dim or "pending"
        logger.info(
            "Successfully loaded model=%s processor=%s embedding_dimension=%s",
            self.config.model_name,
            self.config.model_name,
            dimension,
        )

    def _encode(
        self,
        items: list[Any],
        input_name: str,
        feature_method: str,
        stats: EncodingStats | None,
    ) -> np.ndarray:
        """Encode one modality in configured batches.

        Args:
            items (list[Any): A list of items to encode.
            input_name (str): Type of input (images or text)
            feature_method (str): Type of feature method of inference stage of model.
            stats (EncodinggStats): Statistics of encoding.

        Returns:
            A corresponding vector embedding for the input.
        """
        if not items:
            return np.empty((0, self.embedding_dim), dtype=self.config.dtype)
        self._load_model()
        assert self.model is not None and self.processor is not None

        import torch

        embeddings_list: list[np.ndarray] = []
        batch_times: list[float] = []
        for start in tqdm(
            range(0, len(items), self.config.batch_size),
            desc=f"SigLIP2 encoding {input_name}",
            unit="batch",
            dynamic_ncols=True,
        ):
            batch = items[start : start + self.config.batch_size]
            with Timer() as timer:
                inputs = self._processor_inputs(batch, input_name)

                with torch.inference_mode():
                    outputs = getattr(self.model, feature_method)(**inputs)

                tensor = _extract_tensor(outputs)
                embeddings = tensor.detach().float().cpu().numpy()

                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

                embeddings_list.append(embeddings / np.maximum(norms, 1e-8))
            batch_times.append(timer.elapsed_ms)

        result = np.vstack(embeddings_list).astype(self.config.dtype)
        self.embedding_dim = int(result.shape[1])

        if stats is not None:
            stats.num_encoded += len(items)
            stats.total_time_ms += sum(batch_times)
            stats.batch_times_ms.extend(batch_times)
            stats.embedding_dim = self.embedding_dim
        return result

    def _processor_inputs(self, batch: list[Any], input_name: str) -> Any:
        """Prepare one modality with the checkpoint's required sequence shape."""
        assert self.model is not None and self.processor is not None
        options: dict[str, Any] = {
            input_name: batch,
            "return_tensors": "pt",
        }
        if input_name == "text":
            text_config = getattr(self.model.config, "text_config", None)
            options.update(
                padding="max_length",
                truncation=True,
                max_length=int(
                    getattr(text_config, "max_position_embeddings", 64)
                ),
            )
        return self.processor(**options).to(self.config.device)

    def encode_images(
        self,
        images: list[Image.Image],
        stats: EncodingStats | None = None,
    ) -> np.ndarray:
        """Encode PIL images as L2-normalized vectors."""
        return self._encode(
            images,
            "images",
            "get_image_features",
            stats,
        )

    def encode_text(
        self,
        texts: list[str],
        stats: EncodingStats | None = None,
    ) -> np.ndarray:
        """Encode text strings as L2-normalized vectors."""
        return self._encode(
            texts,
            "text",
            "get_text_features",
            stats,
        )


def _extract_tensor(outputs: Any) -> Any:
    """Extract the pooled tensor from a Transformers model output."""
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is not None:
        return pooled
    if hasattr(outputs, "shape"):
        return outputs
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return getattr(outputs, "last_hidden_state", outputs)
