"""Embedding contracts shared by the service and its adapters."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image

from hcmai.common.config import EncoderConfig
from hcmai.retrieval.embedding.models.stats import EncodingStats


class TextEmbeddingAdapter(Protocol):
    """Encode text into the vector space declared by ``config``."""

    config: EncoderConfig
    embedding_dim: int

    def encode_text(
        self,
        texts: list[str],
        stats: EncodingStats | None = None,
    ) -> np.ndarray: ...


class ImageEmbeddingAdapter(Protocol):
    """Encode images into the vector space declared by ``config``."""

    config: EncoderConfig
    embedding_dim: int

    def encode_images(
        self,
        images: list[Image.Image],
        stats: EncodingStats | None = None,
    ) -> np.ndarray: ...
