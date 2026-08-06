"""Embedding contracts shared by the service and its adapters."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from PIL import Image

from hcmai.common.config import EncoderConfig
from hcmai.embedding.models.stats import EncodingStats


class TextEmbeddingAdapter(Protocol):
    """Encode text into the vector space declared by ``config``."""

    config: EncoderConfig
    embedding_dim: int

    def encode_text(
        self,
        texts: list[str],
        stats: EncodingStats | None = None,
    ) -> np.ndarray: ...


class HostedTextEmbeddingAdapter(TextEmbeddingAdapter, Protocol):
    """Text adapter that supports explicit model warm-up."""

    model: Any | None

    def _load_model(self) -> None: ...


class ImageEmbeddingAdapter(Protocol):
    """Encode images into the vector space declared by ``config``."""

    config: EncoderConfig
    embedding_dim: int

    def encode_images(
        self,
        images: list[Image.Image],
        stats: EncodingStats | None = None,
    ) -> np.ndarray: ...
