"""Provider-agnostic text embedding capability and OpenAI-compatible implementation.

This module owns text embedding through the EmbeddingClient.
It guarantees that returned embedding vectors strictly preserve input order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hcmai.inference.config import ModelEndpointConfig
from hcmai.inference.http import HttpTransport


@dataclass(frozen=True, slots=True)
class TextEmbeddingBatch:
    """A batch of text embeddings with model metadata and ordered vectors."""

    model: str
    vectors: tuple[tuple[float, ...], ...]


class EmbeddingClient:
    """HTTP client for dense text embeddings using remote endpoints."""

    def __init__(
        self,
        endpoint: ModelEndpointConfig,
        transport: HttpTransport | None = None,
    ) -> None:
        """Initialize the client with endpoint configuration and transport."""
        self._endpoint = endpoint
        self._transport = transport or HttpTransport()

    def embed_text(self, texts: list[str]) -> TextEmbeddingBatch:
        """Generate dense embeddings for a list of input texts.

        Args:
            texts: List of strings to embed.

        Returns:
            TextEmbeddingBatch containing the model name and vectors in the same
            order as input texts.

        Raises:
            ValueError: If returned count or indices do not match the input texts.
        """
        if not texts:
            return TextEmbeddingBatch(model=self._endpoint.model, vectors=())

        url = f"{self._endpoint.base_url}/embeddings"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._endpoint.api_key:
            headers["Authorization"] = f"Bearer {self._endpoint.api_key}"

        payload: dict[str, Any] = {
            "model": self._endpoint.model,
            "input": list(texts),
        }

        data = self._transport.post_json(
            url,
            payload,
            headers,
            self._endpoint.timeout_seconds,
        )

        raw_items = data.get("data", [])
        if len(raw_items) != len(texts):
            raise ValueError(
                f"Embedding count mismatch: expected {len(texts)} embeddings, got {len(raw_items)}"
            )

        indices = [item.get("index") for item in raw_items]
        # Reject ambiguous mappings rather than assigning a vector to the wrong text.
        if any(type(index) is not int for index in indices) or sorted(indices) != list(range(len(texts))):
            raise ValueError("Embedding indices must contain each input index exactly once")

        sorted_items = sorted(raw_items, key=lambda item: item["index"])
        vectors = tuple(tuple(float(x) for x in item["embedding"]) for item in sorted_items)
        model_name = data.get("model", self._endpoint.model)

        return TextEmbeddingBatch(
            model=model_name,
            vectors=vectors,
        )
