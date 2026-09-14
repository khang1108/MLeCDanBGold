"""Provider-agnostic text embedding capability and OpenAI-compatible implementation.

This module owns text embedding through the EmbeddingClient protocol.
It guarantees that returned embedding vectors strictly preserve input order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from hcmai.inference.config import ModelEndpointConfig
from hcmai.inference.http import HttpTransport


@dataclass(frozen=True, slots=True)
class TextEmbeddingBatch:
    """A batch of text embeddings with model metadata and ordered vectors."""

    model: str
    vectors: tuple[tuple[float, ...], ...]


class EmbeddingClient(Protocol):
    """Capability protocol for dense text embedding."""

    def embed_text(self, texts: list[str]) -> TextEmbeddingBatch:
        """Embed a list of text strings into vectors preserving input order."""
        ...


class OpenAICompatibleEmbeddingClient:
    """OpenAI embedding API client."""

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
            ValueError: If returned embedding count does not match input texts length.
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

        # Sort by 'index' to guarantee that output vectors match the input text order
        sorted_items = sorted(raw_items, key=lambda item: item.get("index", 0))
        vectors = tuple(tuple(float(x) for x in item["embedding"]) for item in sorted_items)
        model_name = data.get("model", self._endpoint.model)

        return TextEmbeddingBatch(
            model=model_name,
            vectors=vectors,
        )
