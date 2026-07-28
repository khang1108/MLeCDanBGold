"""Synchronous clients adapting remote inference to existing pipeline contracts."""

from __future__ import annotations

import io
import json
import os
from time import perf_counter
from typing import Any, Sequence

import httpx
import numpy as np
from PIL import Image

from hcmai.common.config import EncoderConfig
from hcmai.common.schemas import RerankResponse, TextEmbeddingResponse
from hcmai.common.utils.logging import get_logger
from hcmai.retriever.models import EncodingStats

logger = get_logger(__name__)


class InferenceClient:
    """One bounded HTTP client with optional Cloudflare service credentials."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 10,
        client: httpx.Client | None = None,
    ) -> None:
        headers = _access_headers()
        self.client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            headers=headers,
        )

    def embed_text(self, texts: list[str]) -> TextEmbeddingResponse:
        payload = self._post("/v1/embeddings/text", json={"texts": texts})
        return TextEmbeddingResponse.model_validate(payload)

    def rerank(self, query: str, images: Sequence[Image.Image]) -> list[float]:
        item_ids = [str(index) for index in range(len(images))]
        files = [
            ("images", (f"{item_id}.jpg", _jpeg(image), "image/jpeg"))
            for item_id, image in zip(item_ids, images)
        ]
        payload = self._post(
            "/v1/rerank",
            data={"query": query, "item_ids": json.dumps(item_ids)},
            files=files,
        )
        response = RerankResponse.model_validate(payload)
        if [item.item_id for item in response.items] != item_ids:
            raise InferenceClientError("reranker changed item identity or order")
        return [item.score for item in response.items]

    def resolve_conversation(self, request: dict[str, Any]) -> object:
        return self._post("/v1/conversation/resolve", json=request)

    def _post(self, path: str, **kwargs: Any) -> Any:
        started = perf_counter()
        logger.info("Remote inference request started path=%s", path)
        try:
            response = self.client.post(path, **kwargs)
            response.raise_for_status()
            payload = response.json()
        except Exception as error:
            detail = _response_detail(error)
            logger.warning(
                "Remote inference request failed path=%s elapsed_ms=%d "
                "error=%s detail=%s",
                path, int((perf_counter() - started) * 1_000),
                type(error).__name__, detail,
            )
            raise InferenceClientError(f"{path} failed: {detail}") from error
        logger.info(
            "Remote inference request completed path=%s status=%d elapsed_ms=%d",
            path,
            response.status_code,
            int((perf_counter() - started) * 1_000),
        )
        return payload


class RemoteDenseEncoder:
    """Use hosted text embeddings with one configured local fallback."""

    def __init__(
        self,
        client: InferenceClient,
        config: EncoderConfig,
        embedding_dim: int,
        fallback: Any | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.embedding_dim = embedding_dim
        self.fallback = fallback

    def encode_text(
        self, texts: list[str], stats: EncodingStats | None = None
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=self.config.dtype)
        started = perf_counter()
        try:
            response = self.client.embed_text(texts)
            vectors = self._validate(response, len(texts))
        except Exception as error:
            if self.fallback is None:
                raise
            logger.warning(
                "Remote embedding unavailable; using local fallback "
                "texts=%d error=%s detail=%s",
                len(texts), type(error).__name__, _response_detail(error),
            )
            vectors = self.fallback.encode_text(texts, stats)
            logger.info(
                "Local embedding fallback completed texts=%d dimension=%d",
                len(texts), int(vectors.shape[1]),
            )
            return vectors
        if stats is not None:
            elapsed = (perf_counter() - started) * 1_000
            stats.num_encoded += len(texts)
            stats.total_time_ms += elapsed
            stats.batch_times_ms.append(elapsed)
            stats.embedding_dim = self.embedding_dim
        return vectors

    def _validate(
        self, response: TextEmbeddingResponse, count: int
    ) -> np.ndarray:
        if response.model != self.config.model_name:
            raise InferenceClientError("remote embedding checkpoint mismatch")
        if response.dimension != self.embedding_dim or not response.normalized:
            raise InferenceClientError("remote embedding metadata mismatch")
        vectors = np.asarray(response.embeddings, dtype=self.config.dtype)
        if vectors.shape != (count, self.embedding_dim):
            raise InferenceClientError("remote embedding shape mismatch")
        if not np.all(np.isfinite(vectors)):
            raise InferenceClientError("remote embedding contains non-finite values")
        return vectors


class InferenceClientError(RuntimeError):
    """Bounded remote inference failure consumed by existing fallbacks."""


def _jpeg(image: Image.Image) -> bytes:
    value = image.copy()
    value.thumbnail((768, 768))
    output = io.BytesIO()
    value.save(output, format="JPEG", quality=85)
    value.close()
    return output.getvalue()


def _access_headers() -> dict[str, str]:
    client_id = os.getenv("HCMAI_CF_ACCESS_CLIENT_ID")
    secret = os.getenv("HCMAI_CF_ACCESS_CLIENT_SECRET")
    if not client_id or not secret:
        return {}
    return {
        "CF-Access-Client-Id": client_id,
        "CF-Access-Client-Secret": secret,
    }


def _response_detail(error: Exception) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        try:
            detail = error.response.json().get("detail")
        except Exception:
            detail = None
        if detail:
            return str(detail)[:160]
    return (str(error).strip() or type(error).__name__)[:160]
