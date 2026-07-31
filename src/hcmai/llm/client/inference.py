"""Synchronous clients adapting remote inference to pipeline contracts."""

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
from hcmai.common.schemas import (
    CaptionResponse,
    InferenceReadiness,
    QuerySuggestionInferenceRequest,
    QuerySuggestionResponse,
    RerankResponse,
    TextEmbeddingResponse,
    VQAEvidence,
    VQAResponse,
)
from hcmai.common.utils.logging import get_logger
from hcmai.retriever.dense.models import EncodingStats

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

    def embed_text(
        self, texts: list[str], source: str = "visual"
    ) -> TextEmbeddingResponse:
        payload = self._post(
            "/v1/embeddings/text",
            json={"source": source, "texts": texts},
        )
        return TextEmbeddingResponse.model_validate(payload)

    def readiness(self) -> InferenceReadiness:
        return InferenceReadiness.model_validate(self._request("GET", "/ready"))

    def caption(self, images: Sequence[Image.Image]) -> CaptionResponse:
        item_ids = [str(index) for index in range(len(images))]
        files = [
            ("images", (f"{item_id}.jpg", _jpeg(image), "image/jpeg"))
            for item_id, image in zip(item_ids, images)
        ]
        payload = self._post(
            "/v1/captions",
            data={"item_ids": json.dumps(item_ids)},
            files=files,
        )
        response = CaptionResponse.model_validate(payload)
        if [item.item_id for item in response.items] != item_ids:
            raise InferenceClientError("captioner changed item identity or order")
        return response

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

    def suggest_queries(
        self,
        request: QuerySuggestionInferenceRequest,
        endpoint_path: str = "/v1/query-suggestions",
        timeout_seconds: float | None = None,
    ) -> QuerySuggestionResponse:
        payload = self._post(
            endpoint_path,
            json=request.model_dump(mode="json"),
            timeout=timeout_seconds,
        )
        response = QuerySuggestionResponse.model_validate(payload)
        if (
            response.request_id != request.request_id
            or response.original_query != request.query
        ):
            raise InferenceClientError(
                "query-suggestion provider changed request identity"
            )
        return response

    def answer_vqa(
        self,
        request_id: str,
        frame_id: str,
        question: str,
        image: Image.Image,
        evidence: VQAEvidence | None = None,
    ) -> VQAResponse:
        context = evidence or VQAEvidence()
        payload = self._post(
            "/v1/vqa",
            data={
                "request_id": request_id,
                "frame_id": frame_id,
                "question": question,
                "evidence": context.model_dump_json(),
            },
            files=[("image", (f"{frame_id}.jpg", _jpeg(image), "image/jpeg"))],
        )
        response = VQAResponse.model_validate(payload)
        if response.request_id != request_id or response.frame_id != frame_id:
            raise InferenceClientError("VQA provider changed request/frame identity")
        if response.question != question:
            raise InferenceClientError("VQA provider changed the question")
        return response

    def _post(self, path: str, **kwargs: Any) -> Any:
        return self._request("POST", path, **kwargs)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        started = perf_counter()
        logger.info("Remote inference request started path=%s", path)
        try:
            response = self.client.request(method, path, **kwargs)
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
    """Use hosted text embeddings and fail on remote contract violations."""

    def __init__(
        self,
        client: InferenceClient,
        config: EncoderConfig,
        embedding_dim: int,
        source: str = "visual",
    ) -> None:
        self.client = client
        self.config = config
        self.embedding_dim = embedding_dim
        self.source = source

    def encode_text(
        self, texts: list[str], stats: EncodingStats | None = None
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=self.config.dtype)
        started, batches = perf_counter(), []
        for start in range(0, len(texts), min(self.config.batch_size, 64)):
            batch = texts[start : start + min(self.config.batch_size, 64)]
            response = self.client.embed_text(batch, self.source)
            batches.append(self._validate(response, len(batch)))
        vectors = np.vstack(batches)
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
        if self.embedding_dim == 0:
            self.embedding_dim = response.dimension
        if response.dimension != self.embedding_dim or not response.normalized:
            raise InferenceClientError("remote embedding metadata mismatch")
        vectors = np.asarray(response.embeddings, dtype=self.config.dtype)
        if vectors.shape != (count, self.embedding_dim):
            raise InferenceClientError("remote embedding shape mismatch")
        if not np.all(np.isfinite(vectors)):
            raise InferenceClientError("remote embedding contains non-finite values")
        return vectors


class RemoteFrameCaptioner:
    """Adapt the hosted caption endpoint to the enrichment batch contract."""

    def __init__(self, client: InferenceClient, config: Any) -> None:
        self.client = client
        self.config = config
        self.resolved_revision: str | None = None

    def resolve_revision(self) -> str:
        status = self.client.readiness().models.get("caption_generation")
        if status is None or not status.loaded:
            raise InferenceClientError("remote caption model is not ready")
        if status.checkpoint != self.config.model_checkpoint:
            raise InferenceClientError("remote caption checkpoint mismatch")
        if not status.revision:
            raise InferenceClientError("remote caption revision is unresolved")
        self.resolved_revision = status.revision
        return status.revision

    def caption_batch(self, images: Sequence[Image.Image]) -> list[str]:
        response = self.client.caption(images)
        if response.model != self.config.model_checkpoint:
            raise InferenceClientError("remote caption checkpoint mismatch")
        if self.resolved_revision and response.revision != self.resolved_revision:
            raise InferenceClientError("remote caption revision changed")
        return [item.caption for item in response.items]


class InferenceClientError(RuntimeError):
    """Bounded remote inference or contract failure."""


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
