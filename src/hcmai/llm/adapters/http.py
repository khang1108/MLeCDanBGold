"""HTTP adapter for the bounded remote inference contract."""

from __future__ import annotations

import io
import json
from time import monotonic, perf_counter
from typing import Any, Sequence

import httpx
from PIL import Image

from hcmai.common.config import InferenceConfig
from hcmai.common.schemas import (
    CaptionResponse,
    InferenceReadiness,
    RerankResponse,
    TextEmbeddingResponse,
    VQAInferenceEvidence,
    VQAInferenceResponse,
)
from hcmai.common.utils.logging import get_logger
from hcmai.llm.gateway import InferenceGateway, InferenceGatewayError
from hcmai.llm.resilience import FailureCategory

logger = get_logger(__name__)


class InferenceClient:
    """One bounded HTTP client with optional Cloudflare service credentials."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float | InferenceConfig = 10,
        client: httpx.Client | None = None,
        gateway: InferenceGateway | None = None,
    ) -> None:
        config = (
            timeout_seconds
            if isinstance(timeout_seconds, InferenceConfig)
            else _legacy_config(timeout_seconds)
        )
        self.gateway = gateway or InferenceGateway(
            base_url,
            config,
            client,
        )
        self.client = self.gateway.client

    def embed_text(
        self, texts: list[str], source: str = "visual"
    ) -> TextEmbeddingResponse:
        payload = self._post(
            "/v1/embeddings/text",
            json={"source": source, "texts": texts},
        )
        return _validated(TextEmbeddingResponse, payload)

    def readiness(self, deadline_at: float | None = None) -> InferenceReadiness:
        payload = self._request("GET", "/ready", deadline_at=deadline_at)
        return _validated(InferenceReadiness, payload)

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
        response = _validated(CaptionResponse, payload)
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
        response = _validated(RerankResponse, payload)
        if [item.item_id for item in response.items] != item_ids:
            raise InferenceClientError("reranker changed item identity or order")
        return [item.score for item in response.items]

    def answer_vqa(
        self,
        request_id: str,
        frame_id: str,
        video_id: str,
        question: str,
        image: Image.Image,
        evidence: VQAInferenceEvidence | None = None,
        *,
        scene_context: str = "",
    ) -> VQAInferenceResponse:
        context = evidence or VQAInferenceEvidence()
        payload = self._post(
            "/v1/vqa",
            data={
                "request_id": request_id,
                "frame_id": frame_id,
                "video_id": video_id,
                "scene_context": scene_context,
                "question": question,
                "evidence": context.model_dump_json(),
            },
            files=[("image", (f"{frame_id}.jpg", _jpeg(image), "image/jpeg"))],
        )
        response = _validated(VQAInferenceResponse, payload)
        if response.request_id != request_id or response.video_id != video_id:
            raise InferenceClientError("VQA provider changed request/video identity")
        if response.frame_ids != [frame_id] or response.selected_frame_id != frame_id:
            raise InferenceClientError("VQA provider changed request/frame identity")
        if response.question != question:
            raise InferenceClientError("VQA provider changed the question")
        return response

    def answer_vqa_multi(
        self,
        request_id: str,
        video_id: str,
        frame_ids: list[str],
        question: str,
        images: Sequence[Image.Image],
        evidence: VQAInferenceEvidence | None = None,
        *,
        scene_context: str = "",
    ) -> VQAInferenceResponse:
        if not frame_ids or len(frame_ids) != len(images):
            raise ValueError("frame_ids and images must be non-empty and aligned")
        context = evidence or VQAInferenceEvidence()
        payload = self._post(
            "/v1/vqa/multi",
            data={
                "request_id": request_id,
                "video_id": video_id,
                "frame_ids": json.dumps(frame_ids),
                "scene_context": scene_context,
                "question": question,
                "evidence": context.model_dump_json(),
            },
            files=[
                ("images", (f"{frame_id}.jpg", _jpeg(image), "image/jpeg"))
                for frame_id, image in zip(frame_ids, images)
            ],
        )
        response = _validated(VQAInferenceResponse, payload)
        if response.request_id != request_id or response.video_id != video_id:
            raise InferenceClientError("VQA provider changed request/video identity")
        if response.frame_ids != frame_ids:
            raise InferenceClientError("VQA provider changed request/frame identity")
        if response.question != question:
            raise InferenceClientError("VQA provider changed the question")
        return response

    def _post(self, path: str, **kwargs: Any) -> Any:
        return self._request("POST", path, **kwargs)

    def _request(
        self,
        method: str,
        path: str,
        *,
        deadline_at: float | None = None,
        **kwargs: Any,
    ) -> Any:
        started = perf_counter()
        logger.info("Remote inference request started path=%s", path)
        try:
            response = self.gateway.request(
                method,
                path,
                idempotent=True,
                deadline_at=deadline_at,
                **kwargs,
            )
            payload = response.json()
        except InferenceGatewayError as error:
            logger.warning(
                "Remote inference request failed path=%s elapsed_ms=%d "
                "category=%s attempts=%d circuit=%s",
                path, int((perf_counter() - started) * 1_000),
                error.category.value,
                error.attempt_count,
                self.gateway.circuit.state.value,
            )
            raise InferenceClientError(
                f"{path} failed ({error.category.value})",
                category=error.category,
                attempt_count=error.attempt_count,
                circuit_state=self.gateway.circuit.state.value,
            ) from error
        except ValueError as error:
            raise InferenceClientError(
                f"{path} returned invalid JSON",
                category=FailureCategory.INVALID_RESPONSE,
                attempt_count=1,
                circuit_state=self.gateway.circuit.state.value,
            ) from error
        logger.info(
            "Remote inference request completed path=%s status=%d elapsed_ms=%d",
            path,
            response.status_code,
            int((perf_counter() - started) * 1_000),
        )
        return payload

    def health(self) -> dict[str, Any]:
        return self.gateway.health()

    def close(self) -> None:
        self.gateway.close()


class InferenceClientError(RuntimeError):
    """Bounded remote inference or contract failure."""

    def __init__(
        self,
        message: str,
        *,
        category: FailureCategory = FailureCategory.INVALID_RESPONSE,
        attempt_count: int = 1,
        circuit_state: str = "closed",
    ) -> None:
        super().__init__(message)
        self.category = category
        self.attempt_count = attempt_count
        self.circuit_state = circuit_state


def _jpeg(image: Image.Image) -> bytes:
    value = image.copy()
    value.thumbnail((768, 768))
    output = io.BytesIO()
    value.save(output, format="JPEG", quality=85)
    value.close()
    return output.getvalue()


def _legacy_config(timeout_seconds: float) -> InferenceConfig:
    return InferenceConfig(
        timeout_seconds=timeout_seconds,
        connect_timeout_seconds=timeout_seconds,
        read_timeout_seconds=timeout_seconds,
        write_timeout_seconds=timeout_seconds,
        pool_timeout_seconds=timeout_seconds,
    )


def _validated(model: Any, payload: Any) -> Any:
    try:
        return model.model_validate(payload)
    except Exception as error:
        raise InferenceClientError(
            "remote inference response contract is invalid",
            category=FailureCategory.INVALID_RESPONSE,
        ) from error
