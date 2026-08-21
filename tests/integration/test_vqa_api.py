"""Integration coverage for the thin competition VQA HTTP boundary."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from hcmai.app import create_app
from hcmai.common.schemas import (
    TaskType,
    VQARequest,
    VQAResponse,
    VQASubmission,
)

pytestmark = pytest.mark.usefixtures("inline_router_threadpool")


class FakeSearchService:
    """Record the validated task contract received from the HTTP router."""

    def __init__(self) -> None:
        self.request: VQARequest | None = None

    def search(self, request: VQARequest) -> VQAResponse:
        self.request = request
        return VQAResponse(
            request_id="vqa-request-1",
            search_id=request.search_id,
            event_description=request.event_description,
            question=request.question,
            top_k=request.top_k,
            total_results=1,
            submissions=[
                VQASubmission(
                    rank=1,
                    video_id="video-1",
                    frame_id="frame-1",
                    frame_ids=["frame-1"],
                    frame_idx=42,
                    answer="blue",
                    normalized_answer="blue",
                    retrieval_score=0.9,
                    grounding_score=0.8,
                    answer_score=0.9,
                    joint_score=0.87,
                )
            ],
            latency_ms=7,
        )


def request(app: Any, payload: dict[str, Any]) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.post("/api/v1/vqa", json=payload)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(send())
    finally:
        loop.close()


def test_vqa_route_validates_delegates_and_serializes_response() -> None:
    service = FakeSearchService()
    app = create_app(search_service=service)  # type: ignore[arg-type]

    response = request(
        app,
        {
            "event_description": "A person holds a colored umbrella",
            "question": "What color is the umbrella?",
            "top_k": 100,
            "search_id": "search-session-1",
        },
    )

    assert response.status_code == 200
    assert isinstance(service.request, VQARequest)
    assert service.request.query_type is TaskType.VQA
    assert service.request.top_k == 100
    assert response.json()["search_id"] == "search-session-1"
    submission = response.json()["submissions"][0]
    assert submission["video_id"] == "video-1"
    assert submission["frame_id"] == "frame-1"
    assert submission["frame_ids"] == ["frame-1"]
    assert submission["frame_idx"] == 42
    assert submission["answer"] == "blue"


def test_vqa_route_rejects_invalid_or_incomplete_input() -> None:
    app = create_app(search_service=FakeSearchService())  # type: ignore[arg-type]

    missing_question = request(
        app, {"event_description": "A person holds an umbrella"}
    )
    too_many = request(
        app,
        {
            "event_description": "A person holds an umbrella",
            "question": "What color?",
            "top_k": 101,
        },
    )

    assert missing_question.status_code == 422
    assert too_many.status_code == 422
