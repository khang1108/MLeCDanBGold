from __future__ import annotations
import asyncio
from typing import cast
import httpx
import pytest
from hcmai.app import create_app
from hcmai.common.schemas import (
    FrameRecord, RetrievalCandidate, RetrievalSource,
)
from hcmai.orchestration.pipeline import SearchService
from hcmai.data.pipeline import DataService
from hcmai.retrieval.retriever.pipeline import RetrievalService

pytestmark = pytest.mark.usefixtures("inline_router_threadpool")

FRAME_ID = "TEST_V001_keyframe_000001"

class Store:
    def __init__(self):
        self._records = (FrameRecord(
            frame_id=FRAME_ID, video_id="TEST_V001", frame_idx=0,
            timestamp_ms=0, image_path="missing.jpg", width=640, height=360,
        ),)
    def get(self, frame_id):
        if frame_id == FRAME_ID:
            return self._records[0]
        raise KeyError(frame_id)

    get_frame = get

    @property
    def record_count(self):
        return len(self._records)

    def has_evidence(self, source):
        return False

    def get_evidence(self, frame_id, source):
        return None

class Retriever:
    last_query_encoding_ms = 0.0
    last_index_search_ms = 0.0

    def search(self, query, top_k=20, filters=None, query_type=None):
        return [RetrievalCandidate(
            frame_id=FRAME_ID,
            source_scores={RetrievalSource.VISUAL: 0.9}, final_score=0.9,
            metadata={"frame": {
                "frame_id": FRAME_ID, "video_id": "TEST_V001",
                "frame_idx": 0, "timestamp_ms": 0, "caption": query,
            }},
        )]

def request(app, method, path, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(send())
    finally:
        loop.close()

def test_app_exposes_only_standalone_search_contract() -> None:
    store, retriever = Store(), Retriever()
    service = SearchService(
        cast(DataService, store), cast(RetrievalService, retriever)
    )
    app = create_app(service)
    health = request(app, "GET", "/health").json()
    capabilities = health["capabilities"]
    assert capabilities["search"] is True
    assert capabilities["kis"] is True
    assert capabilities["trake"] is True
    assert capabilities["shared_retrieval"] is True
    assert "query_suggestions" not in capabilities
    assert capabilities["frame_assets"] is False
    assert capabilities["query_types"] == {
        "kis": True,
        "trake": True,
    }
    assert "vqa" not in capabilities
    assert "vkis" not in capabilities
    assert capabilities["remote_inference"] == {
        "embedding": False,
        "reranking": False,
        "structured_parsing": False,
    }
    search = request(app, "POST", "/api/v1/search", json={"query": "red car"})
    assert search.status_code == 200
    result = search.json()["results"][0]
    assert result["frame_url"] == f"/api/v1/frames/{FRAME_ID}/image"
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/api/v1/vqa" not in paths
