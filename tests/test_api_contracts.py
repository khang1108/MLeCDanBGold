from __future__ import annotations
import asyncio
from typing import cast
import httpx
import numpy as np
import pytest
from hcmai.app import create_app
from hcmai.common.schemas import (
    FrameRecord, RetrievalSource,
)
from hcmai.orchestration.pipeline import SearchService
from hcmai.corpus import Corpus
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.retrieval.retriever.video_scores import VideoEventScores

pytestmark = pytest.mark.usefixtures("inline_router_threadpool")

FRAME_ID = "TEST_V001_keyframe_000001"

class Store:
    video_metadata_store = None

    def __init__(self):
        self._records = (FrameRecord(
            frame_id=FRAME_ID, video_id="TEST_V001", frame_idx=0,
            timestamp_ms=0, image_path="missing.jpg", width=640, height=360,
        ),)
    def get(self, frame_id):
        if frame_id == FRAME_ID:
            return self._records[0]
        raise KeyError(frame_id)

    frame = get

    def __len__(self):
        return len(self._records)

    def caption(self, frame_id):
        del frame_id
        return None
    ocr = caption
    def objects(self, frame_id):
        del frame_id
        return ()
    def title(self, video_id):
        del video_id
        return None
    def transcript(self, video_id, start_ms, end_ms):
        del video_id, start_ms, end_ms
        return None

class Retriever:
    def score_event_videos(self, events, filters=None, **kwargs):
        """Return one canonical alignment column for the API contract fixture."""

        del filters, kwargs
        return [VideoEventScores(
            video_id="TEST_V001",
            frame_ids=np.array([FRAME_ID], dtype=object),
            frame_idx=np.array([0]),
            timestamps_ms=np.array([0]),
            scores=np.full((len(events), 1), 0.9),
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
        cast(Corpus, store), cast(RetrievalService, retriever)
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
    assert "query_types" not in capabilities
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
