from __future__ import annotations
import asyncio
import httpx
from hcmai.agents.kisc import ConversationResolver, KISCAgent
from hcmai.app import create_app
from hcmai.common.schemas import (
    FrameRecord, RetrievalCandidate, RetrievalSource,
)
from hcmai.orchestration import SearchEngine

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

class Retriever:
    def search(self, query, top_k=20, filters=None, query_type=None):
        return [RetrievalCandidate(
            frame_id=FRAME_ID,
            source_scores={RetrievalSource.VISUAL: 0.9}, final_score=0.9,
            metadata={"frame": {
                "frame_id": FRAME_ID, "video_id": "TEST_V001",
                "frame_idx": 0, "timestamp_ms": 0, "caption": query,
            }},
        )]

def conversation(request):
    feedback = request.get("feedback") or {}
    return {
        "standalone_query": request["current_message"],
        "positive_constraints": [], "negative_constraints": [],
        "uncertain_constraints": [],
        "accepted_frame_ids": feedback.get("accepted_frame_ids", []),
        "rejected_frame_ids": feedback.get("rejected_frame_ids", []),
    }

def request(app, method, path, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(send())

def test_injected_provider_exposes_search_and_kisc_contracts() -> None:
    store, retriever = Store(), Retriever()
    engine = SearchEngine(store, retriever)
    agent = KISCAgent(ConversationResolver(conversation), engine)
    app = create_app(engine, kisc_agent=agent)
    health = request(app, "GET", "/health").json()
    assert health["capabilities"] == {
        "search": True,
        "kisc": True,
        "frame_assets": True,
        "query_types": {
            "kis": True, "vkis": True, "vqa": False, "trake": False,
        },
    }
    search = request(app, "POST", "/api/v1/search", json={"query": "red car"})
    assert search.status_code == 200
    result = search.json()["results"][0]
    assert result["frame_url"] == f"/api/v1/frames/{FRAME_ID}/image"
    kisc = request(
        app, "POST", "/api/v1/kisc/search",
        json={"current_message": "red car", "top_k": 1},
    )
    assert kisc.status_code == 200
    assert kisc.json()["interpreted_state"]["standalone_query"] == "red car"

def test_missing_structured_provider_disables_kisc() -> None:
    app = create_app(SearchEngine(Store(), Retriever()))
    response = request(
        app, "POST", "/api/v1/kisc/search", json={"current_message": "test"}
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "KISC provider not initialized"


def test_kisc_endpoint_rejects_standalone_query_type() -> None:
    app = create_app(SearchEngine(Store(), Retriever()))
    response = request(
        app,
        "POST",
        "/api/v1/kisc/search",
        json={"query_type": "kis", "current_message": "test"},
    )
    assert response.status_code == 422
