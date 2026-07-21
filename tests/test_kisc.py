"""Smoke unit tests for KISC session manager and API endpoints."""

from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
from hcmai.app import create_app
from hcmai.common.schemas import FrameRecord, RetrievalCandidate
from hcmai.kisc import KiscSessionManager
from hcmai.search import SearchEngine


class MockFrameStore:
    def __init__(self) -> None:
        self.r1 = FrameRecord(
            frame_id="L21_V001_00000090", video_id="L21_V001", frame_idx=90,
            timestamp_ms=3600, image_path="k/090.jpg", width=1920, height=1080
        )
        self.r2 = FrameRecord(
            frame_id="L21_V001_00000091", video_id="L21_V001", frame_idx=91,
            timestamp_ms=3640, image_path="k/091.jpg", width=1920, height=1080
        )
        self._records = [self.r1, self.r2]

    def get(self, frame_id: str) -> FrameRecord:
        if frame_id == "L21_V001_00000090":
            return self.r1
        if frame_id == "L21_V001_00000091":
            return self.r2
        raise KeyError(f"Frame {frame_id} not found")

    def get_neighbors(self, frame_id: str, window: int = 5, include_target: bool = True) -> list[FrameRecord]:
        return [self.r1, self.r2]


class MockRetriever:
    def search(self, query: str, top_k: int = 10, filters: None = None) -> list[RetrievalCandidate]:
        return [
            RetrievalCandidate(frame_id="L21_V001_00000090", source_scores={"visual": 0.9}),
            RetrievalCandidate(frame_id="L21_V001_00000091", source_scores={"visual": 0.8}),
        ]


@pytest.fixture
def api_client() -> TestClient:
    engine = SearchEngine(frame_store=MockFrameStore(), retriever=MockRetriever())
    app = create_app(search_engine=engine, session_manager=KiscSessionManager())
    with TestClient(app) as client:
        yield client


def test_kisc_session_creation_and_feedback(api_client: TestClient) -> None:
    resp = api_client.post("/api/v1/session")
    assert resp.status_code == 200
    sess_id = resp.json()["session_id"]

    fb_resp = api_client.post(
        "/api/v1/feedback",
        params={"session_id": sess_id},
        json={"accepted_frame_ids": ["f1"], "rejected_frame_ids": ["f2"]},
    )
    assert fb_resp.status_code == 200
    assert fb_resp.json()["feedback"]["accepted_frame_ids"] == ["f1"]


def test_kisc_search_turn_filters_rejected_frames(api_client: TestClient) -> None:
    req = {
        "query": "walking person",
        "session_id": "s123",
        "feedback": {"rejected_frame_ids": ["L21_V001_00000090"]},
    }
    resp = api_client.post("/api/v1/search", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_results"] == 1
    assert data["results"][0]["frame_id"] == "L21_V001_00000091"


def test_neighbors_and_submit_endpoints(api_client: TestClient) -> None:
    n_resp = api_client.get("/api/v1/frames/L21_V001_00000090/neighbors?window=2")
    assert n_resp.status_code == 200
    assert len(n_resp.json()) == 2

    s_resp = api_client.post("/api/v1/submit", params={"frame_id": "L21_V001_00000090"})
    assert s_resp.status_code == 200
    assert s_resp.json()["submission_code"] == "L21_V001,90"
