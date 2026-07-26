from __future__ import annotations
import asyncio
import pytest
from fastapi import HTTPException
from hcmai.app import create_app
from hcmai.common.schemas import (
    FrameFeedback,
    FrameRecord,
    RetrievalCandidate,
    SearchRequest,
)
from hcmai.kisc import KiscSessionManager
from hcmai.search import SearchEngine

class FakeStore:
    def __init__(self) -> None:
        self._records = [
            FrameRecord(
                frame_id=f"frame-{index}",
                video_id="video-1",
                frame_idx=index,
                timestamp_ms=index * 40,
                image_path=f"k/{index}.jpg",
                width=8,
                height=6,
            )
            for index in (1, 2)
        ]

    def get(self, frame_id: str) -> FrameRecord:
        return next(row for row in self._records if row.frame_id == frame_id)

class FakeRetriever:
    def search(self, query: str, top_k: int, filters=None):
        values = (("frame-1", 0.9), ("frame-2", 0.8))
        return [
            RetrievalCandidate(frame_id=row, source_scores={"visual": score})
            for row, score in values
        ]

@pytest.fixture
def protocol() -> tuple[KiscSessionManager, SearchEngine]:
    manager = KiscSessionManager()
    engine = SearchEngine(frame_store=FakeStore(), retriever=FakeRetriever())
    return manager, engine

def test_session_feedback_and_submission(protocol) -> None:
    manager, engine = protocol
    assert manager.list_session_ids() == []
    session = manager.create_session(problem_id="problem-7")
    assert manager.list_session_ids() == [session.session_id]
    assert session.problem_id == "problem-7"
    with pytest.raises(KeyError, match="not found"):
        manager.get_session("missing")
    manager.update_feedback(
        session.session_id,
        FrameFeedback(rejected_frame_ids=["frame-1"]),
    )
    updated = manager.update_feedback(
        session.session_id,
        FrameFeedback(accepted_frame_ids=["frame-1"]),
    )
    assert updated.feedback.accepted_frame_ids == ["frame-1"]
    assert updated.feedback.rejected_frame_ids == []
    submission = manager.format_submission("frame-1", engine.frame_store)
    assert submission.submission_code == "video-1,1"

def test_feedback_ranking_and_turn_correlation(protocol) -> None:
    manager, engine = protocol
    session = manager.create_session()
    response = manager.process_search(
        SearchRequest(
            query="find it",
            session_id=session.session_id,
            feedback=FrameFeedback(accepted_frame_ids=["frame-2"]),
        ),
        engine,
    )
    assert [row.frame_id for row in response.results] == ["frame-2", "frame-1"]
    assert [row.rank for row in response.results] == [1, 2]
    assert response.turn_id == session.turns[0].turn_id
    assert response.assistant_turn_id == session.turns[1].turn_id
    assert session.turns[1].reply_to_turn_id == session.turns[0].turn_id
    response = manager.process_search(
        SearchRequest(
            query="not frame one",
            session_id=session.session_id,
            feedback=FrameFeedback(rejected_frame_ids=["frame-1"]),
        ),
        engine,
    )
    assert [row.frame_id for row in response.results] == ["frame-2"]

def test_unknown_search_session_returns_404(protocol) -> None:
    manager, engine = protocol
    app = create_app(search_engine=engine, session_manager=manager)
    route = next(
        route for route in app.routes
        if getattr(route, "path", None) == "/api/v1/search"
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            route.endpoint(SearchRequest(query="find it", session_id="missing"))
        )
    assert error.value.status_code == 404
