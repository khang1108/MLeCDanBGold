"""Smoke tests for stateless KISC orchestration."""

from hcmai.agents.kisc import (
    ConversationResolver,
    KISCAgent,
)
from hcmai.common.schemas import (
    ConversationState,
    FrameFeedback,
    RetrievalCandidate,
)
from hcmai.common.schemas.kisc import KISCSearchRequest
from hcmai.search import SearchEngine


def test_resolver_failure_preserves_state_and_applies_newest_feedback() -> None:
    def fail(_):
        raise RuntimeError("provider unavailable")

    class Retriever:
        def search(self, query, top_k=20, filters=None):
            return [
                RetrievalCandidate(
                    frame_id=f"f{index}", final_score=1.0 - index * 0.1,
                    metadata={"frame": {
                        "video_id": "v1", "frame_idx": index,
                        "timestamp_ms": index * 100,
                    }},
                )
                for index in (1, 2)
            ]

    agent = KISCAgent(
        ConversationResolver(fail),
        SearchEngine(object(), Retriever()),
    )
    previous = ConversationState(
        standalone_query="người áo đỏ",
        positive_constraints=["người"],
        negative_constraints=["xe"],
        uncertain_constraints=["mũ"],
        accepted_frame_ids=["f1"],
    )
    response = agent.search(
        KISCSearchRequest(
            current_message="đang chạy",
            previous_state=previous,
            feedback=FrameFeedback(
                rejected_frame_ids=["f1"]
            ),
            top_k=2,
        )
    )
    state = response.interpreted_state
    assert state.standalone_query == "người áo đỏ đang chạy"
    assert state.positive_constraints == ["người"]
    assert state.negative_constraints == ["xe"]
    assert state.uncertain_constraints == ["mũ"]
    assert state.accepted_frame_ids == []
    assert state.rejected_frame_ids == ["f1"]
    assert response.search.total_results == 1
    assert response.warnings[0].startswith("Conversation fallback:")
