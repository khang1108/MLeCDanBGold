"""Smoke tests for stateless KISC orchestration."""

import pytest

from hcmai.agents.kisc import (
    ConversationResolver,
    ConversationResolverError,
    KISCAgent,
)
from hcmai.common.schemas import (
    ConversationState,
    FrameFeedback,
    RetrievalCandidate,
)
from hcmai.common.schemas.kisc import KISCSearchRequest
from hcmai.orchestration import SearchEngine


def test_resolver_failure_aborts_turn() -> None:
    def fail(_):
        raise RuntimeError("provider unavailable")

    class Retriever:
        def search(self, query, top_k=20, filters=None, query_type=None):
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
    with pytest.raises(ConversationResolverError, match="provider failed"):
        agent.search(KISCSearchRequest(
            current_message="đang chạy",
            previous_state=previous,
            feedback=FrameFeedback(
                rejected_frame_ids=["f1"]
            ),
            top_k=2,
        ))
