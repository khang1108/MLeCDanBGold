"""Tests for bounded LLM feedback resolver."""

from unittest.mock import Mock

import pytest

from hcmai.api.contracts.feedback import RetrievalOverride
from hcmai.kis.feedback.models import (
    AnchorAction,
    ClarifyAction,
    EditIntentAction,
    FeedbackResolution,
    FeedbackResolveContext,
    RefineRetrievalAction,
    RejectCandidateAction,
    RepairEventAction,
    RestructureAction,
)
from hcmai.kis.feedback.resolver import (
    FeedbackResolver,
    FeedbackResolverError,
    validate_action_references,
)
from hcmai.kis.models import KISEvent, KISIntent, KISTemporalEdge


def _create_context(
    events: list[str] | None = None,
    selected_event_id: str | None = None,
    selected_frame_id: str | None = None,
    selected_result_id: str | None = None,
) -> FeedbackResolveContext:
    ev_texts = events or ["A woman talks to a man.", "The woman takes a plate."]
    kis_events = [
        KISEvent(id=f"E{i+1}", text=text) for i, text in enumerate(ev_texts)
    ]
    edges = [
        KISTemporalEdge(source=f"E{i}", target=f"E{i+1}")
        for i in range(1, len(kis_events))
    ]
    intent = KISIntent(
        revision=1,
        query_text=" ".join(ev_texts),
        events=kis_events,
        temporal_edges=edges,
    )
    return FeedbackResolveContext(
        original_query="original test query",
        intent=intent,
        retrieval_overrides={},
        selected_result_id=selected_result_id,
        selected_event_id=selected_event_id,
        selected_frame_id=selected_frame_id,
        anchors={},
        recent_turns=[],
        scope="all_videos",
    )


def test_resolve_edit_intent():
    """Edit extra detail targeting a specific event."""
    context = _create_context()
    llm = Mock()
    expected_action = EditIntentAction(
        event_ids=["E1"],
        replacement_texts={"E1": "A man in a yellow shirt talks to a woman."},
    )
    llm.generate_structured.return_value = FeedbackResolution(action=expected_action)

    resolver = FeedbackResolver(llm=llm)
    action = resolver.resolve(context, "Ở E1 người đàn ông mặc áo màu vàng")

    assert action == expected_action
    assert llm.generate_structured.call_count == 1


def test_resolve_restructure_split():
    """Split sequential cooking steps into new events."""
    context = _create_context(events=["Cooking soup."])
    llm = Mock()
    expected_action = RestructureAction(
        replaced_event_ids=["E1"],
        new_events=["Chopping vegetables.", "Putting vegetables into boiling pot."],
        mapping={"E1": ["1", "2"]},
    )
    llm.generate_structured.return_value = FeedbackResolution(action=expected_action)

    resolver = FeedbackResolver(llm=llm)
    action = resolver.resolve(context, "Tách thành 2 bước: thái rau rồi cho vào nồi")

    assert action == expected_action
    assert llm.generate_structured.call_count == 1


def test_resolve_refine_retrieval():
    """Refine banner action retrieval text without mutating intent."""
    context = _create_context(events=["Two people hanging a blue banner."])
    llm = Mock()
    expected_action = RefineRetrievalAction(
        event_ids=["E1"],
        refinements={"E1": "Two people hanging a blue banner with festival text."},
    )
    llm.generate_structured.return_value = FeedbackResolution(action=expected_action)

    resolver = FeedbackResolver(llm=llm)
    action = resolver.resolve(context, "Tìm banner có chữ festival")

    assert action == expected_action
    assert llm.generate_structured.call_count == 1


def test_deictic_without_selection_returns_clarify():
    """Using 'đây' without selecting an event/frame must clarify rather than guessing."""
    context = _create_context(selected_event_id=None, selected_frame_id=None)
    llm = Mock()
    resolver = FeedbackResolver(llm=llm)

    # When message has deictic 'đây' and no selection in context, resolver must clarify
    action = resolver.resolve(context, "Ở đây người đó mặc áo vàng")
    assert isinstance(action, ClarifyAction)
    assert "chọn" in action.question.lower() or "event" in action.question.lower()
    # No LLM call needed if intercepted, or at most 1 call
    assert llm.generate_structured.call_count <= 1


def test_no_invented_event_ids():
    """Model returning non-existent event IDs must be rejected by validate_action_references."""
    context = _create_context()
    invalid_action = EditIntentAction(
        event_ids=["E99"],
        replacement_texts={"E99": "Ghost event."},
    )
    with pytest.raises(FeedbackResolverError):
        validate_action_references(invalid_action, context)
