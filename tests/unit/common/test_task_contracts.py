from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from hcmai.common.schemas import (
    ExecutionProfile,
    QueryLanguage,
    SearchRequest,
    TextEmbeddingRequest,
    TRAKERequest,
    TRAKEResponse,
    TRAKESubmission,
    TaskRequest,
    TaskResponse,
    TaskType,
    VQAInferenceEvidence,
    VQAInferenceRequest,
    VQAInferenceResponse,
    VQARequest,
    VQAResponse,
    VQASubmission,
)


def _vqa_submission(**updates) -> VQASubmission:
    values = {
        "rank": 1,
        "video_id": "L01_V001",
        "frame_id": "frame-42",
        "frame_idx": 42,
        "answer": "red",
        "retrieval_score": 0.8,
        "grounding_score": 0.9,
        "answer_score": 0.7,
        "joint_score": 0.75,
    }
    values.update(updates)
    return VQASubmission.model_validate(values)


def _trake_submission(**updates) -> TRAKESubmission:
    values = {
        "rank": 1,
        "video_id": "L01_V001",
        "frame_ids": ["frame-10", "frame-20"],
        "frame_idxs": [10, 20],
    }
    values.update(updates)
    return TRAKESubmission.model_validate(values)


def test_vqa_contracts_round_trip_without_losing_submission_text() -> None:
    request = VQARequest(
        event_description="A cook adds butter to a pan.",
        question="What is added?",
        top_k=100,
        language_hint=QueryLanguage.ENGLISH,
        execution_profile=ExecutionProfile.BALANCED,
    )
    response = VQAResponse(
        request_id="vqa-1",
        event_description=request.event_description,
        question=request.question,
        top_k=request.top_k,
        total_results=1,
        submissions=[_vqa_submission(answer="Bơ")],
    )

    assert VQARequest.model_validate_json(request.model_dump_json()) == request
    restored = VQAResponse.model_validate_json(response.model_dump_json())
    assert restored == response
    assert restored.submissions[0].answer == "Bơ"


def test_one_frame_vqa_contract_is_explicitly_provider_scoped() -> None:
    request = VQAInferenceRequest(frame_id="frame-1", question="Màu gì?")
    response = VQAInferenceResponse(
        request_id="inference-1",
        frame_id=request.frame_id,
        question=request.question,
        answer="đỏ",
        grounded=True,
        latency_ms=2,
        evidence=VQAInferenceEvidence(caption="Một ô vuông đỏ."),
    )

    assert response.evidence.caption == "Một ô vuông đỏ."


def test_text_embedding_contract_uses_shared_text_source_name() -> None:
    request = TextEmbeddingRequest(source="text", texts=["red bus"])

    assert request.source == "text"
    with pytest.raises(ValidationError):
        TextEmbeddingRequest(source="caption", texts=["red bus"])


@pytest.mark.parametrize(
    "payload",
    [
        {"event_description": " ", "question": "What?"},
        {"event_description": "event", "question": " "},
        {"event_description": "event", "question": "What?", "top_k": 101},
    ],
)
def test_vqa_request_rejects_invalid_public_input(payload: dict) -> None:
    with pytest.raises(ValidationError):
        VQARequest.model_validate(payload)


def test_vqa_submission_and_response_enforce_official_bounds() -> None:
    with pytest.raises(ValidationError):
        _vqa_submission(answer="x" * 101)
    with pytest.raises(ValidationError, match="total_results"):
        VQAResponse(
            request_id="vqa-1",
            event_description="event",
            question="question",
            top_k=1,
            total_results=0,
            submissions=[_vqa_submission()],
        )
    with pytest.raises(ValidationError, match="consecutive"):
        VQAResponse(
            request_id="vqa-1",
            event_description="event",
            question="question",
            top_k=2,
            total_results=1,
            submissions=[_vqa_submission(rank=2)],
        )


def test_trake_contracts_round_trip_with_canonical_frame_mapping() -> None:
    request = TRAKERequest(
        query="enter kitchen -> add butter",
        events=["enter kitchen", "add butter"],
        top_k=100,
    )
    response = TRAKEResponse(
        request_id="trake-1",
        query=request.query,
        events=request.events or [],
        top_k=request.top_k,
        total_results=1,
        submissions=[_trake_submission()],
    )

    restored = TRAKEResponse.model_validate_json(response.model_dump_json())
    assert restored == response
    assert restored.submissions[0].frame_idxs == [10, 20]


def test_trake_rejects_invalid_event_and_frame_sequences() -> None:
    with pytest.raises(ValidationError):
        TRAKERequest(query="one event", events=["one"])
    with pytest.raises(ValidationError):
        TRAKERequest(query="events", events=["first", " "])
    with pytest.raises(ValidationError):
        TRAKERequest(query="events", top_k=101)
    with pytest.raises(ValidationError, match="equal lengths"):
        _trake_submission(frame_ids=["frame-10", "frame-20", "frame-30"])
    with pytest.raises(ValidationError, match="preserve event order"):
        _trake_submission(frame_idxs=[20, 10])
    with pytest.raises(ValidationError, match="one frame per event"):
        TRAKEResponse(
            request_id="trake-1",
            query="three events",
            events=["one", "two", "three"],
            top_k=1,
            total_results=1,
            submissions=[_trake_submission()],
        )


def test_progressive_search_id_is_optional_and_must_be_a_uuid() -> None:
    search_id = "12345678-1234-5678-1234-567812345678"
    request = SearchRequest(query="red bus", search_id=search_id)

    assert SearchRequest.model_validate_json(request.model_dump_json()) == request
    assert SearchRequest(query="red bus").search_id is None
    assert VQARequest(event_description="event", question="question").search_id is None
    with pytest.raises(ValidationError):
        SearchRequest.model_validate({"query": "red bus", "search_id": "not-a-uuid"})


def test_task_unions_discriminate_all_request_and_response_types() -> None:
    request_adapter = TypeAdapter(TaskRequest)
    response_adapter = TypeAdapter(TaskResponse)

    assert isinstance(
        request_adapter.validate_python({"query": "red bus"}), SearchRequest
    )
    assert isinstance(
        request_adapter.validate_python(
            {
                "query_type": "vqa",
                "event_description": "a bus stops",
                "question": "What color is it?",
            }
        ),
        VQARequest,
    )
    assert isinstance(
        request_adapter.validate_python(
            {"query_type": "trake", "query": "one -> two"}
        ),
        TRAKERequest,
    )

    vqa_response = VQAResponse(
        request_id="vqa-1",
        event_description="event",
        question="question",
        top_k=1,
        total_results=1,
        submissions=[_vqa_submission()],
    )
    trake_response = TRAKEResponse(
        request_id="trake-1",
        query="one -> two",
        events=["one", "two"],
        top_k=1,
        total_results=1,
        submissions=[_trake_submission()],
    )
    assert isinstance(response_adapter.validate_python(vqa_response), VQAResponse)
    assert isinstance(
        response_adapter.validate_python(trake_response), TRAKEResponse
    )


def test_task_union_rejects_unknown_discriminator() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(TaskRequest).validate_python(
            {"query_type": "unknown", "query": "test"}
        )
