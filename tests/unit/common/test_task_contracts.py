from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from hcmai.common.config import EncoderConfig
from hcmai.common.schemas import (
    FrameRecord,
    RetrievalSource,
    SearchRequest,
    TextEmbeddingRequest,
    TRAKERequest,
    TRAKEResponse,
    TRAKESubmission,
    TaskRequest,
    TaskResponse,
    TaskType,
)


def _frame() -> FrameRecord:
    return FrameRecord(
        frame_id="frame-42",
        video_id="video-1",
        frame_idx=42,
        timestamp_ms=4_200,
        image_path="/frames/frame-42.jpg",
        width=1920,
        height=1080,
    )


def _trake_submission(**updates) -> TRAKESubmission:
    values = {
        "rank": 1,
        "video_id": "L01_V001",
        "frame_ids": ["frame-10", "frame-20"],
        "frame_idxs": [10, 20],
        "timestamps_ms": [1_000, 2_000],
    }
    values.update(updates)
    return TRAKESubmission.model_validate(values)


def test_search_requests_remain_compatible_without_search_id() -> None:
    assert SearchRequest.model_validate({"query": "red bus", "top_k": 10}).search_id is None


def test_text_embedding_contract_uses_shared_text_source_name() -> None:
    request = TextEmbeddingRequest(source="text", texts=["red bus"])

    assert request.source == "text"
    with pytest.raises(ValidationError):
        TextEmbeddingRequest(source="caption", texts=["red bus"])


def test_text_embedding_contract_defers_batch_ceiling_to_the_service() -> None:
    """Deployments may raise the model-specific API ceiling above 64 items."""

    request = TextEmbeddingRequest(texts=["red bus"] * 128)

    assert len(request.texts) == 128


@pytest.mark.parametrize("batch_size", [0, -1])
def test_encoder_config_rejects_nonpositive_batch_size(batch_size: int) -> None:
    """All encoder callers rely on a positive batch size as a range step."""

    with pytest.raises(ValidationError, match="greater than 0"):
        EncoderConfig(batch_size=batch_size)


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


def test_task_unions_discriminate_kis_and_trake_contracts() -> None:
    request_adapter = TypeAdapter(TaskRequest)
    response_adapter = TypeAdapter(TaskResponse)

    assert set(TaskType) == {TaskType.KIS, TaskType.TRAKE}
    assert (
        request_adapter.validate_python({"query": "person"}).query_type
        is TaskType.KIS
    )
    assert (
        request_adapter.validate_python(
            {
                "query_type": "trake",
                "query": "E1: walk\\nE2: sit",
                "events": ["walk", "sit"],
            }
        ).query_type
        is TaskType.TRAKE
    )

    trake_response = TRAKEResponse(
        request_id="trake-1",
        query="one -> two",
        events=["one", "two"],
        top_k=1,
        total_results=1,
        submissions=[_trake_submission()],
    )
    assert isinstance(
        response_adapter.validate_python(trake_response), TRAKEResponse
    )


def test_task_union_rejects_unknown_discriminator() -> None:
    for query_type in ("vqa", "vkis", "unknown"):
        with pytest.raises(ValidationError):
            TypeAdapter(TaskRequest).validate_python(
                {"query_type": query_type, "query": "test"}
            )
