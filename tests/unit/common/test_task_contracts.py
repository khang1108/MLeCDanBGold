from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from hcmai.common.config import EncoderConfig
from hcmai.common.schemas import (
    FrameEvidence,
    FrameRecord,
    QueryUnit,
    RetrievalSource,
    SceneCandidate,
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


def test_query_unit_validates_identity_text_order_and_round_trips() -> None:
    unit = QueryUnit(unit_id="H1", text="A red bus arrives.", order=0)

    assert QueryUnit.model_validate_json(unit.model_dump_json()) == unit
    for values in (
        {"unit_id": " ", "text": "event", "order": 0},
        {"unit_id": "H1", "text": " ", "order": 0},
        {"unit_id": "H1", "text": "event", "order": -1},
    ):
        with pytest.raises(ValidationError):
            QueryUnit.model_validate(values)


def test_frame_evidence_preserves_canonical_identity_and_provenance() -> None:
    evidence = FrameEvidence(
        frame=_frame(),
        unit_scores={"H1": 0.9},
        source_scores={RetrievalSource.VISUAL: 0.8},
        source_ranks={RetrievalSource.VISUAL: 2},
        score=0.9,
        provenance=("event", "visual"),
    )

    restored = FrameEvidence.model_validate_json(evidence.model_dump_json())
    assert restored == evidence
    assert restored.frame == _frame()
    assert restored.unit_scores == {"H1": 0.9}
    assert restored.source_scores == {RetrievalSource.VISUAL: 0.8}
    assert restored.source_ranks == {RetrievalSource.VISUAL: 2}
    assert restored.provenance == ("event", "visual")


def test_scene_candidate_validates_range_and_round_trips_evidence_scores() -> None:
    evidence = FrameEvidence(frame=_frame(), score=0.9)
    scene = SceneCandidate(
        scene_id="video-1:1000-5000",
        video_id="video-1",
        start_ms=1_000,
        end_ms=5_000,
        evidence=(evidence,),
        unit_scores={"H1": 0.9},
        semantic_score=0.8,
        coverage_score=0.7,
        temporal_score=0.6,
        relation_score=0.5,
        final_score=0.75,
        reason_labels=("retrieval_similarity",),
    )

    assert SceneCandidate.model_validate_json(scene.model_dump_json()) == scene
    assert scene.evidence == (evidence,)
    assert scene.reason_labels == ("retrieval_similarity",)
    with pytest.raises(ValidationError, match="end_ms"):
        SceneCandidate(
            scene_id="invalid",
            video_id="video-1",
            start_ms=5_000,
            end_ms=1_000,
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
