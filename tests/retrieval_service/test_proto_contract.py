"""Contract tests for the private retrieval protobuf surface."""

from hcmai.retrieval_service.proto import retrieval_pb2


def test_REQ_009_proto_carries_all_canonical_identity_fields() -> None:
    path = retrieval_pb2.AlignedPath.DESCRIPTOR.fields_by_name
    candidate = retrieval_pb2.ImageCandidate.DESCRIPTOR.fields_by_name

    assert {"video_id", "frame_ids", "frame_idxs", "timestamps_ms"} <= set(path)
    assert {"video_id", "frame_id", "frame_idx", "timestamp_ms"} <= set(candidate)


def test_REQ_013_service_exposes_only_the_approved_rpc_surface() -> None:
    service = retrieval_pb2.DESCRIPTOR.services_by_name["RetrievalService"]
    assert [method.name for method in service.methods] == [
        "GetCapabilities",
        "SearchPlan",
        "SearchEvents",
        "ScoreVideo",
        "SearchImage",
    ]
