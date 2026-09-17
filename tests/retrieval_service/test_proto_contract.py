"""Contract tests for the private retrieval protobuf surface."""

import grpc
import google.protobuf
from google.protobuf.descriptor import FieldDescriptor

from hcmai.retrieval_service.proto import retrieval_pb2
from hcmai.retrieval_service.proto import retrieval_pb2_grpc


def _version_tuple(version: str) -> tuple[int, ...]:
    """Return numeric components for comparing released dependency versions."""

    return tuple(int(component) for component in version.split(".")[:3])


def test_REQ_002_generated_bindings_match_declared_runtime_floors() -> None:
    assert _version_tuple(google.protobuf.__version__) >= (6, 31, 1)
    assert _version_tuple(grpc.__version__) >= (1, 76, 0)
    assert _version_tuple(grpc.__version__) >= _version_tuple(
        retrieval_pb2_grpc.GRPC_GENERATED_VERSION
    )


def test_REQ_009_proto_carries_all_canonical_identity_fields() -> None:
    path = retrieval_pb2.AlignedPath.DESCRIPTOR.fields_by_name
    candidate = retrieval_pb2.ImageCandidate.DESCRIPTOR.fields_by_name

    assert {
        name: (field.number, field.type, field.label)
        for name, field in path.items()
        if name in {"video_id", "frame_ids", "frame_idxs", "timestamps_ms"}
    } == {
        "video_id": (1, FieldDescriptor.TYPE_STRING, FieldDescriptor.LABEL_OPTIONAL),
        "frame_ids": (3, FieldDescriptor.TYPE_STRING, FieldDescriptor.LABEL_REPEATED),
        "frame_idxs": (4, FieldDescriptor.TYPE_INT64, FieldDescriptor.LABEL_REPEATED),
        "timestamps_ms": (5, FieldDescriptor.TYPE_INT64, FieldDescriptor.LABEL_REPEATED),
    }
    assert {
        name: (field.number, field.type, field.label)
        for name, field in candidate.items()
        if name in {"video_id", "frame_id", "frame_idx", "timestamp_ms"}
    } == {
        "video_id": (1, FieldDescriptor.TYPE_STRING, FieldDescriptor.LABEL_OPTIONAL),
        "frame_id": (2, FieldDescriptor.TYPE_STRING, FieldDescriptor.LABEL_OPTIONAL),
        "frame_idx": (3, FieldDescriptor.TYPE_INT64, FieldDescriptor.LABEL_OPTIONAL),
        "timestamp_ms": (4, FieldDescriptor.TYPE_INT64, FieldDescriptor.LABEL_OPTIONAL),
    }


def test_REQ_010_video_scores_preserves_versioned_binary_layout() -> None:
    fields = retrieval_pb2.VideoScores.DESCRIPTOR.fields_by_name

    assert {
        name: (field.number, field.type, field.label)
        for name, field in fields.items()
    } == {
        "encoding_version": (1, FieldDescriptor.TYPE_UINT32, FieldDescriptor.LABEL_OPTIONAL),
        "video_id": (2, FieldDescriptor.TYPE_STRING, FieldDescriptor.LABEL_OPTIONAL),
        "frame_ids": (3, FieldDescriptor.TYPE_STRING, FieldDescriptor.LABEL_REPEATED),
        "frame_idxs_i64_le": (4, FieldDescriptor.TYPE_BYTES, FieldDescriptor.LABEL_OPTIONAL),
        "timestamps_ms_i64_le": (5, FieldDescriptor.TYPE_BYTES, FieldDescriptor.LABEL_OPTIONAL),
        "scores_f32_le": (6, FieldDescriptor.TYPE_BYTES, FieldDescriptor.LABEL_OPTIONAL),
        "event_count": (7, FieldDescriptor.TYPE_UINT32, FieldDescriptor.LABEL_OPTIONAL),
        "frame_count": (8, FieldDescriptor.TYPE_UINT32, FieldDescriptor.LABEL_OPTIONAL),
    }


def test_REQ_013_service_exposes_only_approved_application_rpcs() -> None:
    service = retrieval_pb2.DESCRIPTOR.services_by_name["RetrievalService"]
    assert [method.name for method in service.methods] == [
        "GetCapabilities",
        "SearchPlan",
        "SearchEvents",
        "ScoreVideo",
        "SearchImage",
    ]
