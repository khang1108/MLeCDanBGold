"""Tests for gRPC server and servicer validation/error mapping."""

from dataclasses import dataclass
import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
import pytest

from hcmai.orchestration.workflows.image_search import ImageQueryTooLargeError
from hcmai.retrieval_service.proto import retrieval_pb2, retrieval_pb2_grpc
from hcmai.retrieval_service.runtime import RetrievalRuntime
from hcmai.retrieval_service.server import create_server
from tests.retrieval_service.fakes import (
    make_fake_corpus,
    make_fake_image_scorer,
    make_fake_temporal_search_service,
)


@dataclass
class RunningServer:
    server: grpc.Server
    channel: grpc.Channel
    stub: retrieval_pb2_grpc.RetrievalServiceStub
    port: int


@pytest.fixture
def fake_runtime():
    corpus = make_fake_corpus()
    temporal = make_fake_temporal_search_service()
    image_scorer = make_fake_image_scorer()
    return RetrievalRuntime(
        corpus=corpus,
        temporal=temporal,
        image_scorer=image_scorer,
        image_search=None,
        active_modalities=("visual", "context", "bm25"),
        startup_messages=(),
        max_temporal_event_count=5,
        image_max_upload_bytes=10 * 1024 * 1024,
        image_max_pixels=4096 * 4096,
        scoring_revision="rev-server-test",
    )


@pytest.fixture
def grpc_server(fake_runtime):
    server, _, port = create_server(fake_runtime, host="127.0.0.1", port=0)
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = retrieval_pb2_grpc.RetrievalServiceStub(channel)
    try:
        yield RunningServer(server=server, channel=channel, stub=stub, port=port)
    finally:
        channel.close()
        server.stop(grace=None)


@pytest.fixture
def unready_grpc_server():
    server, _, port = create_server(None, host="127.0.0.1", port=0, startup_messages=["Corpus missing"])
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = retrieval_pb2_grpc.RetrievalServiceStub(channel)
    try:
        yield RunningServer(server=server, channel=channel, stub=stub, port=port)
    finally:
        channel.close()
        server.stop(grace=None)


def test_REQ_013_ready_runtime_reports_serving(grpc_server) -> None:
    health = health_pb2_grpc.HealthStub(grpc_server.channel)
    response = health.Check(
        health_pb2.HealthCheckRequest(service="hcmai.retrieval.v1.RetrievalService"),
        timeout=1,
    )
    assert response.status == health_pb2.HealthCheckResponse.SERVING

    overall = health.Check(health_pb2.HealthCheckRequest(service=""), timeout=1)
    assert overall.status == health_pb2.HealthCheckResponse.SERVING


def test_REQ_013_unready_runtime_reports_not_serving_and_unavailable(unready_grpc_server) -> None:
    health = health_pb2_grpc.HealthStub(unready_grpc_server.channel)
    response = health.Check(
        health_pb2.HealthCheckRequest(service="hcmai.retrieval.v1.RetrievalService"),
        timeout=1,
    )
    assert response.status == health_pb2.HealthCheckResponse.NOT_SERVING

    with pytest.raises(grpc.RpcError) as caught:
        unready_grpc_server.stub.SearchPlan(retrieval_pb2.SearchPlanRequest(), timeout=1)
    assert caught.value.code() is grpc.StatusCode.UNAVAILABLE


def test_REQ_011_invalid_plan_maps_to_invalid_argument(grpc_server) -> None:
    with pytest.raises(grpc.RpcError) as caught:
        grpc_server.stub.SearchPlan(retrieval_pb2.SearchPlanRequest(), timeout=1)
    assert caught.value.code() is grpc.StatusCode.INVALID_ARGUMENT


def test_REQ_011_missing_selected_video_maps_to_not_found(grpc_server) -> None:
    request = retrieval_pb2.ScoreVideoRequest(
        video_id="video-unknown",
        plan=retrieval_pb2.SearchPlanRequest(
            events=[retrieval_pb2.RetrievalEvent(event_id="E1", canonical_text="test", dense_text="test")],
            use_dense=True,
            top_k=20,
        ),

    )
    with pytest.raises(grpc.RpcError) as caught:
        grpc_server.stub.ScoreVideo(request, timeout=1)
    assert caught.value.code() is grpc.StatusCode.NOT_FOUND


def test_REQ_011_oversized_image_maps_to_resource_exhausted() -> None:
    corpus = make_fake_corpus()
    temporal = make_fake_temporal_search_service()
    
    def fake_search(*args, **kwargs):
        raise ImageQueryTooLargeError("image exceeds byte limit")

    image_search = type("MockIS", (), {"search": fake_search})()
    runtime = RetrievalRuntime(
        corpus=corpus,
        temporal=temporal,
        image_scorer=None,
        image_search=image_search,
        active_modalities=("visual",),
        startup_messages=(),
        max_temporal_event_count=5,
        image_max_upload_bytes=10,
        image_max_pixels=100,
        scoring_revision="rev-test",
    )
    server, _, port = create_server(runtime, host="127.0.0.1", port=0)
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = retrieval_pb2_grpc.RetrievalServiceStub(channel)
    try:
        request = retrieval_pb2.SearchImageRequest(payload=b"12345678901", content_type="image/jpeg", top_k=5)
        with pytest.raises(grpc.RpcError) as caught:
            stub.SearchImage(request, timeout=1)
        assert caught.value.code() is grpc.StatusCode.RESOURCE_EXHAUSTED
    finally:
        channel.close()
        server.stop(grace=None)


def test_REQ_011_unexpected_error_maps_to_internal_without_stack_trace(grpc_server, monkeypatch) -> None:
    def broken_search(*args, **kwargs):
        raise RuntimeError("database crash at secret_path/db.py:42")

    # Monkeypatch temporal service search_events
    # Find the runtime in the servicer
    servicer = grpc_server.server._servicer
    monkeypatch.setattr(servicer.runtime.temporal, "search", broken_search)

    request = retrieval_pb2.SearchEventsRequest(
        original_events=["event1"],
        use_dense=True,
        top_k=5,
    )
    with pytest.raises(grpc.RpcError) as caught:
        grpc_server.stub.SearchEvents(request, timeout=1)
    assert caught.value.code() is grpc.StatusCode.INTERNAL
    assert "secret_path" not in caught.value.details()
    assert "db.py" not in caught.value.details()

