"""Tests for remote retrieval backend setup and non-blocking startup."""

import pytest
from hcmai.orchestration.setup import load_search_service
from hcmai.retrieval_service.client import RemoteRetrievalStatus, RetrievalGrpcClient


def _unavailable_status(target: str = "127.0.0.1:8002") -> RemoteRetrievalStatus:
    return RemoteRetrievalStatus(
        target=target,
        reachable=False,
        ready=False,
        scoring_revision=None,
        active_modalities=(),
        startup_messages=("Service unavailable",),
    )


def test_REQ_001_backend_setup_never_loads_local_retrieval(monkeypatch) -> None:
    from tests.retrieval_service.fakes import make_fake_corpus
    fake_corpus = make_fake_corpus()
    monkeypatch.setattr(
        "hcmai.orchestration.setup.load_configured_corpus",
        lambda *args, **kwargs: fake_corpus,
    )
    monkeypatch.setattr(
        "hcmai.orchestration.retrieval_setup.load_retrieval",
        lambda *args, **kwargs: pytest.fail("FastAPI attempted local index loading"),
    )
    monkeypatch.setattr(RetrievalGrpcClient, "probe", lambda self: _unavailable_status())

    messages = []
    service = load_search_service(messages)

    assert service.remote_retrieval is not None
    assert service.literal_text is not None


def test_REQ_003_fastapi_lifespan_with_unused_grpc_port(monkeypatch) -> None:
    from fastapi.testclient import TestClient
    from hcmai.app import create_app
    from tests.retrieval_service.fakes import make_fake_corpus

    fake_corpus = make_fake_corpus()
    monkeypatch.setattr(
        "hcmai.orchestration.setup.load_configured_corpus",
        lambda *args, **kwargs: fake_corpus,
    )
    monkeypatch.setenv("HCMAI_RETRIEVAL_GRPC_TARGET", "127.0.0.1:59999")
    monkeypatch.setenv("HCMAI_RETRIEVAL_CLIENT_TIMEOUT", "1.0")
    monkeypatch.setenv("HCMAI_RETRIEVAL_CLIENT_HEALTH_TIMEOUT", "0.5")

    app = create_app()
    with TestClient(app) as client:
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["status"] == "ok"
        assert health_data["remote_retrieval"]["reachable"] is False

        filter_resp = client.post("/api/v1/filter", json={"page_id": 1, "frames_per_pages": 20})
        assert filter_resp.status_code == 200

        kis_resp = client.post(
            "/api/v1/kis/search",
            json={
                "operation": {"kind": "initial_resolve", "text": "a person walking"},
                "expected_revision": 0,
            },
        )
        assert kis_resp.status_code in (503, 502)
