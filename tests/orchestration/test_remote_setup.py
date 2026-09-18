"""Tests for remote retrieval backend setup and non-blocking startup."""

import pytest
from hcmai.orchestration.setup import load_search_service
from hcmai.retrieval.serving.client import RemoteRetrievalStatus, RetrievalHttpClient


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
    from tests.retrieval.serving.fakes import make_fake_corpus
    fake_corpus = make_fake_corpus()
    monkeypatch.setattr(
        "hcmai.orchestration.setup.load_configured_corpus",
        lambda *args, **kwargs: fake_corpus,
    )
    monkeypatch.setattr(
        "hcmai.orchestration.setup.retrieval.load_retrieval",
        lambda *args, **kwargs: pytest.fail("FastAPI attempted local index loading"),
    )
    monkeypatch.setattr(RetrievalHttpClient, "probe", lambda self: _unavailable_status())

    messages = []
    service = load_search_service(messages)

    assert service.remote_retrieval is not None
    assert service.literal_text is not None


def test_REQ_003_fastapi_lifespan_with_unused_grpc_port(monkeypatch) -> None:
    from fastapi.testclient import TestClient
    from hcmai.app import create_app
    from tests.retrieval.serving.fakes import make_fake_corpus

    fake_corpus = make_fake_corpus()
    monkeypatch.setattr(
        "hcmai.orchestration.setup.load_configured_corpus",
        lambda *args, **kwargs: fake_corpus,
    )
    monkeypatch.setenv("HCMAI_RETRIEVAL_TARGET", "127.0.0.1:59999")
    monkeypatch.setenv("HCMAI_RETRIEVAL_TIMEOUT_SECONDS", "1.0")
    monkeypatch.setenv("HCMAI_RETRIEVAL_HEALTH_TIMEOUT_SECONDS", "0.5")

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


def test_health_report_with_active_bm25_and_all_modalities(monkeypatch) -> None:
    from tests.retrieval.serving.fakes import make_fake_corpus

    fake_corpus = make_fake_corpus()
    monkeypatch.setattr(
        "hcmai.orchestration.setup.load_configured_corpus",
        lambda *args, **kwargs: fake_corpus,
    )
    status = RemoteRetrievalStatus(
        target="127.0.0.1:8002",
        reachable=True,
        ready=True,
        scoring_revision=1,
        active_modalities=("visual", "context", "bm25", "asr"),
        startup_messages=(),
    )
    monkeypatch.setattr(RetrievalHttpClient, "probe", lambda self: status)

    messages = []
    service = load_search_service(messages)
    health = service.health(messages)

    assert health["status"] == "ok"
    assert health["capabilities"]["bm25"] is True
    assert health["capabilities"]["hybrid_temporal"] is True
    assert health["retrieval_modalities"]["visual"]["active"] is True
    assert health["retrieval_modalities"]["context"]["active"] is True
    assert health["retrieval_modalities"]["caption"]["active"] is True
    assert health["retrieval_modalities"]["ocr"]["active"] is True
    assert health["retrieval_modalities"]["asr"]["active"] is True

