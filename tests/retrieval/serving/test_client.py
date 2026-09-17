"""Tests for RetrievalHttpClient."""

import httpx
import pytest

from hcmai.retrieval.plan import KISRetrievalEvent, KISRetrievalPlan
from hcmai.retrieval.serving.client import RetrievalHttpClient
from hcmai.retrieval.serving.utils.config import RetrievalClientSettings
from hcmai.retrieval.serving.utils.errors import (
    RetrievalInvalidRequestError,
    RetrievalNotFoundError,
    RetrievalUnavailableError,
)
from tests.retrieval.serving.fakes import make_fake_corpus


def test_client_settings_and_base_url() -> None:
    settings = RetrievalClientSettings(target="127.0.0.1:8002")
    assert settings.base_url == "http://127.0.0.1:8002"

    settings_with_scheme = RetrievalClientSettings(target="http://localhost:9000/")
    assert settings_with_scheme.base_url == "http://localhost:9000"


def test_probe_handles_connection_failure() -> None:
    settings = RetrievalClientSettings(target="127.0.0.1:59999", health_timeout_seconds=0.1)
    client = RetrievalHttpClient(settings)
    status = client.probe()

    assert status.reachable is False
    assert status.ready is False
    assert status.scoring_revision is None


def test_probe_handles_ready_and_unready_responses() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/capabilities":
            return httpx.Response(
                200,
                json={
                    "ready": True,
                    "scoring_revision": "rev-test-1",
                    "active_modalities": ["visual", "bm25"],
                    "startup_messages": [],
                    "max_temporal_event_count": 5,
                    "image_max_upload_bytes": 10485760,
                    "image_max_pixels": 25000000,
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    settings = RetrievalClientSettings(target="testserver")
    client = RetrievalHttpClient(settings)
    client._client = httpx.Client(transport=transport, base_url="http://testserver")

    status = client.probe()
    assert status.reachable is True
    assert status.ready is True
    assert status.scoring_revision == "rev-test-1"
    assert status.active_modalities == ("visual", "bm25")


def test_search_plan_and_events_mocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/search_plan":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "paths": [
                            {
                                "video_id": "video-1",
                                "score": 0.95,
                                "frame_ids": ["v1_f1"],
                                "frame_idxs": [10],
                                "timestamps_ms": [1000],
                            }
                        ],
                        "retrieval_ms": 10.0,
                        "alignment_ms": 2.0,
                    },
                    "video_scores": [
                        {
                            "video_id": "video-1",
                            "frame_ids": ["v1_f1"],
                            "frame_idx": [10],
                            "timestamps_ms": [1000],
                            "scores": [[0.95]],
                        }
                    ],
                    "decoder_config": {
                        "lambda_gap": 0.1,
                        "event_power": 1.0,
                        "cluster_delta": 0.5,
                        "path_min_separation_ms": 1000,
                    },
                    "scoring_revision": "rev-mock",
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = RetrievalHttpClient(RetrievalClientSettings(target="testserver"))
    client._client = httpx.Client(transport=transport, base_url="http://testserver")

    plan = KISRetrievalPlan(
        events=(
            KISRetrievalEvent("E1", canonical_text="car", dense_text="car", bm25_text=None, image_refs=()),
        )
    )
    corpus = make_fake_corpus()
    artifact = client.search_plan(plan, corpus=corpus)
    assert artifact.scoring_revision == "rev-mock"
    assert len(artifact.result.paths) == 1
    assert artifact.result.paths[0].video_id == "video-1"
