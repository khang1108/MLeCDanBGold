"""Task 2 exploration seed transport and branch projection regressions."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import ValidationError

from hcmai.api.contracts.exploration import ExplorationOpenRequest
from hcmai.api.routers.exploration import _open_branch
from hcmai.orchestration.workflows.temporal_exploration import ExplorationView
from hcmai.temporal.constraints import Conditions


def seed_payload():
    return dict(semantic_revision=2, events=[dict(event_id="E1", canonical_text="người đi", dense_text="person walks", bm25_text="người đi")], use_dense=True, use_bm25=True)


def test_task2_step8_router_projects_seed_into_immutable_plan():
    request = ExplorationOpenRequest(seed=seed_payload(), video_id="video-1", window=(0, 100))
    temporal = Mock()
    with patch("hcmai.api.routers.exploration.TemporalExploration") as branch_type:
        _open_branch(temporal, request, "scoring-v1")
    binding, video_id, window = branch_type.return_value.open.call_args.args
    assert binding.semantic_revision == 2
    assert binding.retrieval_plan.event_ids == ("E1",)
    assert binding.retrieval_plan.dense_texts == ("person walks",)
    assert binding.retrieval_plan.bm25_texts == ("người đi",)
    assert (video_id, window) == ("video-1", (0, 100))


@pytest.mark.parametrize("mutation", [
    {"events": [dict(event_id="E2", canonical_text="text", dense_text="text")]},
    {"events": [dict(event_id="E1", canonical_text="text")]},
    {"use_dense": False, "use_bm25": False},
])
def test_task2_step8_open_rejects_invalid_seed(mutation):
    with pytest.raises(ValidationError):
        ExplorationOpenRequest(seed={**seed_payload(), **mutation}, video_id="video-1", window=(0, 100))


def test_task2_step8_open_rejects_legacy_fields_and_reversed_window():
    for extra in ({"query": "legacy"}, {"window": (100, 0)}):
        with pytest.raises(ValidationError):
            ExplorationOpenRequest.model_validate(dict(seed=seed_payload(), video_id="video-1", window=(0, 100)) | extra)


@pytest.mark.anyio
async def test_task2_step8_http_open_accepts_seed_without_legacy_fields():
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from hcmai.api.routers.exploration import create_exploration_router

    app = FastAPI()
    service = Mock()
    service.kis.temporal = object()
    app.include_router(create_exploration_router({"service": service}))
    view = ExplorationView(
        revision=1,
        event_version="events-v1",
        video_id="video-1",
        conditions=Conditions(window=(0, 100), confirmed=(None,), rejected=((),)),
        status="no_valid_path",
        paths=(),
        changed_event_indices=(),
        comparison_available=False,
        can_undo=False,
    )
    with patch(
        "hcmai.api.routers.exploration.run_in_threadpool",
        new=AsyncMock(return_value=(Mock(), view)),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/exploration", json={
                "seed": seed_payload(), "video_id": "video-1", "window": [0, 100],
            })
    assert response.status_code == 200
    assert response.json()["view"]["video_id"] == "video-1"
    assert response.json()["view"]["revision"] == 1


@pytest.mark.anyio
async def test_task9_http_open_accepts_image_only_seed():
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from hcmai.api.routers.exploration import create_exploration_router

    app = FastAPI()
    service = Mock()
    service.kis.temporal = object()
    service.kis.image_scorer = Mock()
    app.include_router(create_exploration_router({"service": service}))
    view = ExplorationView(
        revision=1,
        event_version="events-v1",
        video_id="video-1",
        conditions=Conditions(window=(0, 100), confirmed=(None,), rejected=((),)),
        status="ok",
        paths=(),
        changed_event_indices=(),
        comparison_available=False,
        can_undo=False,
    )
    with patch(
        "hcmai.api.routers.exploration.run_in_threadpool",
        new=AsyncMock(return_value=(Mock(), view)),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/exploration", json={
                "seed": {
                    "semantic_revision": 1,
                    "events": [
                        {
                            "event_id": "E1",
                            "image_refs": [{"asset_id": "sha256:img1", "content_type": "image/png"}],
                        }
                    ],
                    "use_dense": True,
                    "use_bm25": False,
                },
                "video_id": "video-1",
                "window": [0, 100],
            })
    assert response.status_code == 200
    assert response.json()["view"]["video_id"] == "video-1"


@pytest.mark.anyio
async def test_REQ_008_exploration_returns_score_video_revision():
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from hcmai.api.routers.exploration import create_exploration_router

    app = FastAPI()
    service = Mock()
    temporal = Mock()
    temporal.get_scoring_revision.return_value = "remote-v7"
    service.kis.temporal = temporal

    app.include_router(create_exploration_router({"service": service}))
    view = ExplorationView(
        revision=1,
        event_version="events-v1",
        video_id="video-1",
        conditions=Conditions(window=(0, 100), confirmed=(None,), rejected=((),)),
        status="ok",
        paths=(),
        changed_event_indices=(),
        comparison_available=False,
        can_undo=False,
    )
    branch = Mock()
    branch.scoring_revision = "remote-v7"
    with patch(
        "hcmai.api.routers.exploration.run_in_threadpool",
        new=AsyncMock(return_value=(branch, view)),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/exploration", json={
                "seed": seed_payload(),
                "video_id": "video-1",
                "window": [0, 100],
            })
    assert response.status_code == 200
    assert response.json()["scoring_revision"] == "remote-v7"


def test_REQ_008_exploration_opening_fails_when_capability_and_score_video_revisions_differ():
    import numpy as np
    from hcmai.api.contracts.exploration import ExplorationOpenRequest
    from hcmai.orchestration.workflows.temporal_exploration import ExplorationConflict
    from hcmai.orchestration.workflows.temporal_search import DecoderConfigSnapshot, SelectedVideoScoreResult
    from hcmai.retrieval.retriever.video_scores import VideoEventScores

    temporal = Mock()
    temporal.get_scoring_revision.return_value = "remote-v7"
    temporal.score_video.return_value = SelectedVideoScoreResult(
        video=VideoEventScores(
            video_id="video-1",
            frame_ids=np.array(["f1"]),
            frame_idx=np.array([1]),
            timestamps_ms=np.array([100], dtype=np.int64),
            scores=np.array([[0.5]], dtype=np.float32),
        ),
        retrieval_ms=1.0,
        decoder_config=DecoderConfigSnapshot(0.1, 1.0, 0.5, 1000),
        scoring_revision="remote-v8",  # Different from capability revision "remote-v7"!
    )
    temporal.snapshot_decoder_config.return_value = None
    temporal.decode_video.return_value = ()

    request = ExplorationOpenRequest(seed=seed_payload(), video_id="video-1", window=(0, 100))
    with pytest.raises(ExplorationConflict):
        _open_branch(temporal, request)


@pytest.mark.anyio
async def test_REQ_008_http_exploration_open_rejects_revision_conflict():
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from hcmai.api.routers.exploration import ExplorationRegistry, create_exploration_router
    from hcmai.orchestration.workflows.temporal_exploration import ExplorationConflict

    app = FastAPI()
    service = Mock()
    service.kis.temporal = Mock()
    registry = ExplorationRegistry()
    app.include_router(create_exploration_router({"service": service, "exploration_registry": registry}))

    with patch(
        "hcmai.api.routers.exploration.run_in_threadpool",
        new=AsyncMock(side_effect=ExplorationConflict("revision mismatch")),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/exploration", json={
                "seed": seed_payload(),
                "video_id": "video-1",
                "window": [0, 100],
            })
    assert response.status_code == 409
    assert len(registry._entries) == 0
