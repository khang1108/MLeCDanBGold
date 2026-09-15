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

