"""HTTP lifecycle regression for temporal exploration transport.

This test uses the real exploration branch with a deterministic scorer. It
does not duplicate temporal decoding or scoring behaviour at the API boundary.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

from hcmai.app import create_app
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import AlignedPath

pytestmark = pytest.mark.usefixtures("inline_router_threadpool")


@pytest.fixture
def inline_router_threadpool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid the Python 3.14 ASGI worker shutdown deadlock in this route test."""

    async def run_inline(
        function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute the scheduling boundary inline while preserving route logic."""

        return function(*args, **kwargs)

    monkeypatch.setattr(
        "hcmai.api.routers.exploration.run_in_threadpool",
        run_inline,
    )


class _TemporalScorer:
    """Return one canonical path for the real exploration branch."""

    def score_videos(
        self,
        events: tuple[str, ...],
        **_: object,
    ) -> tuple[tuple[VideoEventScores, ...], float]:
        """Return deterministic scores for the selected canonical video."""

        return (
            (
                VideoEventScores(
                    video_id="video-1",
                    frame_ids=np.asarray(["frame-1", "frame-2"], dtype=object),
                    frame_idx=np.asarray([101, 202]),
                    timestamps_ms=np.asarray([1_000, 2_000]),
                    scores=np.ones((len(events), 2)),
                ),
            ),
            0.0,
        )

    def decode_video(
        self,
        _: VideoEventScores,
        **__: object,
    ) -> tuple[AlignedPath, ...]:
        """Return a canonical path whose timestamps expose serialization."""

        return (
            AlignedPath(
                video_id="video-1",
                score=2.0,
                frame_ids=("frame-1", "frame-2"),
                frame_idxs=(101, 202),
                timestamps_ms=(1_000, 2_000),
            ),
        )


def _app() -> FastAPI:
    """Create an application with only the temporal dependency required here."""

    temporal = _TemporalScorer()
    service = SimpleNamespace(kis=SimpleNamespace(temporal=temporal))
    return create_app(search_service=service)


def test_exploration_route_open_confirm_undo_delete_and_old_handle_not_found() -> None:
    """Expose one branch lifecycle while preserving canonical millisecond paths."""

    async def exercise() -> None:
        transport = httpx.ASGITransport(app=_app())
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            opened = await client.post(
                "/api/v1/exploration",
                json={
                    "query": "person enters then leaves",
                    "events": ["person enters", "person leaves"],
                    "retrieval_events": ["person enters", "person leaves"],
                    "caption_events": None,
                    "use_dense": True,
                    "use_bm25": False,
                    "video_id": "video-1",
                    "window": [0, 3_000],
                },
            )
            assert opened.status_code == 200
            envelope = opened.json()
            assert envelope["scoring_revision"]
            assert envelope["view"]["paths"][0]["timestamps_ms"] == [1_000, 2_000]

            confirmed = await client.post(
                f"/api/v1/exploration/{envelope['handle']}/actions",
                json={
                    "expected_revision": envelope["view"]["revision"],
                    "event_version": envelope["view"]["event_version"],
                    "scoring_revision": envelope["scoring_revision"],
                    "action": "confirm",
                    "event_index": 0,
                    "interval": [1_000, 1_000],
                },
            )
            assert confirmed.status_code == 200

            undone = await client.post(
                f"/api/v1/exploration/{envelope['handle']}/actions",
                json={
                    "expected_revision": confirmed.json()["view"]["revision"],
                    "event_version": envelope["view"]["event_version"],
                    "scoring_revision": envelope["scoring_revision"],
                    "action": "undo",
                },
            )
            assert undone.status_code == 200

            deleted = await client.delete(
                f"/api/v1/exploration/{envelope['handle']}",
                params={"expected_revision": undone.json()["view"]["revision"]},
            )
            assert deleted.status_code == 204

            missing = await client.get(f"/api/v1/exploration/{envelope['handle']}")
            assert missing.status_code == 404

    asyncio.run(exercise())
