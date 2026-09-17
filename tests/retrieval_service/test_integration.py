"""Transport equivalence tests proving remote retrieval matches local retrieval.

These integration tests instantiate a hand-checkable two-video runtime, serve it
over an in-process gRPC server, and verify that the remote adapters return
bit-level equivalent paths, decoder configurations, video score matrices, and
image search results.
"""

from __future__ import annotations

import io
from typing import Iterator
import numpy as np
from PIL import Image
import pytest

from hcmai.api.contracts.search import SearchLatency, SearchResult, SearchResultMetadata
from hcmai.api.contracts import ImageSearchResponse
from hcmai.corpus import Corpus
from hcmai.corpus.models import Frame
from hcmai.orchestration.utils.materializer import SearchMaterializer
from hcmai.orchestration.workflows.temporal_search import (
    DecoderConfigSnapshot,
    SelectedVideoScoreResult,
    TemporalSearchArtifact,
    TemporalSearchResult,
)
from hcmai.retrieval.plan import KISRetrievalEvent, KISRetrievalPlan
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.retrieval_service.client import RetrievalGrpcClient
from hcmai.retrieval_service.config import RetrievalClientSettings
from hcmai.retrieval_service.remote import (
    RemoteImageSearchService,
    RemoteTemporalSearchService,
)
from hcmai.retrieval_service.runtime import RetrievalRuntime
from hcmai.retrieval_service.server import create_server
from hcmai.temporal.dp import AlignedPath


def _create_test_png() -> bytes:
    """Generate a minimal valid 1x1 PNG image."""
    buffer = io.BytesIO()
    img = Image.new("RGB", (1, 1), color="red")
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class _FakeCorpus:
    """Hand-checkable two-video corpus with two frames each."""

    def __init__(self) -> None:
        self.frames = {
            "v1_f1": Frame(
                video_id="video-1",
                frame_id="v1_f1",
                frame_idx=10,
                timestamp_ms=1000,
                image_path="/tmp/v1_f1.jpg",
            ),
            "v1_f2": Frame(
                video_id="video-1",
                frame_id="v1_f2",
                frame_idx=20,
                timestamp_ms=2000,
                image_path="/tmp/v1_f2.jpg",
            ),
            "v2_f1": Frame(
                video_id="video-2",
                frame_id="v2_f1",
                frame_idx=100,
                timestamp_ms=5000,
                image_path="/tmp/v2_f1.jpg",
            ),
            "v2_f2": Frame(
                video_id="video-2",
                frame_id="v2_f2",
                frame_idx=200,
                timestamp_ms=6000,
                image_path="/tmp/v2_f2.jpg",
            ),
        }

    def frame(self, frame_id: str) -> Frame:
        if frame_id not in self.frames:
            raise KeyError(frame_id)
        return self.frames[frame_id]

    def title(self, video_id: str) -> str:
        return f"Title {video_id}"

    def caption(self, frame_id: str) -> str:
        return f"Caption {frame_id}"

    def ocr(self, frame_id: str) -> str:
        return f"OCR {frame_id}"

    def transcript(self, video_id: str, start_ms: int = 0, end_ms: int | None = None) -> str:
        return f"Transcript {video_id}"

    def objects(self, frame_id: str) -> tuple[()]:
        return ()


class _FakeTemporalService:
    """Deterministic two-video temporal search service."""

    def __init__(self, corpus: _FakeCorpus, scoring_revision: str) -> None:
        self.corpus = corpus
        self.scoring_revision = scoring_revision
        self.decoder_config = DecoderConfigSnapshot(
            lambda_gap=0.15,
            event_power=1.0,
            cluster_delta=0.8,
            path_min_separation_ms=1500,
        )

        self.v1_scores = VideoEventScores(
            video_id="video-1",
            frame_ids=np.array(["v1_f1", "v1_f2"]),
            frame_idx=np.array([10, 20]),
            timestamps_ms=np.array([1000, 2000], dtype=np.int64),
            scores=np.array([[0.85, 0.40], [0.30, 0.90]], dtype=np.float32),
        )
        self.v2_scores = VideoEventScores(
            video_id="video-2",
            frame_ids=np.array(["v2_f1", "v2_f2"]),
            frame_idx=np.array([100, 200]),
            timestamps_ms=np.array([5000, 6000], dtype=np.int64),
            scores=np.array([[0.75, 0.20], [0.10, 0.80]], dtype=np.float32),
        )

        self.path1 = AlignedPath(
            video_id="video-1",
            score=0.92,
            frame_ids=("v1_f1", "v1_f2"),
            frame_idxs=(10, 20),
            timestamps_ms=(1000, 2000),
        )
        self.path2 = AlignedPath(
            video_id="video-2",
            score=0.78,
            frame_ids=("v2_f1", "v2_f2"),
            frame_idxs=(100, 200),
            timestamps_ms=(5000, 6000),
        )

    def search_plan_artifact(
        self,
        plan: KISRetrievalPlan,
        *,
        image_component: object | None = None,
        use_dense: bool = True,
        use_bm25: bool = False,
        top_k: int = 20,
    ) -> TemporalSearchArtifact:
        result = TemporalSearchResult(
            paths=(self.path1, self.path2)[:top_k],
            retrieval_ms=12.5,
            alignment_ms=3.2,
        )
        scores = (self.v1_scores, self.v2_scores)[:top_k]
        return TemporalSearchArtifact(
            result=result,
            video_scores=scores,
            decoder_config=self.decoder_config,
            scoring_revision=self.scoring_revision,
        )

    def search(
        self,
        original_events: Sequence[str],
        *,
        top_k: int = 20,
        retrieval_events: Sequence[str] | None = None,
        caption_events: Sequence[str] | None = None,
        use_dense: bool = True,
        use_bm25: bool = False,
    ) -> TemporalSearchResult:
        return TemporalSearchResult(
            paths=(self.path1, self.path2)[:top_k],
            retrieval_ms=14.0,
            alignment_ms=4.1,
        )

    def score_video(
        self,
        plan: KISRetrievalPlan,
        *,
        video_id: str,
        image_component: object | None = None,
        use_dense: bool = True,
        use_bm25: bool = False,
    ) -> SelectedVideoScoreResult:
        if video_id == "video-1":
            return SelectedVideoScoreResult(
                video=self.v1_scores,
                retrieval_ms=5.5,
                decoder_config=self.decoder_config,
                scoring_revision=self.scoring_revision,
            )
        elif video_id == "video-2":
            return SelectedVideoScoreResult(
                video=self.v2_scores,
                retrieval_ms=6.0,
                decoder_config=self.decoder_config,
                scoring_revision=self.scoring_revision,
            )
        raise KeyError(f"video {video_id!r} not found")

    def snapshot_decoder_config(self) -> DecoderConfigSnapshot:
        return self.decoder_config


class _FakeImageSearchService:
    """Deterministic image search service returning hand-checkable results."""

    def __init__(self, corpus: _FakeCorpus) -> None:
        self.corpus = corpus
        self.materializer = SearchMaterializer(corpus)  # type: ignore[arg-type]

    def search(
        self,
        payload: bytes,
        *,
        content_type: str | None,
        top_k: int,
    ) -> ImageSearchResponse:
        f1 = self.corpus.frame("v1_f1")
        f2 = self.corpus.frame("v2_f1")
        p1 = AlignedPath(
            video_id=f1.video_id,
            score=0.96,
            frame_ids=(f1.frame_id,),
            frame_idxs=(f1.frame_idx,),
            timestamps_ms=(f1.timestamp_ms,),
        )
        p2 = AlignedPath(
            video_id=f2.video_id,
            score=0.88,
            frame_ids=(f2.frame_id,),
            frame_idxs=(f2.frame_idx,),
            timestamps_ms=(f2.timestamp_ms,),
        )
        results = [
            self.materializer.build_kis_result(p1),
            self.materializer.build_kis_result(p2),
        ][:top_k]
        return ImageSearchResponse(
            results=results,
            latency=SearchLatency(
                query_ms=8.0,
                retrieval_ms=15.0,
                materialization_ms=1.5,
                total_ms=24.5,
            ),
        )


@pytest.fixture
def test_environment() -> Iterator[
    tuple[RetrievalRuntime, RemoteTemporalSearchService, RemoteImageSearchService]
]:
    corpus = _FakeCorpus()
    revision = "rev-integration-equiv"
    temporal = _FakeTemporalService(corpus, scoring_revision=revision)
    image_search = _FakeImageSearchService(corpus)

    runtime = RetrievalRuntime(
        corpus=corpus,  # type: ignore[arg-type]
        temporal=temporal,
        image_scorer=None,
        image_search=image_search,  # type: ignore[arg-type]
        active_modalities=("visual", "context", "bm25"),
        startup_messages=(),
        max_temporal_event_count=5,
        image_max_upload_bytes=10 * 1024 * 1024,
        image_max_pixels=4096 * 4096,
        scoring_revision=revision,
    )

    server, _, port = create_server(runtime, host="127.0.0.1", port=0)
    server.start()

    client = RetrievalGrpcClient(
        settings=RetrievalClientSettings(
            target=f"127.0.0.1:{port}",
            timeout_seconds=5.0,
        )
    )
    remote_temporal = RemoteTemporalSearchService(corpus=corpus, client=client)  # type: ignore[arg-type]
    remote_image = RemoteImageSearchService(
        corpus=corpus,  # type: ignore[arg-type]
        client=client,
        max_upload_bytes=10 * 1024 * 1024,
        max_pixels=4096 * 4096,
    )

    try:
        yield runtime, remote_temporal, remote_image
    finally:
        client.close()
        server.stop(grace=None)


def test_REQ_004_search_plan_artifact_transport_equivalence(test_environment) -> None:
    runtime, remote_temporal, _ = test_environment
    plan = KISRetrievalPlan(
        events=(
            KISRetrievalEvent("E1", "a red car", "a red car", "red car"),
            KISRetrievalEvent("E2", "a person runs", "a person runs", "person runs"),
        )
    )

    local = runtime.temporal.search_plan_artifact(plan, top_k=2)
    remote = remote_temporal.search_plan_artifact(plan, top_k=2)

    # 1. Aligned paths equivalence
    assert remote.result.paths == local.result.paths

    # 2. Decoder configuration equivalence
    assert remote.decoder_config == local.decoder_config

    # 3. Scoring revision equivalence
    assert remote.scoring_revision == local.scoring_revision

    # 4. Video score matrices equivalence
    assert [v.video_id for v in remote.video_scores] == [v.video_id for v in local.video_scores]
    for r_score, l_score in zip(remote.video_scores, local.video_scores, strict=True):
        assert r_score.video_id == l_score.video_id
        assert tuple(r_score.frame_ids) == tuple(l_score.frame_ids)
        np.testing.assert_array_equal(r_score.frame_idx, l_score.frame_idx)
        np.testing.assert_array_equal(r_score.timestamps_ms, l_score.timestamps_ms)
        np.testing.assert_allclose(r_score.scores, l_score.scores, rtol=0, atol=0)


def test_REQ_006_trake_search_events_transport_equivalence(test_environment) -> None:
    runtime, remote_temporal, _ = test_environment
    events = ["person walks into building", "person leaves by rear door"]

    local = runtime.temporal.search(events, top_k=2)
    remote = remote_temporal.search(events, top_k=2)

    assert remote.paths == local.paths


def test_REQ_008_score_video_transport_equivalence(test_environment) -> None:
    runtime, remote_temporal, _ = test_environment
    plan = KISRetrievalPlan(
        events=(
            KISRetrievalEvent("E1", "a red car", "a red car", "red car"),
            KISRetrievalEvent("E2", "a person runs", "a person runs", "person runs"),
        )
    )

    local = runtime.temporal.score_video(plan, video_id="video-1")
    remote = remote_temporal.score_video(plan, video_id="video-1")

    assert remote.video.video_id == local.video.video_id
    assert tuple(remote.video.frame_ids) == tuple(local.video.frame_ids)
    np.testing.assert_array_equal(remote.video.frame_idx, local.video.frame_idx)
    np.testing.assert_array_equal(remote.video.timestamps_ms, local.video.timestamps_ms)
    np.testing.assert_allclose(remote.video.scores, local.video.scores, rtol=0, atol=0)
    assert remote.decoder_config == local.decoder_config
    assert remote.scoring_revision == local.scoring_revision


def test_REQ_007_image_search_transport_equivalence(test_environment) -> None:
    runtime, _, remote_image = test_environment
    png_bytes = _create_test_png()

    local = runtime.image_search.search(png_bytes, content_type="image/png", top_k=2)
    remote = remote_image.search(png_bytes, content_type="image/png", top_k=2)

    assert len(remote.results) == len(local.results)
    local_summary = [
        (r.video_id, r.frame_id, r.frame_idx, r.timestamp_ms, r.score)
        for r in local.results
    ]
    remote_summary = [
        (r.video_id, r.frame_id, r.frame_idx, r.timestamp_ms, r.score)
        for r in remote.results
    ]
    assert remote_summary == local_summary


def test_REQ_009_canonical_identity_preserved_across_transport(test_environment) -> None:
    runtime, remote_temporal, remote_image = test_environment
    corpus = runtime.corpus
    plan = KISRetrievalPlan(
        events=(
            KISRetrievalEvent("E1", "test event 1", "test event 1", "test event 1"),
            KISRetrievalEvent("E2", "test event 2", "test event 2", "test event 2"),
        )
    )

    # 1. KIS Paths
    artifact = remote_temporal.search_plan_artifact(plan, top_k=2)
    for path in artifact.result.paths:
        for fid, fidx, ts in zip(path.frame_ids, path.frame_idxs, path.timestamps_ms, strict=True):
            frame = corpus.frame(fid)
            assert frame.video_id == path.video_id
            assert frame.frame_idx == fidx
            assert frame.timestamp_ms == ts

    # 2. Selected Video
    selected = remote_temporal.score_video(plan, video_id="video-1")
    for fid, fidx, ts in zip(
        selected.video.frame_ids,
        selected.video.frame_idx,
        selected.video.timestamps_ms,
        strict=True,
    ):
        frame = corpus.frame(fid)
        assert frame.video_id == selected.video.video_id
        assert frame.frame_idx == fidx
        assert frame.timestamp_ms == ts

    # 3. Image Search
    png_bytes = _create_test_png()
    img_resp = remote_image.search(png_bytes, content_type="image/png", top_k=2)
    for res in img_resp.results:
        frame = corpus.frame(res.frame_id)
        assert frame.video_id == res.video_id
        assert frame.frame_idx == res.frame_idx
        assert frame.timestamp_ms == res.timestamp_ms
