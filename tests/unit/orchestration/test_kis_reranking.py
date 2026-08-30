"""Regression tests for path-score KIS without default reranking."""

from __future__ import annotations

from types import SimpleNamespace

from hcmai.api.contracts import SearchRequest
from hcmai.common.schemas import FrameRecord
from hcmai.orchestration.workflows.kis import KISPipeline
from hcmai.temporal import AlignedPath


class _Data:
    """Minimal canonical data provider for the KIS materializer."""

    video_metadata_store = None

    def get_frame(self, frame_id: str) -> FrameRecord:
        """Resolve a canonical frame without adding specialist evidence."""

        return FrameRecord(
            frame_id=frame_id,
            video_id="video-1",
            frame_idx=1,
            timestamp_ms=1_000,
            image_path=f"{frame_id}.jpg",
            width=640,
            height=360,
        )

    def get_evidence(self, frame_id, source):
        """Keep optional evidence absent in the visual alignment baseline."""

        del frame_id, source
        return None

    def get_object_counts(self, frame_id):
        """Keep object evidence absent in the visual alignment baseline."""

        del frame_id
        return None

    def get_transcript_segments_at_time(self, video_id, timestamp_ms):
        """Keep transcript evidence absent in the visual alignment baseline."""

        del video_id, timestamp_ms
        return []


class _Alignment:
    """Return a fixed path whose score differs from all frame-local scores."""

    def search(self, events, *, top_k):
        """Expose a path result without invoking a reranker or model provider."""

        del events, top_k
        frame = _Data().get_frame("frame-a")
        return SimpleNamespace(
            paths=(
                AlignedPath(
                    video_id="video-1",
                    score=1.75,
                    frame_ids=(frame.frame_id,),
                    frame_idxs=(frame.frame_idx,),
                    timestamps_ms=(frame.timestamp_ms,),
                ),
            ),
            retrieval_ms=0.0,
            alignment_ms=0.0,
        )


def test_default_kis_uses_dp_path_score_without_a_rerank_stage() -> None:
    """Ensure a single-frame reranker cannot overwrite baseline path ordering."""

    response = KISPipeline(_Data(), _Alignment()).execute(
        SearchRequest(query="red bus", top_k=1)
    )

    assert response.results[0].score == 1.75
