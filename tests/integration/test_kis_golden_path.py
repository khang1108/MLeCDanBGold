from typing import cast

from hcmai.common.config import SearchConfig
from hcmai.common.schemas import (
    FrameRecord,
    RetrievalCandidate,
    RetrievalResult,
    RetrievalSource,
    SearchRequest,
    TaskType,
)
from hcmai.data.pipeline import DataService
from hcmai.orchestration.workflows.kis import KISPipeline
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.temporal import TemporalEvidenceCore


class CanonicalData:
    frames = {
        "a1": ("video-a", 10, 1_000),
        "a2": ("video-a", 11, 1_300),
        "a3": ("video-a", 40, 8_000),
        "b1": ("video-b", 20, 2_000),
        "c1": ("video-c", 30, 3_000),
    }

    def get_frame(self, frame_id):
        video_id, frame_idx, timestamp_ms = self.frames[frame_id]
        return FrameRecord(
            frame_id=frame_id,
            video_id=video_id,
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            image_path=f"{frame_id}.jpg",
            width=640,
            height=360,
        )

    def get_evidence(self, frame_id, source):
        del frame_id, source
        return None

    def neighbors(self, frame_id, window_ms, include_self=True):
        frame = self.get_frame(frame_id)
        results = []
        for fid, (vid, fidx, t_ms) in self.frames.items():
            if vid == frame.video_id and abs(t_ms - frame.timestamp_ms) <= window_ms:
                if include_self or fid != frame_id:
                    results.append(self.get_frame(fid))
        return results


class BatchRetrieval:
    def __init__(self):
        self.calls = []

    def search(self, query, top_k, filters=None, query_type=None):
        self.calls.append((query, top_k, filters, query_type))
        original = ["a1", "a2", "a3", "b1", "c1"]
        return RetrievalResult(candidates=[
            RetrievalCandidate(
                frame_id=frame_id,
                source_scores={RetrievalSource.VISUAL: 0.9},
                final_score=0.9,
            )
            for frame_id in original
        ])


def test_golden_kis_path_searches_original_query_and_preserves_identity():
    retrieval = BatchRetrieval()
    config = SearchConfig(candidate_count=10, rerank_count=0, temporal_window_ms=500)
    data = cast(DataService, CanonicalData())
    temporal_core = TemporalEvidenceCore(data, cast(RetrievalService, retrieval), config)
    pipeline = KISPipeline(
        TaskType.KIS,
        data,
        cast(RetrievalService, retrieval),
        config,
        temporal_core,
    )

    first = pipeline.execute(SearchRequest(query="red bus 7", top_k=4))
    second = pipeline.execute(SearchRequest(query="red bus 7", top_k=4))

    assert len(first.results) > 0
    assert first.results[0].video_id in {"video-a", "video-b", "video-c"}
    assert [item.frame_ids for item in first.results] == [
        item.frame_ids for item in second.results
    ]
    assert len(retrieval.calls) == 2
    assert retrieval.calls[0][0] == "red bus 7"
