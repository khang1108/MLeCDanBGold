from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from hcmai.common.config import ProgressiveSearchConfig, VQAConfig
from hcmai.common.schemas import (
    FrameEvidence,
    FrameRecord,
    RetrievalCandidate,
    RetrievalResult,
    RetrievalSource,
    RetrievalTrace,
    SceneCandidate,
    VQAInferenceResponse,
    VQARequest,
)
from hcmai.pipelines.vqa.pipeline import VQAPipeline
from hcmai.orchestration.workflows.base import TaskPipelineDependencyError
from hcmai.temporal import ProgressiveLocalizationResult
from hcmai.temporal.query import diff_snapshot


class Data:
    def __init__(self, image_path: Path):
        self.frames = {
            "f1": FrameRecord(
                frame_id="f1", video_id="video-1", frame_idx=12,
                timestamp_ms=1_000, image_path=str(image_path), width=4, height=4,
            ),
            "f2": FrameRecord(
                frame_id="f2", video_id="video-1", frame_idx=13,
                timestamp_ms=2_000, image_path=str(image_path), width=4, height=4,
            ),
        }

    def get_frame(self, frame_id):
        return self.frames[frame_id]

    def neighbors(self, frame_id, *, window_ms, include_self=True):
        center = self.frames[frame_id]
        return [
            frame for frame in self.frames.values()
            if abs(frame.timestamp_ms - center.timestamp_ms) <= window_ms
            and (include_self or frame.frame_id != frame_id)
        ]

    def get_evidence(self, frame_id, source):
        values = {
            RetrievalSource.CAPTION: "A red bus is passing.",
            RetrievalSource.OCR: "BUS 01",
        }
        return values.get(source)


class Retrieval:
    def search_batch(self, queries, top_k, filters, query_type):
        assert len(queries) >= 2
        return [
            RetrievalResult(candidates=[RetrievalCandidate(
                frame_id="f1", source_scores={RetrievalSource.VISUAL: 0.9},
                source_ranks={RetrievalSource.VISUAL: 1}, final_score=0.9,
            )]),
            *[
                RetrievalResult(candidates=[RetrievalCandidate(
                    frame_id="f2", source_scores={RetrievalSource.OCR: 0.8},
                    source_ranks={RetrievalSource.OCR: 1}, final_score=0.8,
                )])
                for _ in queries[1:]
            ],
        ]


class TemporalCore:
    """Small shared-core stand-in for the pipeline contract test."""

    def __init__(self, data: Data) -> None:
        self.data = data

    def localize(self, event_description, *, search_id, **_):
        frame = self.data.get_frame("f1")
        evidence = FrameEvidence(
            frame=frame,
            unit_scores={"h0": 0.9},
            source_scores={RetrievalSource.VISUAL: 0.9},
            source_ranks={RetrievalSource.VISUAL: 1},
            score=0.9,
            provenance=("h0",),
        )
        scene = SceneCandidate(
            scene_id="video-1:1000-1000",
            video_id="video-1",
            start_ms=1_000,
            end_ms=1_000,
            evidence=(evidence,),
            final_score=0.9,
        )
        return ProgressiveLocalizationResult(
            search_id=search_id or "search-session-1",
            version=1,
            scenes=(scene,),
            diff=diff_snapshot(None, event_description),
            warnings=(),
            diagnostics=ProgressiveSearchConfig().diagnostics(),
            trace=RetrievalTrace(),
        )


class LLM:
    adapter = SimpleNamespace()

    def answer_vqa(
        self, *, request_id, frame_id, video_id, scene_context, question, image, evidence
    ):
        assert scene_context == "A bus passes on the road"
        assert question == "What color is the bus?"
        assert video_id == "video-1"
        assert {(item.frame_id, item.start_ms) for item in evidence.items} == {
            ("f1", 1_000),
        }
        return VQAInferenceResponse(
            request_id=request_id, video_id=video_id, frame_ids=[frame_id],
            selected_frame_id=frame_id,
            question=question,
            answer="red", grounded=True, latency_ms=1, evidence=evidence,
        )


def _data(tmp_path: Path) -> Data:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (4, 4), "red").save(image_path)
    return Data(image_path)


def test_vqa_pipeline_runs_retrieval_to_grounded_submission(tmp_path: Path):
    data = _data(tmp_path)
    pipeline = VQAPipeline(
        data,
        Retrieval(),
        LLM(),
        VQAConfig(),
        temporal_core=TemporalCore(data),
    )

    response = pipeline.execute(VQARequest(
        event_description="A bus passes on the road",
        question="What color is the bus?",
        top_k=5,
        search_id="search-session-1",
    ))

    assert response.total_results == 1
    assert response.search_id == "search-session-1"
    assert response.submissions[0].video_id == "video-1"
    assert response.submissions[0].frame_idx in {12, 13}
    assert response.submissions[0].normalized_answer == "red"
    assert response.submissions[0].caption == "A red bus is passing."
    assert response.evidence_candidates[0].frame_id == "f1"
    assert set(response.trace.stages) >= {
        "parse", "search", "video_aggregation", "window_construction",
        "evidence_construction", "localization", "answer", "joint_ranking",
        "materialization",
    }


def test_vqa_pipeline_degrades_to_canonical_retrieval_evidence(tmp_path: Path):
    data = _data(tmp_path)
    pipeline = VQAPipeline(
        data,
        Retrieval(),
        None,
        VQAConfig(),
        temporal_core=TemporalCore(data),
    )

    response = pipeline.execute(VQARequest(
        event_description="A bus passes on the road",
        question="What color is the bus?",
    ))

    assert response.submissions == []
    assert response.evidence_candidates
    assert response.evidence_candidates[0].frame_idx == 12
    assert "returning retrieval evidence" in response.warnings[0]
    assert response.trace.stages["answer"].fallback_used is True


def test_vqa_pipeline_requires_shared_temporal_core(tmp_path: Path):
    pipeline = VQAPipeline(_data(tmp_path), Retrieval(), None, VQAConfig())

    with pytest.raises(TaskPipelineDependencyError, match="Temporal evidence core"):
        pipeline.execute(VQARequest(
            event_description="A bus passes on the road",
            question="What color is the bus?",
        ))
