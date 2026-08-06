from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from hcmai.common.config import VQAConfig
from hcmai.common.schemas import (
    FrameRecord,
    RetrievalCandidate,
    RetrievalResult,
    RetrievalSource,
    VQAInferenceResponse,
    VQARequest,
)
from hcmai.orchestration.pipelines.vqa import VQAPipeline


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


class LLM:
    adapter = SimpleNamespace()

    def answer_vqa(self, *, request_id, frame_id, question, image, evidence):
        assert question == "What color is the bus?"
        return VQAInferenceResponse(
            request_id=request_id, frame_id=frame_id, question=question,
            answer="red", grounded=True, latency_ms=1, evidence=evidence,
        )


def _data(tmp_path: Path) -> Data:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (4, 4), "red").save(image_path)
    return Data(image_path)


def test_vqa_pipeline_runs_retrieval_to_grounded_submission(tmp_path: Path):
    pipeline = VQAPipeline(_data(tmp_path), Retrieval(), LLM(), VQAConfig())

    response = pipeline.execute(VQARequest(
        event_description="A bus passes on the road",
        question="What color is the bus?",
        top_k=5,
    ))

    assert response.total_results == 1
    assert response.submissions[0].video_id == "video-1"
    assert response.submissions[0].frame_idx in {12, 13}
    assert response.submissions[0].normalized_answer == "red"
    assert response.evidence_candidates[0].frame_id == "f1"
    assert set(response.trace.stages) >= {
        "parse", "search", "video_aggregation", "window_construction",
        "evidence_construction", "localization", "answer", "joint_ranking",
        "materialization",
    }


def test_vqa_pipeline_degrades_to_canonical_retrieval_evidence(tmp_path: Path):
    pipeline = VQAPipeline(_data(tmp_path), Retrieval(), None, VQAConfig())

    response = pipeline.execute(VQARequest(
        event_description="A bus passes on the road",
        question="What color is the bus?",
    ))

    assert response.submissions == []
    assert response.evidence_candidates
    assert response.evidence_candidates[0].frame_idx == 12
    assert "returning retrieval evidence" in response.warnings[0]
    assert response.trace.stages["answer"].fallback_used is True
