from __future__ import annotations

import pytest
from pydantic import ValidationError

from hcmai.common.config import AppConfig, ProgressiveSearchConfig
from hcmai.common.schemas import FrameRecord, RetrievalCandidate
from hcmai.temporal.evidence import retrieval_to_evidence


class Data:
    frame = FrameRecord(
        frame_id="f1", video_id="v1", frame_idx=7, timestamp_ms=700,
        image_path="f1.jpg", width=10, height=10,
    )

    def get_frame(self, frame_id):
        assert frame_id == "f1"
        return self.frame


def test_yaml_exposes_the_same_progressive_budget_contract():
    config = AppConfig.from_yaml("configs/baseline.yaml")
    assert config.search.progressive.architecture == "temporal"
    assert config.search.progressive.top_m_evidence == 5
    overridden = config.search.progressive.model_copy(update={"top_m_evidence": 2})
    assert overridden.top_m_evidence == 2


def test_progressive_budgets_and_weights_are_validated():
    with pytest.raises(ValidationError):
        ProgressiveSearchConfig(candidate_pool_size=0)
    with pytest.raises(ValidationError, match="at least one"):
        ProgressiveSearchConfig(
            scene_semantic_weight=0,
            scene_coverage_weight=0,
            scene_temporal_weight=0,
            scene_relation_weight=0,
        )
    with pytest.raises(ValidationError, match="candidate weight"):
        ProgressiveSearchConfig(
            candidate_semantic_weight=0,
            candidate_match_weight=0,
            candidate_evaluation_weight=0,
        )


def test_retrieval_adapter_uses_canonical_frame_and_rejects_conflicts():
    item = retrieval_to_evidence(
        RetrievalCandidate(frame_id="f1", final_score=0.8), "h0", Data()
    )
    assert item.frame is Data.frame
    with pytest.raises(ValueError, match="conflicts with canonical"):
        retrieval_to_evidence(
            RetrievalCandidate(
                frame_id="f1", final_score=0.8,
                metadata={"video_id": "wrong"},
            ),
            "h0",
            Data(),
        )
    with pytest.raises(ValueError, match="conflicts with canonical"):
        retrieval_to_evidence(
            RetrievalCandidate(
                frame_id="f1",
                final_score=0.8,
                metadata={"frame": {"video_id": "wrong"}},
            ),
            "h0",
            Data(),
        )
