from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from hcmai.common.schemas import (
    VQAInferenceEvidence,
    VQAInferenceResponse,
)
from thundercompute.adapters.vqa import GroundedVQAModel
from thundercompute.config import HostedVQAConfig


class Inputs(dict):
    def to(self, _device):
        return self


class Processor:
    def apply_chat_template(self, messages, **_kwargs):
        images = [item for item in messages[0]["content"] if item["type"] == "image"]
        assert len(images) == 2
        return Inputs(input_ids=np.asarray([[1, 2]]))

    def decode(self, _tokens, **_kwargs):
        return (
            '{"answer":"red","selected_frame_id":"f2",'
            '"answerable":true,"confidence":0.9}'
        )


class Model:
    device = "cpu"
    config = SimpleNamespace(_commit_hash="test", model_type="glm4v")

    def generate(self, **_kwargs):
        return np.asarray([[1, 2, 3]])


def test_multiframe_model_selects_only_from_ordered_supplied_frames():
    model = GroundedVQAModel(
        HostedVQAConfig(checkpoint="test/model"),
        backend_loader=lambda _: (Processor(), Model()),
    )

    result = model.answer_vqa_multi(
        "What color?",
        [Image.new("RGB", (2, 2), "blue"), Image.new("RGB", (2, 2), "red")],
        ["f1", "f2"],
        VQAInferenceEvidence(),
    )

    assert result["selected_frame_id"] == "f2"
    assert model.supports_multi_image is True


def test_unified_inference_response_rejects_invented_frame_identity():
    with pytest.raises(ValueError, match="selected_frame_id"):
        VQAInferenceResponse(
            request_id="q1",
            video_id="video-1",
            frame_ids=["f1", "f2"],
            selected_frame_id="invented",
            question="What color?",
            answer="red",
            grounded=True,
            latency_ms=1,
        )


def test_unified_inference_response_supports_one_or_many_frames():
    one = VQAInferenceResponse(
        request_id="q1",
        video_id="video-1",
        frame_ids=["f1"],
        selected_frame_id="f1",
        question="What color?",
        answer="red",
        grounded=True,
        latency_ms=1,
    )
    many = VQAInferenceResponse(
        request_id="q2",
        video_id="video-1",
        frame_ids=["f1", "f2"],
        selected_frame_id="f2",
        question="What color?",
        answer="red",
        grounded=True,
        latency_ms=1,
    )

    assert one.selected_frame_id == "f1"
    assert many.frame_ids == ["f1", "f2"]
    assert many.selected_frame_id == "f2"
