from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from hcmai.common.schemas import VQAInferenceEvidence, VQAInferenceEvidenceItem
from thundercompute.config import HostedVQAConfig, LLMServiceConfig
from thundercompute.adapters.vqa import GroundedVQAModel, _load_backend


class FakeInputs(dict):
    def to(self, device):
        self.device = device
        return self


class FakeModel:
    device = "cuda:0"
    config = SimpleNamespace(_commit_hash="resolved")

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return np.asarray([[1, 2, 3]])


class FakeVQAProcessor:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        assert messages[0]["content"][0]["type"] == "image"
        assert isinstance(messages[0]["content"][0]["image"], Image.Image)
        return FakeInputs(input_ids=np.asarray([[1, 2]]))

    def decode(self, _tokens, **_kwargs):
        return "<think>visual reasoning</think>\nMàu đỏ"


class FakeMultiVQAProcessor:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        images = [
            item for item in messages[0]["content"] if item["type"] == "image"
        ]
        assert len(images) == 2
        return FakeInputs(input_ids=np.asarray([[1, 2]]))

    def decode(self, _tokens, **_kwargs):
        return (
            '{"answer":"red","selected_frame_id":"f2",'
            '"answerable":true,"confidence":0.9}'
        )


def test_vqa_uses_image_and_returns_only_short_final_answer():
    processor, model = FakeVQAProcessor(), FakeModel()
    hosted = GroundedVQAModel(
        HostedVQAConfig(checkpoint="test/model"),
        backend_loader=lambda _: (processor, model),
    )
    answer = hosted.answer_vqa(
        "Màu gì?",
        Image.new("RGB", (2, 2), "red"),
        VQAInferenceEvidence(caption="Một hình vuông màu đỏ."),
        scene_context="Một người chỉ vào hình vuông.",
    )
    assert answer == "Màu đỏ"
    assert "Một hình vuông màu đỏ." in processor.messages[0]["content"][1]["text"]
    assert "Scene context: Một người chỉ vào hình vuông." in processor.messages[0]["content"][1]["text"]
    assert "Question: Màu gì?" in processor.messages[0]["content"][1]["text"]


def test_multi_frame_vqa_preserves_selected_supplied_frame_identity():
    hosted = GroundedVQAModel(
        HostedVQAConfig(checkpoint="test/model"),
        backend_loader=lambda _: (FakeMultiVQAProcessor(), FakeModel()),
    )

    result = hosted.answer_vqa_multi(
        "What color?",
        [Image.new("RGB", (2, 2), "blue"), Image.new("RGB", (2, 2), "red")],
        ["f1", "f2"],
        VQAInferenceEvidence(items=[VQAInferenceEvidenceItem(
            source="caption",
            value="a red square",
            frame_id="f2",
            start_ms=2_000,
            end_ms=2_000,
            confidence=0.9,
            provenance="caption-index-v1",
        )]),
        scene_context="A person points at a square.",
    )

    assert result == {
        "answer": "red",
        "selected_frame_id": "f2",
        "answerable": True,
        "confidence": 0.9,
    }
    content = hosted.processor.messages[0]["content"]
    assert any(item.get("text") == "Frame ID: f2 | timestamp_ms: 2000" for item in content)
    prompt = content[-1]["text"]
    assert "Scene context: A person points at a square." in prompt
    assert "Question: What color?" in prompt
    assert '"frame_id": "f2"' in prompt


def test_glm_backend_uses_official_multimodal_classes(monkeypatch):
    import transformers

    processor, model = FakeVQAProcessor(), FakeModel()
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(model_type="glm4v"),
    )
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda *_args, **_kwargs: processor,
    )
    monkeypatch.setattr(
        transformers.Glm4vForConditionalGeneration,
        "from_pretrained",
        lambda *_args, **_kwargs: model,
    )

    loaded_processor, loaded_model = _load_backend(
        HostedVQAConfig(checkpoint="zai-org/glm", revision="commit")
    )
    assert loaded_processor is processor
    assert loaded_model is model


def test_checked_in_vqa_config_matches_the_active_backend():
    config = LLMServiceConfig.from_yaml("thundercompute/config.yaml")
    assert config.vqa_model.checkpoint == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert config.vqa_model.revision is None
    assert config.vqa_model.max_new_tokens == 256
