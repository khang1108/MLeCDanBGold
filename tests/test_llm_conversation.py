from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from hcmai.common.schemas import VQAEvidence
from hcmai.llm.config import HostedConversationConfig, LLMServiceConfig
from hcmai.llm.models.conversation import StructuredConversationModel, _load_backend

STATE = {
    "standalone_query": "người đàn ông mặc áo đỏ",
    "positive_constraints": ["áo đỏ"],
    "negative_constraints": [],
    "uncertain_constraints": [],
    "accepted_frame_ids": [],
    "rejected_frame_ids": [],
}


class FakeInputs(dict):
    def to(self, device):
        self.device = device
        return self


class FakeProcessor:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        assert all(isinstance(message["content"], list) for message in messages)
        assert messages[0]["content"][0]["type"] == "text"
        return FakeInputs(input_ids=np.asarray([[1, 2]]))

    def decode(self, _tokens, **_kwargs):
        return f"internal reasoning {{not json}}\n{STATE!r}".replace("'", '"')


class FakeModel:
    device = "cuda:0"
    config = SimpleNamespace(_commit_hash="resolved")

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return np.asarray([[1, 2, 3]])


class FakeVQAProcessor(FakeProcessor):
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        assert messages[0]["content"][0]["type"] == "image"
        assert isinstance(messages[0]["content"][0]["image"], Image.Image)
        return FakeInputs(input_ids=np.asarray([[1, 2]]))

    def decode(self, _tokens, **_kwargs):
        return "<think>visual reasoning</think>\nMàu đỏ"


def test_structured_model_extracts_complete_state_after_reasoning():
    processor, model = FakeProcessor(), FakeModel()
    config = HostedConversationConfig(checkpoint="test/model", max_new_tokens=1024)
    hosted = StructuredConversationModel(
        config, backend_loader=lambda _: (processor, model)
    )
    output = hosted(
        {
            "instruction": "Resolve the conversation.",
            "history": [],
            "current_message": "người đó",
        }
    )
    assert output == STATE
    assert processor.template_kwargs["tokenize"] is True
    assert model.generate_kwargs["max_new_tokens"] == 1024
    assert hosted.revision == "resolved"


def test_vqa_uses_image_and_returns_only_short_final_answer():
    processor, model = FakeVQAProcessor(), FakeModel()
    hosted = StructuredConversationModel(
        HostedConversationConfig(checkpoint="test/model"),
        backend_loader=lambda _: (processor, model),
    )
    answer = hosted.answer_vqa(
        "Màu gì?",
        Image.new("RGB", (2, 2), "red"),
        VQAEvidence(caption="Một hình vuông màu đỏ."),
    )
    assert answer == "Màu đỏ"
    assert "Một hình vuông màu đỏ." in processor.messages[0]["content"][1]["text"]


def test_glm_backend_uses_official_multimodal_classes(monkeypatch):
    import transformers

    processor, model = FakeProcessor(), FakeModel()
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
        HostedConversationConfig(checkpoint="zai-org/glm", revision="commit")
    )
    assert loaded_processor is processor
    assert loaded_model is model


def test_checked_in_glm_config_is_valid_and_pinned():
    config = LLMServiceConfig.from_yaml("llm/config.yaml")
    assert config.conversation.checkpoint == "zai-org/GLM-4.1V-9B-Thinking"
    assert config.conversation.revision
    assert config.conversation.max_new_tokens == 256
