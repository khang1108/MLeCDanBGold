"""Unit tests for VLLMCaptionAdapter with mocked OpenAI Vision endpoints."""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image
import pytest

from offline.enrichment.caption.adapter import (
    VLLMCaptionAdapter,
    _encode_image_to_base64,
)


def _make_dummy_image(color: tuple[int, int, int] = (255, 0, 0)) -> Image.Image:
    """Create a 32x32 RGB image for testing."""
    return Image.new("RGB", (32, 32), color=color)


def test_REQ_001_encode_image_to_base64() -> None:
    """Encode PIL image to valid base64 JPEG string."""
    img = _make_dummy_image()
    b64_str = _encode_image_to_base64(img)
    assert isinstance(b64_str, str)
    decoded = base64.b64decode(b64_str)
    assert len(decoded) > 0


def test_REQ_002_adapter_uses_custom_batch_fn() -> None:
    """Support direct mock batch_fn for testing without HTTP."""
    config = SimpleNamespace(
        model_checkpoint="Qwen/Qwen2.5-VL-7B-Instruct",
        revision="test-rev",
    )
    adapter = VLLMCaptionAdapter(
        config=config,
        batch_fn=lambda imgs: [f"caption-{i}" for i in range(len(imgs))],
    )
    assert adapter.resolve_revision() == "test-rev"
    results = adapter.caption_batch([_make_dummy_image(), _make_dummy_image()])
    assert results == ["caption-0", "caption-1"]


def test_REQ_003_caption_batch_openai_request() -> None:
    """Format and send OpenAI vision chat completion requests."""
    config = SimpleNamespace(
        model_checkpoint="Qwen/Qwen2.5-VL-7B-Instruct",
        revision=None,
    )
    adapter = VLLMCaptionAdapter(
        config=config,
        base_url="http://mock-vllm:8001/v1",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "A high-angle view of a red surface in clear lighting."
                }
            }
        ]
    }

    with patch("requests.post", return_value=mock_response) as mock_post:
        captions = adapter.caption_batch([_make_dummy_image()])
        assert len(captions) == 1
        assert captions[0] == "A high-angle view of a red surface in clear lighting."
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert "http://mock-vllm:8001/v1/chat/completions" in mock_post.call_args[0]
        json_data = call_kwargs["json"]
        assert json_data["model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
        content_items = json_data["messages"][0]["content"]
        assert any(item.get("type") == "image_url" for item in content_items)
