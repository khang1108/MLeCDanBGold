"""Verify aligned caption batching across local preparation and the GPU API."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from llm.config import LLMServiceConfig
from llm.server.routers import captions
from offline.config import VBSConfig
from offline.enrichment.caption.config import CaptionJobConfig
from offline.pipeline.runner import command_for


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_REQ_001_caption_batch_limit_is_aligned_at_320() -> None:
    """Keep the local request batch within the configured server ceiling."""

    server = LLMServiceConfig.from_yaml(PROJECT_ROOT / "llm/config.yaml")
    local = CaptionJobConfig.from_yaml(PROJECT_ROOT / "configs/vbs_prepare.yaml")

    assert server.caption_generation.max_batch_size == 320
    assert local.caption.batch_size == server.caption_generation.max_batch_size


def test_REQ_002_caption_route_uses_configured_batch_limit(monkeypatch) -> None:
    """Use the hosted caption limit instead of a router-level magic number."""

    observed: dict[str, int] = {}

    class DecodedImage:
        def close(self) -> None:
            """Match the PIL close contract used by the route cleanup."""

    def fake_decode_images(item_ids, images, *, maximum):
        observed["maximum"] = maximum
        return ["frame-1"], [DecodedImage()]

    runtime = SimpleNamespace(
        config=SimpleNamespace(
            caption_generation=SimpleNamespace(
                max_batch_size=17,
                model_checkpoint="test-captioner",
            )
        ),
        captioner=SimpleNamespace(resolved_revision="revision-1"),
        caption=lambda images: ["a test frame"],
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(runtime=runtime))
    )
    monkeypatch.setattr(captions, "decode_images", fake_decode_images)

    response = asyncio.run(captions.caption(request, item_ids="[]", images=[]))

    assert observed["maximum"] == 17
    assert response.items[0].item_id == "frame-1"


def test_REQ_003_caption_runner_uses_sixteen_image_workers() -> None:
    """Prepare a 256-image request with enough parallel local image loaders."""

    config_path = PROJECT_ROOT / "configs/vbs_prepare.yaml"
    command = command_for("caption", config_path, VBSConfig.from_yaml(config_path))

    assert command[-2:] == ["--image-workers", "16"]
