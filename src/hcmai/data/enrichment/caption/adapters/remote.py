"""Remote caption-generation adapter."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from PIL import Image

from hcmai.common.schemas import CaptionResponse, InferenceReadiness


class CaptionClient(Protocol):
    def readiness(self) -> InferenceReadiness: ...

    def caption(self, images: Sequence[Image.Image]) -> CaptionResponse: ...


class RemoteCaptionAdapter:
    """Adapt hosted caption inference to the enrichment batch contract."""

    def __init__(self, client: CaptionClient, config: Any) -> None:
        self.client = client
        self.config = config
        self.resolved_revision: str | None = None

    def resolve_revision(self) -> str:
        status = self.client.readiness().models.get("caption_generation")
        if status is None or not status.loaded:
            raise RuntimeError("remote caption model is not ready")
        if status.checkpoint != self.config.model_checkpoint:
            raise ValueError("remote caption checkpoint mismatch")
        if not status.revision:
            raise ValueError("remote caption revision is unresolved")
        self.resolved_revision = status.revision
        return status.revision

    def caption_batch(self, images: Sequence[Any]) -> list[str]:
        response = self.client.caption(images)
        if response.model != self.config.model_checkpoint:
            raise ValueError("remote caption checkpoint mismatch")
        if self.resolved_revision and response.revision != self.resolved_revision:
            raise ValueError("remote caption revision changed")
        return [item.caption for item in response.items]
