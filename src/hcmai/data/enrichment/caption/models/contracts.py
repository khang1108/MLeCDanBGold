"""Định nghĩa giao ước (Contracts) cho Caption Enrichment.

Chứa các interfaces (giao thức) mà các mô hình hoặc module captioning cần phải tuân thủ để tích hợp vào hệ thống."""

from __future__ import annotations

from typing import Any, Protocol, Sequence


class CaptionModelConfig(Protocol):
    @property
    def model_checkpoint(self) -> str: ...

    @property
    def revision(self) -> str | None: ...

    @property
    def prompt(self) -> str: ...

    @property
    def decoding(self) -> dict[str, Any]: ...

    @property
    def device(self) -> str: ...

    @property
    def dtype(self) -> str: ...


class CaptionAdapter(Protocol):
    resolved_revision: str | None

    def resolve_revision(self) -> str: ...

    def caption_batch(self, images: Sequence[Any]) -> list[Any]: ...
