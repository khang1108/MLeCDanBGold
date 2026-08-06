"""Protocols for local and remote LLM deployments."""

from __future__ import annotations

from typing import Protocol


class LoadableLLMAdapter(Protocol):
    """Lifecycle contract required by the hosted inference server."""

    def load(self) -> None: ...

    def readiness(self) -> object: ...
