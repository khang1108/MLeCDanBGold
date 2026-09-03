"""Stable import and Uvicorn entry point for the hosted inference API.

Endpoint implementations live in ``llm.server.routers``. Keeping this module
small preserves ``llm.server.api:create_llm_app`` and lazy ``:app`` loading for
operators and tests without concentrating transport logic here again.
"""

from __future__ import annotations

from llm.server.app import create_llm_app

__all__ = ["create_llm_app"]


def __getattr__(name: str):
    """Construct the default app only when an ASGI server requests it."""

    if name == "app":
        import sys

        module = sys.modules[__name__]
        application = create_llm_app()
        setattr(module, "app", application)
        return application
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
