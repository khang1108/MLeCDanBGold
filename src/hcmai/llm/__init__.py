"""Remote model hosting and local inference clients."""

from .client import InferenceClient, RemoteDenseEncoder, RemoteFrameCaptioner

__all__ = ["InferenceClient", "RemoteDenseEncoder", "RemoteFrameCaptioner"]
