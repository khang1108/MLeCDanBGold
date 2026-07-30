"""Clients for the remote HCMAI inference service."""

from .inference import (
    InferenceClient,
    InferenceClientError,
    RemoteDenseEncoder,
    RemoteFrameCaptioner,
)

__all__ = [
    "InferenceClient",
    "InferenceClientError",
    "RemoteDenseEncoder",
    "RemoteFrameCaptioner",
]
