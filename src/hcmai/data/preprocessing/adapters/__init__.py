"""Remote-capable preprocessing adapters."""

from .remote import (
    RemoteDinoEncoder,
    RemoteEfficientGEBDDetector,
    RemoteTransNetDetector,
)

__all__ = [
    "RemoteDinoEncoder",
    "RemoteEfficientGEBDDetector",
    "RemoteTransNetDetector",
]

