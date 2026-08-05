"""Internal stores owned by the data capability."""

from hcmai.data.stores.evidence import ASRStore, CaptionStore, OCRStore
from hcmai.data.stores.frame import FrameStore

__all__ = ["ASRStore", "CaptionStore", "FrameStore", "OCRStore"]
