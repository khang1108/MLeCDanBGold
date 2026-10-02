"""Adapters for local and hosted frame caption generation.

The local caption backend is Qwen VL; OCR remains owned by the separate
Florence adapter under the OCR package.
"""

from offline.enrichment.caption.adapters.vllm import VLLMCaptionAdapter

__all__ = ["VLLMCaptionAdapter"]
