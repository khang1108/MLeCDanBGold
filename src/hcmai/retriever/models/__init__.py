"""Configuration contracts for the retriever pipeline.

Canonical configuration classes reside in ``hcmai.common.config``.
This module re-exports them to preserve backward compatibility.
"""

from __future__ import annotations

from hcmai.common.config import RECALL_CUTOFFS, BenchmarkConfig, EncoderConfig

__all__ = ["EncoderConfig", "BenchmarkConfig", "RECALL_CUTOFFS", "IndexMetadata", "EncodingStats"]