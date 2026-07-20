"""Configuration contracts (inputs/knobs) for the retriever pipeline.

Configs are user-facing inputs that steer how models are built and evaluated,
kept separate from provenance metadata (``metadata.py``) and runtime stats
(``stats.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Recall cut-offs frozen for the baseline comparison.
RECALL_CUTOFFS = (1, 5, 10, 100)


@dataclass
class EncoderConfig:
    """Configuration for the dense encoder."""

    model_name: str = "google/siglip2-base-patch16-224"
    device: str = "cpu"
    batch_size: int = 32
    image_size: int = 224
    dtype: str = "float32"
    precision: str = "fp32"

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> EncoderConfig:
        """Create config from dictionary, using defaults for missing keys."""
        return cls(
            model_name=config.get("name", cls.model_name),
            device=config.get("device", cls.device),
            batch_size=config.get("batch_size", cls.batch_size),
            image_size=config.get("image_size", cls.image_size),
            dtype=config.get("dtype", cls.dtype),
            precision=config.get("precision", cls.precision),
        )


@dataclass
class BenchmarkConfig:
    """Configuration recorded alongside benchmark results for reproducibility."""

    run_name: str
    dataset_version: str
    model_name: str
    index_type: str
    num_queries: int
    top_k: int
    recall_cutoffs: list[int] = field(default_factory=lambda: list(RECALL_CUTOFFS))
