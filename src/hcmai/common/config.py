"""Configuration scaffolding and contracts for the HCMAI 2026 search pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Recall cut-offs frozen for baseline comparison
RECALL_CUTOFFS: tuple[int, ...] = (1, 5, 10, 100)


class DatasetConfig(BaseModel):
    """Configuration for corpus metadata and enrichment paths."""

    version: str = "hcmai2026_v1"
    root: Path = Path("data")
    frames_path: Path = Path("data/metadata/frames.parquet")
    enrichment_path: Path = Path("artifacts/enrichment/frame_enrichment.parquet")


class EncoderConfig(BaseModel):
    """Configuration for the dense visual encoder."""

    model_name: str = "google/siglip2-base-patch16-224"
    device: str = "cpu"
    batch_size: int = 32
    image_size: int = 224
    dtype: str = "float32"
    precision: str = "fp32"

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> EncoderConfig:
        """Create config from dictionary, preserving backward compatibility."""
        data = dict(config)
        if "name" in data and "model_name" not in data:
            data["model_name"] = data["name"]
        default_inst = cls()
        return cls(
            model_name=data.get("model_name", default_inst.model_name),
            device=data.get("device", default_inst.device),
            batch_size=data.get("batch_size", default_inst.batch_size),
            image_size=data.get("image_size", default_inst.image_size),
            dtype=data.get("dtype", default_inst.dtype),
            precision=data.get("precision", default_inst.precision),
        )


class RerankerConfig(BaseModel):
    """Configuration for the reranking stage."""

    enabled: bool = True
    model_name: str = "Qwen/Qwen3-VL-Reranker-2B"
    device: str = "cuda"
    batch_size: int = 8


class ModelConfig(BaseModel):
    """Container for visual embedding and reranking models."""

    embedding: EncoderConfig = Field(default_factory=EncoderConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)


class IndexConfig(BaseModel):
    """Configuration for FAISS vector index."""

    type: str = "flat_ip"
    path: Path = Path("artifacts/indexes/visual.index")
    mapping_path: Path = Path("artifacts/embeddings/frame_mapping.parquet")


class SearchProfile(BaseModel):
    """Parameters for a specific search profile (fast vs accurate)."""

    visual_candidates: int = 100
    rerank_count: int = 20
    temporal_window_ms: int = 1000


class FusionConfig(BaseModel):
    """Fusion configuration for multi-modal candidate merging."""

    method: str = "rrf"
    rrf_k: int = 60


class SearchConfig(BaseModel):
    """Search pipeline configuration containing profiles and fusion settings."""

    profiles: dict[str, SearchProfile] = Field(
        default_factory=lambda: {
            "fast": SearchProfile(
                visual_candidates=100,
                rerank_count=20,
                temporal_window_ms=1000,
            ),
            "accurate": SearchProfile(
                visual_candidates=500,
                rerank_count=100,
                temporal_window_ms=3000,
            ),
        }
    )
    fusion: FusionConfig = Field(default_factory=FusionConfig)


class ApiConfig(BaseModel):
    """Configuration for API parameters."""

    default_top_k: int = 20
    maximum_top_k: int = 100


class BenchmarkConfig(BaseModel):
    """Configuration recorded alongside benchmark results for reproducibility."""

    run_name: str
    dataset_version: str
    model_name: str
    index_type: str
    num_queries: int
    top_k: int
    recall_cutoffs: list[int] = Field(default_factory=lambda: list(RECALL_CUTOFFS))


class AppConfig(BaseSettings):
    """Central settings for the HCMAI 2026 search pipeline."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        """Load configuration from a YAML file."""
        from hcmai.common.utils.io import load_yaml

        data = load_yaml(path)
        return cls.model_validate(data)


Settings = AppConfig
