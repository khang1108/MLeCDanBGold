"""Configuration scaffolding and contracts for the HCMAI 2026 search pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Recall cut-offs frozen for baseline comparison
RECALL_CUTOFFS: tuple[int, ...] = (1, 5, 10, 100)


class EnrichmentArtifactsConfig(BaseModel):
    """Paths to source-specific frame-enrichment artifacts."""

    caption_path: Path | None = Path(
        "artifacts/enrichment/caption/frame_enrichment.parquet"
    )
    ocr_path: Path | None = Path(
        "artifacts/enrichment/ocr/frame_enrichment.parquet"
    )
    asr_path: Path | None = Path(
        "artifacts/enrichment/asr/frame_enrichment.parquet"
    )


class DatasetConfig(BaseModel):
    """Configuration for corpus metadata and enrichment paths."""

    version: str = "hcmai2026_v1"
    root: Path = Path("data")
    frames_path: Path = Path("data/metadata/frames.parquet")
    enrichment: EnrichmentArtifactsConfig = Field(
        default_factory=EnrichmentArtifactsConfig
    )


class EncoderConfig(BaseModel):
    """Configuration for the dense visual encoder."""

    backend: Literal["siglip", "bge_m3"] = "siglip"
    model_name: str = "google/siglip2-base-patch16-224"
    device: str = "cpu"
    batch_size: int = 32
    image_size: int = 224
    max_length: int = 8192
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
            backend=data.get("backend", default_inst.backend),
            model_name=data.get("model_name", default_inst.model_name),
            device=data.get("device", default_inst.device),
            batch_size=data.get("batch_size", default_inst.batch_size),
            image_size=data.get("image_size", default_inst.image_size),
            max_length=data.get("max_length", default_inst.max_length),
            dtype=data.get("dtype", default_inst.dtype),
            precision=data.get("precision", default_inst.precision),
        )


class IndexConfig(BaseModel):
    """Configuration for the self-contained FAISS artifact directory."""

    type: str = "flat_ip"
    path: Path = Path("artifacts/indexes/visual")
    caption_path: Path = Path("artifacts/indexes/caption")


class FusionConfig(BaseModel):
    """Fusion configuration for multi-modal candidate merging."""

    method: Literal["rrf"] = "rrf"
    rrf_k: int = Field(default=60, gt=0)


class SearchConfig(BaseModel):
    """Single search configuration selected for the competition pipeline."""

    candidate_count: int = Field(default=500, ge=1)
    rerank_count: int = Field(default=100, ge=0)
    temporal_window_ms: int = Field(default=3000, ge=0)
    fusion: FusionConfig = Field(default_factory=FusionConfig)


class ApiConfig(BaseModel):
    """Configuration for API parameters."""

    default_top_k: int = 20
    maximum_top_k: int = 100


class InferenceConfig(BaseModel):
    """Connection from the local data plane to the private GPU service."""

    enabled: bool = False
    base_url: str = "https://api.iamphuckhang.dev"
    timeout_seconds: float = Field(default=10, gt=0, le=120)
    local_embedding_fallback: bool = True
    local_fallback_device: str = "cpu"
    local_fallback_batch_size: int = Field(default=32, ge=1)


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
    index: IndexConfig = Field(default_factory=IndexConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        """Load configuration from a YAML file."""
        from hcmai.common.utils.io import read_yaml

        data = read_yaml(path)
        return cls.model_validate(data)


Settings = AppConfig
