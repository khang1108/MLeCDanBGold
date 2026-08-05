"""Configuration scaffolding and contracts for the HCMAI 2026 search pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from hcmai.common.schemas.enum import RetrievalSource, TaskType

# Recall cut-offs frozen for baseline comparison
RECALL_CUTOFFS: tuple[int, ...] = (1, 5, 10, 100)
FUSION_SOURCES: tuple[RetrievalSource, ...] = tuple(RetrievalSource)
TEXT_RETRIEVAL_SOURCES: tuple[RetrievalSource, ...] = (
    RetrievalSource.CAPTION,
    RetrievalSource.OCR,
    RetrievalSource.ASR,
)


def _equal_fusion_weights() -> dict[TaskType, dict[RetrievalSource, float]]:
    """Return neutral weights until each task has labeled validation queries."""

    return {
        task: {source: 1.0 for source in FUSION_SOURCES}
        for task in TaskType
    }


def _text_embedding_filenames() -> dict[RetrievalSource, str]:
    """Return default artifact names for frame-aligned text embeddings."""

    return {
        RetrievalSource.CAPTION: "caption_embeddings.npy",
        RetrievalSource.OCR: "ocr_embeddings.npy",
        RetrievalSource.ASR: "asr_embeddings.npy",
    }


def _required_retrieval_sources() -> set[RetrievalSource]:
    return {RetrievalSource.VISUAL}


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


class ASRConfig(BaseModel):
    """Configuration for offline video transcription."""

    model_name: str = "Qwen/Qwen3-ASR-1.7B-hf"
    device: str = "cuda"
    dtype: str = "bfloat16"
    language: str | None = None
    prompt: str | None = None
    max_new_tokens: int = 256
    batch_size: int = Field(default=8, gt=0)
    attn_implementation: str | None = None
    compile_model: bool = False
    audio_sample_rate: int = 16_000
    vad_threshold: float = Field(default=0.5, ge=0, le=1)
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 200
    max_segment_seconds: int = 30


class DiarizationConfig(BaseModel):
    """Configuration for offline speaker diarization."""

    model_name: str = "pyannote/speaker-diarization-community-1"
    device: str = "cuda"
    audio_sample_rate: int = 16_000


class IndexConfig(BaseModel):
    """Configuration for the self-contained FAISS artifact directory."""

    type: str = "flat_ip"
    path: Path = Path("artifacts/indexes/visual")
    caption_path: Path = Path("artifacts/indexes/caption")
    ocr_path: Path = Path("artifacts/indexes/ocr")
    asr_path: Path = Path("artifacts/indexes/asr")
    text_embedding_filenames: dict[RetrievalSource, str] = Field(
        default_factory=_text_embedding_filenames
    )

    @field_validator("text_embedding_filenames")
    @classmethod
    def validate_text_embedding_filenames(
        cls,
        filenames: dict[RetrievalSource, str],
    ) -> dict[RetrievalSource, str]:
        if set(filenames) != set(TEXT_RETRIEVAL_SOURCES):
            raise ValueError(
                "text_embedding_filenames must configure caption, ocr, and asr"
            )
        for filename in filenames.values():
            path = Path(filename)
            if path.name != filename or path.suffix != ".npy":
                raise ValueError(
                    "text embedding filenames must be plain .npy filenames"
                )
        return filenames


class FusionConfig(BaseModel):
    """Fusion configuration for multi-modal candidate merging."""

    method: Literal["rrf"] = "rrf"
    rrf_k: int = Field(default=60, gt=0)
    modality_max_workers: int = Field(default=4, ge=1)
    required_sources: set[RetrievalSource] = Field(
        default_factory=_required_retrieval_sources
    )
    normalize_active_weights: bool = True
    task_weights: dict[TaskType, dict[RetrievalSource, float]] = Field(
        default_factory=_equal_fusion_weights
    )

    @model_validator(mode="after")
    def validate_task_weights(self) -> FusionConfig:
        """Require an explicit positive weight for every task and modality."""

        if set(self.task_weights) != set(TaskType):
            raise ValueError("fusion task_weights must configure every TaskType")
        expected = set(FUSION_SOURCES)
        if not self.required_sources.issubset(expected):
            raise ValueError("required_sources contains an unknown modality")
        for task, weights in self.task_weights.items():
            if set(weights) != expected:
                raise ValueError(
                    f"fusion task_weights[{task.value!r}] must configure "
                    "visual, caption, ocr, and asr"
                )
            if any(weight <= 0 for weight in weights.values()):
                raise ValueError("fusion weights must be greater than zero")
        return self


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
