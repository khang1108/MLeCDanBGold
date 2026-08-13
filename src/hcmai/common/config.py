"""Configuration scaffolding and contracts for the HCMAI 2026 search pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from hcmai.common.schemas.enum import RetrievalSource, TaskType
from hcmai.common.schemas.vqa import VQABaselineProfile

# Recall cut-offs frozen for baseline comparison
RECALL_CUTOFFS: tuple[int, ...] = (1, 5, 20, 50, 100)
FUSION_SOURCES: tuple[RetrievalSource, ...] = tuple(RetrievalSource)
TEXT_RETRIEVAL_SOURCES: tuple[RetrievalSource, ...] = (
    RetrievalSource.CAPTION,
    RetrievalSource.OCR,
    RetrievalSource.ASR,
)


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
    root: Path = Path("artifacts/frame_store")
    frames_path: Path = Path("artifacts/frame_store/frames.parquet")
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
    subset_search_threshold: int = Field(default=100_000, ge=1)
    text_embedding_filenames: dict[RetrievalSource, str] = Field(
        default_factory=lambda: {
            RetrievalSource.CAPTION: "caption_embeddings.npy",
            RetrievalSource.OCR: "ocr_embeddings.npy",
            RetrievalSource.ASR: "asr_embeddings.npy",
        }
    )

    @field_validator("text_embedding_filenames")
    @classmethod
    def validate_text_embedding_filenames(
        cls,
        filenames: dict[RetrievalSource, str],
    ) -> dict[RetrievalSource, str]:
        """Require one safe NumPy artifact filename per text modality."""

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
        default_factory=lambda: {RetrievalSource.VISUAL}
    )
    normalize_active_weights: bool = True
    task_weights: dict[TaskType, dict[RetrievalSource, float]] = Field(
        default_factory=lambda: {
            task: {source: 1.0 for source in FUSION_SOURCES}
            for task in TaskType
        }
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


class RerankerPolicyConfig(BaseModel):
    """Online behavior when the optional reranker is unavailable."""

    required: bool = False


class RetrievalCacheConfig(BaseModel):
    """Bounds for process-local immutable retrieval caches."""

    enabled: bool = True
    prompt_version: str = Field(default="query-v1", min_length=1)
    embedding_ttl_seconds: float = Field(default=3600, gt=0)
    embedding_max_entries: int = Field(default=4096, ge=1)
    embedding_max_bytes: int = Field(default=67_108_864, ge=1)
    thumbnail_ttl_seconds: float = Field(default=3600, gt=0)
    thumbnail_max_entries: int = Field(default=1024, ge=1)
    thumbnail_max_bytes: int = Field(default=134_217_728, ge=1)
    disk_enabled: Literal[False] = False


class ProgressiveSearchConfig(BaseModel):
    """Transactional state, retrieval, and scene budgets for KIS/VQA."""

    architecture: Literal["temporal", "legacy"] = "temporal"

    progressive_state_ttl_seconds: float = Field(default=1800, gt=0)
    progressive_state_max_entries: int = Field(default=256, gt=0)
    progressive_max_hints: int = Field(default=10, gt=0)

    candidate_pool_size: int = Field(default=50, gt=0)

    global_quota: int = Field(default=100, gt=0)
    local_quota: int = Field(default=50, gt=0)

    top_m_evidence: int = Field(default=5, gt=0)

    backfill_max_videos: int = Field(default=10, gt=0)
    backfill_max_units_per_video: int = Field(default=5, gt=0)

    candidate_semantic_weight: float = Field(default=0.45, ge=0)
    candidate_match_weight: float = Field(default=0.25, ge=0)
    candidate_evaluation_weight: float = Field(default=0.30, ge=0)

    scene_max_gap_ms: int = Field(default=5_000, gt=0)
    scene_max_span_ms: int = Field(default=30_000, gt=0)
    scene_coherence_ms: int = Field(default=15_000, gt=0)
    scene_top_b_per_video: int = Field(default=3, gt=0)
    scene_top_p_global: int = Field(default=30, gt=0)
    scene_semantic_weight: float = Field(default=0.45, ge=0)
    scene_coverage_weight: float = Field(default=0.30, ge=0)
    scene_temporal_weight: float = Field(default=0.15, ge=0)
    scene_relation_weight: float = Field(default=0.10, ge=0)

    @model_validator(mode="after")
    def validate_scene_weights(self) -> ProgressiveSearchConfig:
        """Require nonnegative weights with at least one active component."""

        weights = (
            self.scene_semantic_weight,
            self.scene_coverage_weight,
            self.scene_temporal_weight,
            self.scene_relation_weight,
        )
        if sum(weights) <= 0:
            raise ValueError(
                "at least one progressive scene weight must be positive"
            )
        if self.scene_max_gap_ms > self.scene_max_span_ms:
            raise ValueError("scene_max_gap_ms must not exceed scene_max_span_ms")
        candidate_weights = (
            self.candidate_semantic_weight,
            self.candidate_match_weight,
            self.candidate_evaluation_weight,
        )
        if sum(candidate_weights) <= 0:
            raise ValueError(
                "at least one progressive candidate weight must be positive"
            )
        return self

    def diagnostics(self) -> dict[str, int | float]:
        """Return reproducible active budgets without state implementation details."""

        return self.model_dump()


class SearchConfig(BaseModel):
    """Single search configuration selected for the competition pipeline."""

    candidate_count: int = Field(default=500, ge=1)
    rerank_count: int = Field(default=100, ge=0)
    temporal_window_ms: int = Field(default=3000, ge=0)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    reranker: RerankerPolicyConfig = Field(default_factory=RerankerPolicyConfig)
    cache: RetrievalCacheConfig = Field(default_factory=RetrievalCacheConfig)
    progressive: ProgressiveSearchConfig = Field(
        default_factory=lambda: ProgressiveSearchConfig()
    )


class VQAProfileConfig(BaseModel):
    """Hard budgets for one reproducible competition VQA baseline."""

    candidate_videos: int = Field(default=5, ge=1, le=100)
    candidates_per_branch: int = Field(default=100, ge=1, le=1_000)
    window_ms: int = Field(default=15_000, ge=1_000, le=120_000)
    max_windows: int = Field(default=12, ge=1, le=100)
    max_frames_per_window: int = Field(default=4, ge=1, le=32)
    max_evidence_items: int = Field(default=24, ge=1, le=256)
    max_vlm_calls: int = Field(default=8, ge=0, le=100)
    localizer_enabled: bool = True


def _default_vqa_profiles() -> dict[VQABaselineProfile, VQAProfileConfig]:
    """Return reproducible defaults for every supported VQA baseline."""

    return {
        VQABaselineProfile.SINGLE_FRAME: VQAProfileConfig(
            candidate_videos=1,
            window_ms=8_000,
            max_windows=1,
            max_frames_per_window=1,
            max_vlm_calls=1,
            localizer_enabled=False,
        ),
        VQABaselineProfile.VRAG: VQAProfileConfig(
            candidate_videos=10,
            window_ms=15_000,
            max_windows=20,
            max_frames_per_window=4,
            max_vlm_calls=10,
            localizer_enabled=False,
        ),
        VQABaselineProfile.LOCALIZER: VQAProfileConfig(),
        VQABaselineProfile.HIERARCHICAL: VQAProfileConfig(
            candidate_videos=8,
            candidates_per_branch=150,
            window_ms=30_000,
            max_windows=16,
            max_frames_per_window=8,
            max_vlm_calls=12,
        ),
    }


class VQAConfig(BaseModel):
    """Executable VQA profiles selected without hidden inference budgets."""

    default_profile: VQABaselineProfile = VQABaselineProfile.LOCALIZER
    profiles: dict[VQABaselineProfile, VQAProfileConfig] = Field(
        default_factory=_default_vqa_profiles
    )

    @model_validator(mode="after")
    def validate_profiles(self) -> VQAConfig:
        """Require complete baseline coverage and a configured default."""

        if set(self.profiles) != set(VQABaselineProfile):
            raise ValueError("vqa profiles must configure every baseline profile")
        if self.default_profile not in self.profiles:
            raise ValueError("default VQA profile must be configured")
        return self


class ApiConfig(BaseModel):
    """Configuration for API parameters."""

    default_top_k: int = 20
    maximum_top_k: int = 100


class InferenceConfig(BaseModel):
    """Connection from the local data plane to the private GPU service."""

    enabled: bool = False
    base_url: str = "https://api.iamphuckhang.dev"
    timeout_seconds: float = Field(default=10, gt=0, le=120)
    connect_timeout_seconds: float = Field(default=5, gt=0, le=120)
    read_timeout_seconds: float = Field(default=120, gt=0, le=600)
    write_timeout_seconds: float = Field(default=30, gt=0, le=600)
    pool_timeout_seconds: float = Field(default=5, gt=0, le=120)
    max_attempts: int = Field(default=3, ge=1, le=10)
    backoff_initial_seconds: float = Field(default=0.1, ge=0, le=10)
    backoff_max_seconds: float = Field(default=2, ge=0, le=60)
    backoff_jitter_ratio: float = Field(default=0.2, ge=0, le=1)
    circuit_failure_threshold: int = Field(default=3, ge=1, le=100)
    circuit_cooldown_seconds: float = Field(default=30, gt=0, le=3600)
    max_concurrency: int = Field(default=4, ge=1, le=128)
    minimum_retry_budget_seconds: float = Field(default=0.05, ge=0, le=10)

    @model_validator(mode="after")
    def validate_resilience_ranges(self) -> InferenceConfig:
        """Ensure retry backoff bounds form a valid increasing range."""

        if self.backoff_max_seconds < self.backoff_initial_seconds:
            raise ValueError("backoff_max_seconds must not be below initial backoff")
        return self


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
    vqa: VQAConfig = Field(default_factory=VQAConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        """Load configuration from a YAML file."""
        from hcmai.common.utils.io import read_yaml

        data = read_yaml(path)
        return cls.model_validate(data)


Settings = AppConfig
