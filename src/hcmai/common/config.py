"""Configuration scaffolding and contracts for the HCMAI 2026 search pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from pydantic import ConfigDict, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from hcmai.common.schemas.enum import RetrievalSource, TaskType

# Recall cut-offs frozen for baseline comparison
RECALL_CUTOFFS: tuple[int, ...] = (1, 5, 20, 50, 100)
FUSION_SOURCES: tuple[RetrievalSource, ...] = tuple(RetrievalSource)
TEXT_RETRIEVAL_SOURCES: tuple[RetrievalSource, ...] = (
    RetrievalSource.CAPTION,
    RetrievalSource.OCR,
    RetrievalSource.ASR,
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LEGACY_DATASET_ROOT = (REPOSITORY_ROOT / "artifacts/frame_store").resolve()


def resolve_repository_path(value: str | Path) -> Path:
    """Resolve one runtime path from the repository instead of process CWD.

    Backend processes may be launched by Uvicorn, Supervisor, or an IDE from
    different working directories. Repository-owned ``data`` and ``artifacts``
    paths must therefore have one stable meaning across every launcher.
    """

    path = Path(value).expanduser()
    base = path if path.is_absolute() else REPOSITORY_ROOT / path
    return base.resolve()


def resolve_dataset_root(value: str | Path) -> Path:
    """Resolve the visual-data root and migrate the retired frame-store root.

    Older launchers set ``HCMAI_DATASET_ROOT=artifacts/frame_store`` because
    metadata and images once shared a root. Metadata remains there, but BTC
    keyframes now live exclusively under ``data/keyframes``. Treat only that
    exact legacy value as a compatibility alias; other explicit roots retain
    their configured meaning.
    """

    resolved = resolve_repository_path(value)
    if resolved == LEGACY_DATASET_ROOT:
        return (REPOSITORY_ROOT / "data").resolve()
    return resolved


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
    context_path: Path | None = Path(
        "artifacts/enrichment/context/frame_context_v1.parquet"
    )
    transcripts_path: Path | None = Path("artifacts/enrichment/transcripts")


class DatasetConfig(BaseModel):
    """Configuration for corpus metadata and enrichment paths."""

    version: str = "hcmai2026_v1"
    root: Path = Path("data")
    frames_path: Path = Path("artifacts/frame_store/frames.parquet")
    enrichment: EnrichmentArtifactsConfig = Field(
        default_factory=EnrichmentArtifactsConfig
    )


class EncoderConfig(BaseModel):
    """Configuration for one dense visual or text encoder.

    ``batch_size`` is the explicit inference and remote-request ceiling for
    this encoder. It must remain positive so every caller can safely use it as
    a range step without silently falling back to a different batch size.
    """

    backend: Literal["siglip", "bge_m3"] = "siglip"
    model_name: str = "google/siglip2-base-patch16-224"
    revision: str | None = None
    device: str = "cpu"
    batch_size: int = Field(default=32, gt=0)
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
            revision=data.get("revision", default_inst.revision),
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
    revision: str = Field(
        default="bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
        pattern=r"^[0-9a-f]{40}$",
    )
    device: str = "cuda"
    dtype: str = "bfloat16"
    language: str | None = None
    prompt: str | None = None
    max_new_tokens: int = 256
    batch_size: int = Field(default=32, gt=0)
    attn_implementation: str | None = None
    compile_model: bool = False
    audio_sample_rate: int = Field(default=16_000, gt=0)
    vad_threshold: float = Field(default=0.5, ge=0, le=1)
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 200
    max_segment_seconds: int = 30


class DiarizationConfig(BaseModel):
    """Configuration for offline speaker diarization."""

    enabled: bool = True
    model_name: str = "pyannote/speaker-diarization-community-1"
    revision: str = Field(
        default="3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
        pattern=r"^[0-9a-f]{40}$",
    )
    device: str = "cuda"
    audio_sample_rate: int = Field(default=16_000, gt=0)


class TranscriptJobConfig(BaseModel):
    """Reproducible transcript preparation and frame-materialization settings."""

    asr: ASRConfig = Field(default_factory=ASRConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    pipeline_version: str = Field(default="transcript-pipeline-v1", min_length=1)
    schema_version: str = Field(default="transcript-segment-v1", min_length=1)
    enrichment_version: str = Field(default="asr-frame-v1", min_length=1)
    frame_evidence_window_ms: int = Field(default=2_000, ge=0)
    output_dir: Path = Path("artifacts/enrichment/transcripts")
    frames_path: Path = Path("artifacts/frame_store/frames.parquet")
    frame_enrichment_path: Path = Path(
        "artifacts/enrichment/asr/frame_enrichment.parquet"
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TranscriptJobConfig:
        """Load the transcript section from the shared enrichment YAML."""

        from hcmai.common.utils.io import read_yaml

        config_path = Path(path).expanduser().resolve()
        raw = read_yaml(config_path)
        transcript = raw.get("transcript") if isinstance(raw, dict) else None
        if not isinstance(transcript, dict):
            raise ValueError("Enrichment YAML requires a transcript mapping")
        config = cls.model_validate(transcript)
        project_root = Path(__file__).resolve().parents[3]
        return config.model_copy(update={
            "output_dir": _project_path(config.output_dir, project_root),
            "frames_path": _project_path(config.frames_path, project_root),
            "frame_enrichment_path": _project_path(
                config.frame_enrichment_path, project_root
            ),
        })


def _project_path(path: Path, project_root: Path) -> Path:
    """Resolve repository-owned configuration paths independently of cwd."""

    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else project_root / expanded


class IndexConfig(BaseModel):
    """Configuration for the self-contained FAISS artifact directory."""

    model_config = ConfigDict(extra="forbid")

    type: str = "flat_ip"
    path: Path = Path("artifacts/indexes/visual")
    context_path: Path = Path("artifacts/indexes/context")
    asr_segment_path: Path = Path("artifacts/indexes/asr_segments")
    caption_path: Path = Path("artifacts/indexes/caption")
    ocr_path: Path = Path("artifacts/indexes/ocr")
    asr_path: Path = Path("artifacts/indexes/asr")
    context_embedding_filename: str = "context_embeddings.npy"
    asr_segment_embedding_filename: str = "asr_embeddings.npy"
    asr_projection_max_gap_ms: int = Field(default=5_000, ge=0)
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
                    "visual, context, caption, ocr, and asr"
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
    """Transactional state, retrieval, and scene budgets for KIS."""

    model_config = ConfigDict(extra="forbid")

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
    # Keep the temporal candidate pool at the public search ceiling so a
    # request with top_k=100 is not silently capped before materialization.
    scene_top_p_global: int = Field(default=100, gt=0)
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
    api: ApiConfig = Field(default_factory=ApiConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        """Load configuration from a YAML file."""
        from hcmai.common.utils.io import read_yaml

        data = read_yaml(path)
        return cls.model_validate(data)


Settings = AppConfig
