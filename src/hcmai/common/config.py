"""Configuration scaffolding and contracts for the HCMAI 2026 search pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from math import isclose
from pathlib import Path
from typing import Any, Literal

from hcmai.retrieval.models import RetrievalSource
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


# Recall cut-offs frozen for baseline comparison
RECALL_CUTOFFS: tuple[int, ...] = (1, 5, 20, 50, 100)
DEFAULT_MAX_TEMPORAL_EVENT_COUNT = 32
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

    caption_path: Path | None = Path("artifacts/corpus/caption.parquet")
    ocr_path: Path | None = Path("artifacts/corpus/ocr_frames.parquet")
    object_path: Path | None = Path("artifacts/corpus/object_frames.parquet")
    asr_path: Path | None = Path("artifacts/enrichment/asr/frame_enrichment.parquet")
    context_path: Path | None = Path(
        "artifacts/enrichment/context_vi/frame_context_v1.parquet"
    )
    transcripts_path: Path | None = Path("artifacts/enrichment/transcripts")


class DatasetConfig(BaseModel):
    """Configuration for corpus metadata and enrichment paths."""

    version: str = "hcmai2026_v1"
    root: Path = Path("data")
    frames_path: Path = Path("artifacts/frame_store/frames.parquet")
    media_info_path: Path | None = Path("data/media-info")
    enrichment: EnrichmentArtifactsConfig = Field(default_factory=EnrichmentArtifactsConfig)



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
    frame_enrichment_path: Path = Path("artifacts/enrichment/asr/frame_enrichment.parquet")
    frame_store_id: str | None = None

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        dataset: Mapping[str, object] | None = None,
    ) -> TranscriptJobConfig:
        """Load transcript settings from the shared preparation YAML."""

        from hcmai.common.utils.io import read_yaml_section

        config_path = Path(path).expanduser().resolve()
        raw = read_yaml_section(config_path, "enrichment")
        configured_dataset = raw.get("dataset", {})
        if configured_dataset is None:
            configured_dataset = {}
        if not isinstance(configured_dataset, dict):
            raise ValueError("Enrichment YAML dataset section must be a mapping")
        dataset_values = dict(configured_dataset)
        if dataset:
            dataset_values.update(dict(dataset))
        transcript = raw.get("transcript")
        if not isinstance(transcript, dict):
            raise ValueError("Enrichment YAML requires a transcript mapping")
        transcript_values = dict(transcript)
        if "frames_path" not in transcript_values:
            frames_path = dataset_values.get("frames_path")
            if frames_path is not None:
                transcript_values["frames_path"] = frames_path
        if dataset_values.get("frame_store_id") is not None:
            transcript_values["frame_store_id"] = dataset_values["frame_store_id"]
        config = cls.model_validate(transcript_values)
        project_root = Path(__file__).resolve().parents[3]
        return config.model_copy(
            update={
                "output_dir": _project_path(config.output_dir, project_root),
                "frames_path": _project_path(config.frames_path, project_root),
                "frame_enrichment_path": _project_path(config.frame_enrichment_path, project_root),
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
    context_path: Path = Path("artifacts/indexes/context_vi")
    asr_segment_path: Path = Path("artifacts/indexes/asr_segments")
    caption_path: Path = Path("artifacts/indexes/caption")
    ocr_path: Path = Path("artifacts/indexes/ocr")
    bm25_path: Path = Path("artifacts/indexes/bm25")
    context_embedding_filename: str = "context_embeddings.npy"
    asr_segment_embedding_filename: str = "asr_embeddings.npy"
    asr_projection_max_gap_ms: int = Field(default=5_000, ge=0)
    text_embedding_filenames: dict[RetrievalSource, str] = Field(
        default_factory=lambda: {
            RetrievalSource.CAPTION: "caption_embeddings.npy",
            RetrievalSource.OCR: "ocr_embeddings.npy",
            RetrievalSource.ASR: "asr_embeddings.npy",
    })

    @field_validator("text_embedding_filenames")
    @classmethod
    def validate_text_embedding_filenames(
        cls,
        filenames: dict[RetrievalSource, str],
    ) -> dict[RetrievalSource, str]:
        """Require one safe NumPy artifact filename per text modality."""

        del cls
        if set(filenames) != set(TEXT_RETRIEVAL_SOURCES):
            raise ValueError("text_embedding_filenames must configure caption, ocr, and asr")
        for filename in filenames.values():
            path = Path(filename)
            if path.name != filename or path.suffix != ".npy":
                raise ValueError("text embedding filenames must be plain .npy filenames")
        return filenames


class FusionConfig(BaseModel):
    """Fusion configuration for multi-modal candidate merging."""

    method: Literal["rrf"] = "rrf"
    rrf_k: int = Field(default=60, gt=0)
    modality_max_workers: int = Field(default=4, ge=1)
    required_sources: set[RetrievalSource] = Field(default_factory=lambda: {RetrievalSource.VISUAL})
    normalize_active_weights: bool = True
    source_weights: dict[RetrievalSource, float] = Field(
        default_factory=lambda: {source: 1.0 for source in FUSION_SOURCES}
    )

    @model_validator(mode="after")
    def validate_source_weights(self) -> FusionConfig:
        """Require one explicit positive weight for every retrieval source."""

        expected = set(FUSION_SOURCES)
        if not self.required_sources.issubset(expected):
            raise ValueError("required_sources contains an unknown modality")
        if set(self.source_weights) != expected:
            raise ValueError(
                "fusion source_weights must configure visual, context, " "caption, ocr, and asr"
            )
        if any(weight <= 0 for weight in self.source_weights.values()):
            raise ValueError("fusion weights must be greater than zero")
        return self


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


class AlignmentConfig(BaseModel):
    """Task-agnostic numerical settings for ordered event alignment.

    The Phase A temporal baseline scores the full visual corpus for every
    event, so this config retains only the DP inputs still consumed by the
    shared scorer and decoder.
    """

    model_config = ConfigDict(extra="forbid")

    lambda_gap: float = Field(default=1e-5, ge=0.0)
    event_power: float = Field(default=1.0, gt=0.0, le=1.0)
    chunk_size: int = Field(default=65_536, ge=1)
    cluster_delta: float = Field(default=0.0, ge=0.0)


class DenseTemporalWeights(BaseModel):
    """Convex source weights for full-corpus Dense temporal evidence."""

    model_config = ConfigDict(extra="forbid")

    visual_weight: float = Field(default=1 / 3, ge=0)
    context_weight: float = Field(default=1 / 3, ge=0)
    asr_weight: float = Field(default=1 / 3, ge=0)

    @model_validator(mode="after")
    def validate_sum(self) -> DenseTemporalWeights:
        """Require normalized Dense source weights."""

        total = self.visual_weight + self.context_weight + self.asr_weight
        if not isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError("dense temporal weights must sum to 1.0")
        return self


class RobustCalibrationConfig(BaseModel):
    """Parameters for robust quantile-based score calibration and reliability."""

    model_config = ConfigDict(extra="forbid")

    q_low: float = Field(default=0.05, ge=0.0, lt=1.0)
    q_high: float = Field(default=0.95, gt=0.0, le=1.0)
    top_fraction: float = Field(default=0.01, gt=0.0, le=0.25)
    top_k_max: int = Field(default=128, ge=1)
    eps: float = Field(default=1e-6, gt=0.0)

    @model_validator(mode="after")
    def validate_quantiles(self) -> RobustCalibrationConfig:
        if self.q_low >= self.q_high:
            raise ValueError("q_low must be less than q_high")
        return self


class BM25FieldWeights(BaseModel):
    """Non-negative field weights for frame-native BM25 evidence."""

    model_config = ConfigDict(extra="forbid")

    title_weight: float = Field(default=1.0, ge=0)
    caption_weight: float = Field(default=1.0, ge=0)
    ocr_weight: float = Field(default=1.0, ge=0)
    asr_weight: float = Field(default=1.0, ge=0)


class AdaptiveTemporalFusionConfig(BaseModel):
    """Configuration for adaptive event-driven multimodal temporal evidence fusion."""

    model_config = ConfigDict(extra="forbid")

    calibration: RobustCalibrationConfig = Field(default_factory=RobustCalibrationConfig)
    robust_calibration: bool = True
    confidence_gating: bool = True
    event_routing: bool = True
    base_component_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "visual_dense": 0.35,
            "context_dense": 0.35,
            "asr_dense": 0.08,
            "bm25_title": 0.02,
            "bm25_caption": 0.10,
            "bm25_ocr": 0.04,
            "bm25_asr": 0.06,
        }
    )
    visual_boost: float = Field(default=1.4, ge=1.0)
    speech_boost: float = Field(default=5.0, ge=1.0)
    ocr_boost: float = Field(default=3.0, ge=1.0)

    @model_validator(mode="after")
    def validate_component_weights(self) -> AdaptiveTemporalFusionConfig:
        if any(weight < 0.0 for weight in self.base_component_weights.values()):
            raise ValueError("adaptive component weights must be non-negative")
        if sum(self.base_component_weights.values()) <= 0.0:
            raise ValueError("adaptive component weights must contain positive mass")
        return self


class HybridTemporalConfig(BaseModel):
    """Full-corpus Dense/BM25 temporal evidence configuration."""

    model_config = ConfigDict(extra="forbid")

    dense: DenseTemporalWeights = Field(default_factory=DenseTemporalWeights)
    bm25_fields: BM25FieldWeights = Field(default_factory=BM25FieldWeights)
    dense_weight: float = Field(default=0.5, ge=0)
    bm25_weight: float = Field(default=0.5, ge=0)
    fusion_mode: Literal["legacy", "adaptive_p0"] = "legacy"
    adaptive: AdaptiveTemporalFusionConfig = Field(default_factory=AdaptiveTemporalFusionConfig)

    @model_validator(mode="after")
    def validate_sum(self) -> HybridTemporalConfig:
        """Require normalized hybrid source weights."""

        if not isclose(self.dense_weight + self.bm25_weight, 1.0, abs_tol=1e-6):
            raise ValueError("hybrid temporal weights must sum to 1.0")
        return self


class SearchConfig(BaseModel):
    """Single search configuration selected for the competition pipeline."""

    model_config = ConfigDict(extra="forbid")

    max_temporal_event_count: int = Field(
        default=DEFAULT_MAX_TEMPORAL_EVENT_COUNT,
        ge=1,
        le=DEFAULT_MAX_TEMPORAL_EVENT_COUNT,
    )
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    cache: RetrievalCacheConfig = Field(default_factory=RetrievalCacheConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    hybrid_temporal: HybridTemporalConfig = Field(default_factory=HybridTemporalConfig)



class ApiConfig(BaseModel):
    """Configuration for API parameters."""

    default_top_k: int = 20
    maximum_top_k: int = 100
    image_max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    image_max_pixels: int = Field(default=40_000_000, gt=0)


class InferenceConfig(BaseModel):
    """Connection from the local data plane to the private GPU service."""

    # The shared gateway is the normal execution path. Keep this compatibility
    # field for older consumers, but omission must never silently disable it.
    enabled: bool = True
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


class QueryPreparationConfig(BaseModel):
    """Configuration for stateless Qwen query preparation."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = "Qwen/Qwen3-4B"
    model_revision: str = Field(
        default="1cfa9a7208912126459214e8b04321603b3df60c",
        min_length=40,
        max_length=40,
        pattern=r"^[0-9a-f]{40}$",
    )
    prompt_version: str = Field(default="query-prep-v1", min_length=1)
    candidate_count: Literal[5] = 5
    cache_enabled: bool = True
    cache_ttl_seconds: float = Field(default=3600, gt=0)
    cache_max_entries: int = Field(default=2048, ge=1)


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
    query_preparation: QueryPreparationConfig = Field(default_factory=QueryPreparationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        """Load configuration from a YAML file."""

        from hcmai.common.utils.io import read_yaml

        data = read_yaml(path)
        return cls.model_validate(data)


Settings = AppConfig
