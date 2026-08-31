"""Configuration scaffolding and contracts for the HCMAI 2026 search pipeline."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from typing import Any, Literal
from pydantic import ConfigDict, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from hcmai.retrieval.models import RetrievalSource

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
        "artifacts/enrichment/captions/captions.parquet"
    )
    ocr_path: Path | None = Path(
        "artifacts/enrichment/ocr/frames.parquet"
    )
    object_path: Path | None = Path("artifacts/enrichment/objects/frames.parquet")
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
    media_info_path: Path | None = Path("data/media-info-aic25-b1/media-info")
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
                "fusion source_weights must configure visual, context, "
                "caption, ocr, and asr"
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


class SearchConfig(BaseModel):
    """Single search configuration selected for the competition pipeline."""

    model_config = ConfigDict(extra="forbid")

    fusion: FusionConfig = Field(default_factory=FusionConfig)
    cache: RetrievalCacheConfig = Field(default_factory=RetrievalCacheConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)


class ApiConfig(BaseModel):
    """Configuration for API parameters."""

    default_top_k: int = 20
    maximum_top_k: int = 100


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
