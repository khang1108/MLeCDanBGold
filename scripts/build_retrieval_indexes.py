"""Build and validate the fast-track multimodal retrieval indexes offline.

This command owns strict input preflight, sequential Visual/Context/ASR index
construction, and final bundle validation. It never extracts keyframes,
rewrites canonical frame metadata, or rebuilds indexes from an online request.
"""

from __future__ import annotations

import argparse
import gc
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


LOGGER = logging.getLogger("hcmai.offline_indexes")
STAGES = ("preflight", "visual", "context", "asr", "all", "validate")


class DatasetInputs(BaseModel):
    """Canonical and derived offline inputs with expected corpus coverage."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    frames_path: Path
    frame_manifest: Path
    keyframes_root: Path
    map_keyframes_root: Path
    context_path: Path
    transcripts_path: Path
    expected_video_count: int = Field(gt=0)
    expected_frame_count: int = Field(gt=0)


class BuildOptions(BaseModel):
    """Visual build strictness and deterministic checkpoint sizing."""

    model_config = ConfigDict(extra="forbid")

    strict_visual_coverage: bool = True
    resume: bool = True
    visual_shard_size: int = Field(default=2_048, gt=0)


class IndexOutputs(BaseModel):
    """Published retrieval bundle destinations."""

    model_config = ConfigDict(extra="forbid")

    visual: Path
    context: Path
    asr_segments: Path

    @model_validator(mode="after")
    def _require_common_root(self) -> "IndexOutputs":
        parents = {path.parent for path in (self.visual, self.context, self.asr_segments)}
        if len(parents) != 1:
            raise ValueError("visual, context, and asr_segments must share one output root")
        if len({path.name for path in (self.visual, self.context, self.asr_segments)}) != 3:
            raise ValueError("index output directories must be distinct")
        return self


class ProjectionOptions(BaseModel):
    """Timeline projection settings recorded with the offline workflow."""

    model_config = ConfigDict(extra="forbid")

    max_projection_gap_ms: int = Field(default=5_000, ge=0)


class OfflineIndexConfig(BaseModel):
    """Validated shape of the dedicated ``configs/indexing.yaml`` file."""

    model_config = ConfigDict(extra="forbid")

    dataset: DatasetInputs
    build: BuildOptions
    indexes: IndexOutputs
    projection: ProjectionOptions
    config_path: Path = Field(exclude=True)
    model_config_path: Path = Field(exclude=True)

    @property
    def output_root(self) -> Path:
        """Return the common parent used for indexes and the build report."""

        return self.indexes.visual.parent

    @property
    def projected_frames_path(self) -> Path:
        """Return the indexing-only portable frame projection path."""

        return self.output_root / ".indexing" / "projected_frames.parquet"

    @property
    def report_path(self) -> Path:
        """Return the report committed only after clean final validation."""

        return self.output_root / "build_report.json"


@dataclass(frozen=True)
class PreflightResult:
    """Validated input counts and the portable visual frame projection."""

    projected: pd.DataFrame
    mapping_video_count: int
    mapping_row_count: int
    canonical_frame_count: int
    context_frame_count: int
    transcript_segment_count: int
    duplicate_submission_rows: int

    def report(self) -> dict[str, int]:
        """Return JSON-safe corpus counts for diagnostics."""

        return {
            "mapping_video_count": self.mapping_video_count,
            "mapping_row_count": self.mapping_row_count,
            "canonical_frame_count": self.canonical_frame_count,
            "context_frame_count": self.context_frame_count,
            "transcript_segment_count": self.transcript_segment_count,
            "duplicate_submission_rows": self.duplicate_submission_rows,
        }


@dataclass
class RemoteEmbeddingAdapters:
    """One checked private inference client and the adapters needed by a stage.

    The service is retained solely for readiness and cleanup.  Adapters receive
    its HTTP client so the image adapter can use ``embed_images`` directly.
    Neither this object nor any index metadata persists the inference URL.
    """

    service: Any
    visual: Any | None = None
    text: Any | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the explicit offline stage and optional path overrides."""

    parser = argparse.ArgumentParser(
        description="Build and validate Visual + FrameContext + segment-ASR indexes"
    )
    parser.add_argument("--config", default="configs/indexing.yaml")
    parser.add_argument("--model-config", default="configs/indexing.models.yaml")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--frames")
    parser.add_argument("--keyframes-root")
    parser.add_argument("--map-keyframes-root")
    parser.add_argument("--context")
    parser.add_argument("--transcripts")
    parser.add_argument("--output-root")
    parser.add_argument(
        "--inference-url",
        help=(
            "Use this explicit private inference endpoint for remote SigLIP and "
            "BGE embeddings. The offline builder never reads an inference URL "
            "from the environment."
        ),
    )
    parser.add_argument(
        "--s3",
        action="store_true",
        help=(
            "Download required inputs from the configured S3 bucket, build "
            "locally, and publish the validated bundle back to S3"
        ),
    )
    parser.add_argument(
        "--s3-config",
        default="configs/preparation.s3.yaml",
        help="YAML containing preprocessing.s3 credentials/region settings",
    )
    parser.add_argument("--s3-keyframes-prefix", default="data/keyframes")
    parser.add_argument(
        "--s3-map-keyframes-prefix",
        default="data/features/map-keyframes",
    )
    parser.add_argument(
        "--s3-frame-store-prefix",
        default="data/artifacts/frame_store",
    )
    parser.add_argument(
        "--s3-context-prefix",
        default="data/artifacts/enrichment/context",
    )
    parser.add_argument(
        "--s3-transcripts-prefix",
        default="data/artifacts/enrichment/transcripts",
    )
    parser.add_argument("--s3-output-prefix", default="data/artifacts/indexes")
    parser.add_argument(
        "--s3-sync-workers",
        type=int,
        default=16,
        help="Concurrent S3 input transfers (default: 16)",
    )
    parser.add_argument(
        "--s3-upload-workers",
        type=int,
        default=8,
        help="Concurrent validated-index uploads (default: 8)",
    )
    parser.add_argument(
        "--s3-dry-run",
        action="store_true",
        help="List required S3 inputs without downloading or building",
    )
    return parser.parse_args(argv)


def load_offline_config(
    config_path: str | Path,
    model_config_path: str | Path,
    *,
    frames: str | Path | None = None,
    keyframes_root: str | Path | None = None,
    map_keyframes_root: str | Path | None = None,
    context: str | Path | None = None,
    transcripts: str | Path | None = None,
    output_root: str | Path | None = None,
) -> OfflineIndexConfig:
    """Load the dedicated indexing mapping without coercing it to AppConfig."""

    from hcmai.common.utils.io import read_yaml

    config_file = Path(config_path)
    model_file = Path(model_config_path)
    raw = read_yaml(config_file)
    if not isinstance(raw, dict):
        raise ValueError(f"Indexing config must contain a mapping: {config_file}")
    config = OfflineIndexConfig.model_validate(
        {**raw, "config_path": config_file, "model_config_path": model_file}
    )

    dataset_updates: dict[str, Path] = {}
    if frames is not None:
        frames_path = Path(frames)
        dataset_updates.update(
            frames_path=frames_path,
            frame_manifest=frames_path.with_name("manifest.json"),
        )
    for name, value in (
        ("keyframes_root", keyframes_root),
        ("map_keyframes_root", map_keyframes_root),
        ("context_path", context),
        ("transcripts_path", transcripts),
    ):
        if value is not None:
            dataset_updates[name] = Path(value)
    if dataset_updates:
        config = config.model_copy(
            update={"dataset": config.dataset.model_copy(update=dataset_updates)}
        )

    if output_root is not None:
        root = Path(output_root)
        config = config.model_copy(
            update={
                "indexes": IndexOutputs(
                    visual=root / "visual",
                    context=root / "context",
                    asr_segments=root / "asr_segments",
                )
            }
        )
    return config


def load_model_config(path: str | Path) -> Any:
    """Load the pinned Visual and shared evidence encoder configuration."""

    from hcmai.llm.config import LLMServiceConfig

    config = LLMServiceConfig.from_yaml(path)
    for label, encoder in (
        ("visual_embedding", config.visual_embedding),
        ("evidence_embedding", config.resolved_evidence_embedding),
    ):
        if not isinstance(encoder.revision, str) or re.fullmatch(
            r"[0-9a-fA-F]{40}", encoder.revision
        ) is None:
            raise ValueError(
                f"{label}.revision must be a 40-character hexadecimal commit"
            )
    return config


def _load_s3_transport(config_path: str | Path) -> tuple[Any, str]:
    """Create an S3 client from the repository's standard credential config.

    The repository's production preparation config wraps this section below a
    top-level ``preparation`` key, so use its validated loader to normalize
    environment overrides and nested S3 settings.  The retrieval builder does
    not run the corpus-preparation stages themselves. Credentials continue to
    come from boto3's standard chain or the existing ``HCMAI_S3_*`` overrides.
    """

    from hcmai.data.corpus_build.config import S3CorpusPreparationConfig
    from hcmai.data.s3 import create_s3_client

    preparation = S3CorpusPreparationConfig.from_yaml(config_path)
    storage = preparation.preprocessing.s3
    if storage is None:
        raise ValueError(f"S3 settings are missing from {config_path}")
    return create_s3_client(storage), storage.bucket


def _s3_module() -> Any:
    """Import the S3 transfer helper in both CLI and package-test contexts."""

    try:
        from scripts import retrieval_s3
    except ModuleNotFoundError:
        # When this file is executed directly, Python places ``scripts/`` on
        # sys.path rather than the repository root.
        import retrieval_s3

        return retrieval_s3
    return retrieval_s3


def _download_s3_inputs(
    client: Any,
    bucket: str,
    config: OfflineIndexConfig,
    args: argparse.Namespace,
) -> None:
    """Stage only the inputs required by the fast-track index builder.

    Raw videos and unrelated enrichment/index directories are intentionally
    excluded.  Each prefix is resumable and writes atomically, which permits a
    stopped ThunderCompute job to be restarted without deleting its cache.
    """

    transfer = _s3_module()
    inputs = (
        (
            "BTC keyframes",
            args.s3_keyframes_prefix,
            config.dataset.keyframes_root,
        ),
        (
            "BTC map_keyframes",
            args.s3_map_keyframes_prefix,
            config.dataset.map_keyframes_root,
        ),
        (
            "canonical FrameStore",
            args.s3_frame_store_prefix,
            config.dataset.frames_path.parent,
        ),
        (
            "FrameContext",
            args.s3_context_prefix,
            config.dataset.context_path.parent,
        ),
        (
            "timestamped transcripts",
            args.s3_transcripts_prefix,
            config.dataset.transcripts_path,
        ),
    )
    for label, prefix, destination in inputs:
        stats = transfer.download_prefix(
            client,
            bucket,
            prefix,
            destination,
            workers=args.s3_sync_workers,
            dry_run=args.s3_dry_run,
        )
        LOGGER.info(
            "S3 input ready label=%s prefix=%s destination=%s files=%d downloaded=%d skipped=%d",
            label,
            stats.prefix,
            Path(destination).expanduser().resolve(),
            stats.files,
            stats.downloaded,
            stats.skipped,
        )


def _publish_s3_bundle(
    client: Any,
    bucket: str,
    config: OfflineIndexConfig,
    args: argparse.Namespace,
) -> None:
    """Publish a passed local bundle and advance its S3 latest pointer."""

    transfer = _s3_module()
    publication = transfer.publish_retrieval_bundle(
        client,
        bucket,
        config.output_root,
        args.s3_output_prefix,
        workers=args.s3_upload_workers,
    )
    LOGGER.info(
        "S3 retrieval bundle ready: s3://%s/%s (latest=s3://%s/%s)",
        publication.bucket,
        publication.version_prefix,
        publication.bucket,
        publication.latest_key,
    )


def _close_s3_transport(client: Any | None) -> None:
    """Close a boto3 client when the installed botocore version supports it."""

    if client is None:
        return
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _remote_stage_requirements(stage: str) -> tuple[bool, bool]:
    """Return whether one requested stage needs Visual and/or BGE embeddings."""

    return stage in {"visual", "all"}, stage in {"context", "asr", "all"}


def _check_remote_encoder_status(
    statuses: dict[str, Any],
    status_name: str,
    encoder: Any,
) -> list[str]:
    """Describe incompatibilities between one hosted model and a pinned encoder."""

    status = statuses.get(status_name)
    if status is None:
        return [f"{status_name} is not advertised"]
    if not status.enabled:
        return [f"{status_name} is disabled"]
    if not status.loaded:
        return [f"{status_name} is not loaded"]
    if status.checkpoint != encoder.model_name:
        return [f"{status_name} checkpoint does not match the pinned model"]
    if encoder.revision is not None and status.revision != encoder.revision:
        return [f"{status_name} revision does not match the pinned model"]
    return []


def _require_remote_embedding_readiness(
    service: Any,
    models: Any,
    *,
    require_visual: bool,
    require_text: bool,
) -> None:
    """Verify remote SigLIP/BGE compatibility before any offline build work.

    Index metadata is stamped from the pinned local model configuration, so a
    reachable but differently-versioned hosted model would make the published
    provenance false.  Check the advertised checkpoint and revision before
    preflight can spend time reading the corpus.
    """

    try:
        readiness = service.readiness()
    except Exception as error:
        raise RuntimeError(
            "Remote embedding readiness check failed; ensure the private "
            "SigLIP+BGE service is reachable and ready"
        ) from error

    failures: list[str] = []
    if not readiness.ready:
        failures.append("service reported not ready")

    statuses = readiness.models
    capabilities = readiness.capabilities
    if require_visual:
        failures.extend(
            _check_remote_encoder_status(
                statuses,
                "visual_embedding",
                models.visual_embedding,
            )
        )
        if not capabilities.image_embedding:
            failures.append("visual image-embedding capability is unavailable")
    if require_text:
        failures.extend(
            _check_remote_encoder_status(
                statuses,
                "caption_embedding",
                models.resolved_evidence_embedding,
            )
        )
        if not capabilities.embedding:
            failures.append("BGE text-embedding capability is unavailable")

    if failures:
        raise RuntimeError(
            "Remote embedding service is incompatible with the requested "
            "offline stage: "
            + "; ".join(failures)
        )


def create_remote_embedding_adapters(
    inference_url: str,
    models: Any,
    *,
    require_visual: bool,
    require_text: bool,
) -> RemoteEmbeddingAdapters:
    """Create checked remote adapters without deriving an endpoint from env vars."""

    from hcmai.common.config import InferenceConfig
    from hcmai.llm.pipeline import LLMService
    from hcmai.retrieval.embedding.pipeline import EmbeddingService

    base_url = inference_url.strip()
    if not base_url:
        raise ValueError("--inference-url must not be empty")

    inference = InferenceConfig(enabled=True, base_url=base_url)
    service = LLMService.remote(base_url, inference)
    try:
        _require_remote_embedding_readiness(
            service,
            models,
            require_visual=require_visual,
            require_text=require_text,
        )
        client = service.adapter
        visual = (
            EmbeddingService.create_remote_visual_adapter(
                client,
                models.visual_embedding,
            )
            if require_visual
            else None
        )
        text = (
            EmbeddingService.create_remote_adapter(
                client,
                models.resolved_evidence_embedding,
                embedding_dim=0,
                source="text",
            )
            if require_text
            else None
        )
        return RemoteEmbeddingAdapters(service=service, visual=visual, text=text)
    except Exception:
        service.close()
        raise


def _require_file(path: Path, label: str) -> Path:
    """Require one non-empty offline input file before any model is loaded."""

    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} is not available at {path}")
    return path


def _require_directory(path: Path, label: str) -> Path:
    """Require one existing offline input directory."""

    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory is not available at {path}")
    return path


def project_staged_keyframes(
    frames: pd.DataFrame,
    keyframes_root: str | Path,
) -> pd.DataFrame:
    """Project config-relative keyframes to absolute, root-contained paths.

    ``EmbeddingArtifactBuilder`` accepts both absolute paths and paths relative
    to its dataset root. Publishing absolute paths here prevents a relative
    configured root such as ``data/keyframes`` from being prepended twice.
    """

    from hcmai.data.ingestion.keyframe_map import project_keyframe_paths

    root = Path(keyframes_root).expanduser().resolve()
    projected = project_keyframe_paths(frames, root)
    for value in projected["image_path"]:
        image = Path(str(value)).resolve()
        if not image.is_file() or not image.is_relative_to(root):
            raise ValueError(f"Projected keyframe is outside the staged root: {image}")
    return projected


def _require_usable_context_ids(data: Any) -> set[str]:
    """Return frame IDs whose deterministic Context text can be indexed."""

    usable = {
        context.frame_id
        for context in data.iter_frame_contexts()
        if data.get_frame_context_text(context.frame_id) is not None
    }
    if not usable:
        raise ValueError("FrameContext artifact contains no usable context_text")
    return usable


def _require_usable_completed_segments(segments: tuple[Any, ...]) -> list[Any]:
    """Return ASR segments accepted by the segment corpus builder."""

    from hcmai.common.schemas import ProcessingStatus

    usable = [
        segment
        for segment in segments
        if segment.status is ProcessingStatus.COMPLETED
        and bool(" ".join(segment.text.split()))
    ]
    if not usable:
        raise ValueError("Transcript artifact contains no usable completed segments")
    return usable


def _inspect_inputs(config: OfflineIndexConfig) -> PreflightResult:
    """Validate canonical identity, mapping, Context lineage, and transcripts."""

    from hcmai.common.utils.io import read_json
    from hcmai.data.ingestion.keyframe_map import (
        join_btc_mapping,
        load_btc_keyframe_map,
    )
    from hcmai.data.pipeline import DataService
    from hcmai.data.enrichment.transcripts.store import TranscriptStore

    dataset = config.dataset
    frames_path = _require_file(dataset.frames_path, "Canonical frames")
    manifest_path = _require_file(dataset.frame_manifest, "Canonical frame manifest")
    keyframes_root = _require_directory(dataset.keyframes_root, "BTC keyframes")
    mapping_root = _require_directory(dataset.map_keyframes_root, "BTC map_keyframes")
    context_path = _require_file(dataset.context_path, "FrameContext")
    _require_file(context_path.with_name("manifest.json"), "FrameContext manifest")
    transcripts_path = dataset.transcripts_path
    if not transcripts_path.exists():
        raise FileNotFoundError(
            f"Transcript artifact is not available at {transcripts_path}"
        )

    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("Canonical frame manifest must contain an object")
    frames = cast(pd.DataFrame, pd.read_parquet(frames_path))
    required_columns = {
        "frame_id",
        "video_id",
        "frame_idx",
        "timestamp_ms",
        "fps",
        "keyframe_order",
        "image_path",
    }
    missing = sorted(required_columns.difference(frames.columns))
    if missing:
        raise ValueError("Canonical frames are missing columns: " + ", ".join(missing))
    if len(frames) != dataset.expected_frame_count:
        raise ValueError(
            f"Canonical frame count {len(frames)} != {dataset.expected_frame_count}"
        )
    frame_ids = cast(pd.Series, frames["frame_id"])
    if bool(frame_ids.isna().any()) or bool(frame_ids.duplicated().any()):
        raise ValueError("Canonical frame_id values must be present and unique")
    video_count = int(cast(Any, frames["video_id"]).nunique())
    if video_count != dataset.expected_video_count:
        raise ValueError(
            f"Canonical video count {video_count} != {dataset.expected_video_count}"
        )
    if manifest.get("frame_count") not in (None, len(frames)):
        raise ValueError("Canonical manifest frame_count does not match frames.parquet")
    if manifest.get("video_count") not in (None, video_count):
        raise ValueError("Canonical manifest video_count does not match frames.parquet")

    mapping = load_btc_keyframe_map(mapping_root)
    mapping_video_count = int(cast(Any, mapping["video_id"]).nunique())
    if mapping_video_count != dataset.expected_video_count:
        raise ValueError(
            f"BTC mapping video count {mapping_video_count} != {dataset.expected_video_count}"
        )
    if len(mapping) != dataset.expected_frame_count:
        raise ValueError(
            f"BTC mapping row count {len(mapping)} != {dataset.expected_frame_count}"
        )
    frame_keys = set(zip(frames["video_id"], frames["keyframe_order"], strict=True))
    mapping_keys = set(
        zip(mapping["video_id"], mapping["keyframe_order"], strict=True)
    )
    if frame_keys != mapping_keys:
        raise ValueError("Canonical frames and BTC mapping keys do not join completely")
    mapping_source = cast(
        pd.DataFrame,
        frames[["frame_id", "video_id", "keyframe_order"]],
    )
    mapped = join_btc_mapping(mapping_source, mapping)
    comparison = frames[
        ["frame_id", "frame_idx", "timestamp_ms", "fps"]
    ].merge(
        mapped[["frame_id", "frame_idx", "timestamp_ms", "fps"]],
        on="frame_id",
        suffixes=("_canonical", "_btc"),
        validate="one_to_one",
    )
    coordinate_pairs = (
        ("frame_idx", np.int64),
        ("timestamp_ms", np.int64),
        ("fps", float),
    )
    if any(
        not np.array_equal(
            comparison[f"{name}_canonical"].to_numpy(dtype=dtype),
            comparison[f"{name}_btc"].to_numpy(dtype=dtype),
        )
        for name, dtype in coordinate_pairs
    ):
        raise ValueError("Canonical frame coordinates do not match BTC mapping")

    projected = project_staged_keyframes(frames, keyframes_root)

    data = DataService.load(frames_path, context_path=context_path)
    canonical_ids = set(frames["frame_id"].astype(str))
    context_ids = {
        context.frame_id for context in data.iter_frame_contexts()
    }
    if not context_ids.issubset(canonical_ids):
        raise ValueError("FrameContext contains non-canonical frame_id values")
    _require_usable_context_ids(data)

    transcripts = TranscriptStore(transcripts_path)
    segments = tuple(transcripts.iter_records())
    if not segments:
        raise ValueError("TranscriptStore contains no segments")
    if any(segment.end_ms <= segment.start_ms for segment in segments):
        raise ValueError("Transcript segments must have positive durations")
    segment_ids = [segment.segment_id for segment in segments]
    if len(set(segment_ids)) != len(segment_ids):
        raise ValueError("Transcript segment_id values must be unique")
    _require_usable_completed_segments(segments)

    duplicate_rows = int(
        frames.duplicated(["video_id", "frame_idx"], keep=False).sum()
    )
    return PreflightResult(
        projected=projected,
        mapping_video_count=mapping_video_count,
        mapping_row_count=len(mapping),
        canonical_frame_count=len(frames),
        context_frame_count=len(context_ids),
        transcript_segment_count=len(segments),
        duplicate_submission_rows=duplicate_rows,
    )


def run_preflight(config: OfflineIndexConfig) -> Path:
    """Validate all build inputs and write only an indexing path projection."""

    from hcmai.common.utils.io import atomic_write, write_parquet

    result = _inspect_inputs(config)
    path = config.projected_frames_path
    atomic_write(
        path,
        lambda staged: write_parquet(result.projected, staged, index=False),
    )
    LOGGER.info("Preflight passed: %s", result.report())
    return path


def _require_projected_frames(config: OfflineIndexConfig) -> Path:
    """Require the indexing-only frame projection produced by preflight."""

    return _require_file(
        config.projected_frames_path,
        "Projected frames (run --stage preflight first)",
    )


def _config_fingerprint(config: OfflineIndexConfig) -> str:
    """Fingerprint both immutable inputs that define the index build."""

    from hcmai.retrieval.retriever.artifacts import fingerprint_files

    return fingerprint_files([config.config_path, config.model_config_path])


def _visual_source_fingerprint(config: OfflineIndexConfig) -> str:
    """Fingerprint canonical identity, manifest lineage, and organizer mapping."""

    from hcmai.retrieval.retriever.artifacts import fingerprint_files

    mapping_files = sorted(config.dataset.map_keyframes_root.glob("*.csv"))
    return fingerprint_files(
        [
            config.dataset.frames_path,
            config.dataset.frame_manifest,
            *mapping_files,
        ]
    )


def build_visual(
    config: OfflineIndexConfig,
    models: Any,
    projected_frames: str | Path,
    *,
    encoder: Any | None = None,
) -> Any:
    """Build strict SigLIP2 vectors with a local or injected remote encoder."""

    from hcmai.retrieval.embedding.artifacts import EmbeddingArtifactBuilder
    from hcmai.retrieval.retriever.dense.index import DenseIndex

    output = config.indexes.visual.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(mkdtemp(prefix=".visual-build-", dir=output.parent))
    builder: EmbeddingArtifactBuilder | None = None
    try:
        builder = EmbeddingArtifactBuilder(
            frames_path=Path(projected_frames),
            dataset_root=config.dataset.keyframes_root,
            output_dir=workspace,
            encoder_config=models.visual_embedding,
            dataset_version=config.dataset.version,
            strict=config.build.strict_visual_coverage,
            resume=config.build.resume,
            shard_size=config.build.visual_shard_size,
            checkpoint_dir=output.parent / ".visual-checkpoints",
            encoder=encoder,
        )
        metadata = builder.run()
        if metadata.successful_frames != config.dataset.expected_frame_count:
            raise RuntimeError(
                "Visual build does not match strict expected frame coverage"
            )
        vectors = np.load(builder.embeddings_file, mmap_mode="r")
        mapping = pd.read_parquet(builder.mapping_file)
        index = DenseIndex.build(
            vectors,
            mapping,
            dataset_version=config.dataset.version,
            model_name=models.visual_embedding.model_name,
            index_type="flat_ip",
            show_progress=True,
        )
        index.metadata.entity_kind = "frame"
        index.metadata.retrieval_source = "visual"
        index.metadata.model_revision = models.visual_embedding.revision
        index.metadata.source_fingerprint = _visual_source_fingerprint(config)
        index.metadata.config_fingerprint = _config_fingerprint(config)
        index.save(output)
        return DenseIndex.load(output)
    finally:
        if builder is not None:
            del builder
        shutil.rmtree(workspace, ignore_errors=True)


def create_text_encoder(models: Any) -> Any:
    """Load the shared BGE evidence adapter once for Context followed by ASR."""

    from hcmai.retrieval.embedding.pipeline import EmbeddingService

    return EmbeddingService.create_text_adapter(models.resolved_evidence_embedding)


def _stamp_config_fingerprint(
    index: Any,
    output_dir: str | Path,
    loader: Any,
    config_fingerprint: str,
) -> Any:
    """Atomically add workflow provenance without dropping supplemental vectors.

    Context and ASR low-level builders publish their supplemental embedding
    files alongside the core index. Re-saving only the core index would discard
    those files, so metadata is committed through its own atomic replacement
    and immediately checksum-loaded again.
    """

    from hcmai.common.utils.io import atomic_write, write_json

    index.metadata.config_fingerprint = config_fingerprint
    metadata_path = Path(output_dir) / "metadata.json"
    atomic_write(
        metadata_path,
        lambda staged: write_json(index.metadata.to_dict(), staged),
    )
    return loader.load(output_dir)


def build_context(
    config: OfflineIndexConfig,
    models: Any,
    projected_frames: str | Path,
    *,
    encoder: Any | None = None,
) -> Any:
    """Build FrameContext directly from its typed store and shared BGE encoder."""

    from hcmai.data.pipeline import DataService
    from hcmai.retrieval.retriever.artifacts import fingerprint_files
    from hcmai.retrieval.retriever.dense.index import DenseIndex
    from hcmai.retrieval.retriever.text.retriever import build_context_index

    selected = encoder or create_text_encoder(models)
    data = DataService.load(projected_frames, context_path=config.dataset.context_path)
    index = build_context_index(
        data,
        selected,
        config.indexes.context,
        embeddings_filename="context_embeddings.npy",
        dataset_version=config.dataset.version,
        index_type="flat_ip",
        source_fingerprint=fingerprint_files(
            [
                config.dataset.context_path,
                config.dataset.context_path.with_name("manifest.json"),
            ]
        ),
    )
    return _stamp_config_fingerprint(
        index,
        config.indexes.context,
        DenseIndex,
        _config_fingerprint(config),
    )


def build_asr(
    config: OfflineIndexConfig,
    models: Any,
    *,
    encoder: Any | None = None,
) -> Any:
    """Build the segment-native ASR index without inventing frame identity."""

    from hcmai.data.enrichment.transcripts.store import TranscriptStore
    from hcmai.retrieval.retriever.artifacts import fingerprint_files
    from hcmai.retrieval.retriever.segment.artifacts import (
        build_asr_segment_index,
        transcript_lineage_files,
    )
    from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex

    selected = encoder or create_text_encoder(models)
    lineage_files = transcript_lineage_files(config.dataset.transcripts_path)
    if not lineage_files:
        raise ValueError("Transcript artifact contains no lineage files")
    index = build_asr_segment_index(
        TranscriptStore(config.dataset.transcripts_path),
        selected,
        config.indexes.asr_segments,
        embeddings_filename="asr_embeddings.npy",
        dataset_version=config.dataset.version,
        index_type="flat_ip",
        source_fingerprint=fingerprint_files(lineage_files),
    )
    return _stamp_config_fingerprint(
        index,
        config.indexes.asr_segments,
        SegmentDenseIndex,
        _config_fingerprint(config),
    )


def release_gpu_memory() -> None:
    """Release the Visual model before the shared BGE text stages begin."""

    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _validate_index_metadata(
    index: Any,
    encoder: Any,
    label: str,
    *,
    retrieval_source: str,
    entity_kind: str,
    source_fingerprint: str,
    config_fingerprint: str,
) -> None:
    """Check one loaded v2 bundle against its configured encoder contract."""

    metadata = index.metadata
    if metadata.schema_version != "dense-index-v2":
        raise ValueError(f"{label} is not a v2 index bundle")
    if not metadata.checksums:
        raise ValueError(f"{label} does not contain checksum provenance")
    if metadata.model_name != encoder.model_name:
        raise ValueError(f"{label} model does not match configured encoder")
    if metadata.model_revision != encoder.revision:
        raise ValueError(f"{label} revision does not match configured encoder")
    if metadata.normalization != "l2" or metadata.embedding_dim <= 0:
        raise ValueError(f"{label} embedding contract is invalid")
    if metadata.entity_kind != entity_kind:
        raise ValueError(f"{label} entity kind does not match its index contract")
    if metadata.retrieval_source != retrieval_source:
        raise ValueError(f"{label} retrieval source does not match its index contract")
    if metadata.source_fingerprint != source_fingerprint:
        raise ValueError(f"{label} source fingerprint does not match current inputs")
    if metadata.config_fingerprint != config_fingerprint:
        raise ValueError(f"{label} config fingerprint does not match current inputs")


def _bundle_size(path: Path) -> int:
    """Return the byte size of regular files in one published bundle."""

    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _validate_projected_frames(
    expected: pd.DataFrame, path: Path
) -> None:
    """Require the persisted projection to match freshly inspected inputs."""

    actual = pd.read_parquet(_require_file(path, "Projected frames"))
    columns = [
        "frame_id",
        "video_id",
        "frame_idx",
        "timestamp_ms",
        "keyframe_order",
        "image_path",
    ]
    if not actual[columns].reset_index(drop=True).equals(
        expected[columns].reset_index(drop=True)
    ):
        raise ValueError("Persisted projected frames are stale or misaligned")


def _validate_frame_identity(
    mapping: pd.DataFrame,
    canonical: pd.DataFrame,
    label: str,
) -> None:
    """Require indexed frame IDs to retain their canonical public coordinates."""

    columns = ["frame_id", "video_id", "frame_idx", "timestamp_ms"]
    paired = mapping[columns].merge(
        canonical[columns],
        on="frame_id",
        how="left",
        suffixes=("_index", "_canonical"),
        validate="one_to_one",
    )
    if bool(paired["video_id_canonical"].isna().any()):
        raise ValueError(f"{label} mapping contains unknown canonical frame_id values")
    for name, dtype in (
        ("video_id", str),
        ("frame_idx", np.int64),
        ("timestamp_ms", np.int64),
    ):
        if not np.array_equal(
            paired[f"{name}_index"].to_numpy(dtype=dtype),
            paired[f"{name}_canonical"].to_numpy(dtype=dtype),
        ):
            raise ValueError(f"{label} mapping changed canonical {name}")


def _validate_segment_identity(index: Any, transcripts_path: Path) -> None:
    """Match every indexed segment to its typed transcript timeline identity."""

    from hcmai.common.schemas import ProcessingStatus
    from hcmai.data.enrichment.transcripts.store import TranscriptStore

    records = {
        segment.segment_id: segment
        for segment in TranscriptStore(transcripts_path).iter_records()
        if segment.status is ProcessingStatus.COMPLETED and segment.text.strip()
    }
    indexed_ids = set(index.mapping["segment_id"].astype(str))
    if indexed_ids != set(records):
        raise ValueError("ASR index segment coverage does not match usable transcripts")
    for row in index.mapping.itertuples(index=False):
        segment = records[str(row.segment_id)]
        if (
            str(row.video_id) != segment.video_id
            or int(row.segment_index) != segment.segment_index
            or int(row.start_ms) != segment.start_ms
            or int(row.end_ms) != segment.end_ms
        ):
            raise ValueError(f"ASR mapping changed segment identity: {row.segment_id}")


def run_validate(config: OfflineIndexConfig, models: Any) -> Path:
    """Round-trip all bundles and atomically write the successful build report."""

    from hcmai.common.utils.io import atomic_write, write_json
    from hcmai.retrieval.retriever.artifacts import fingerprint_files
    from hcmai.retrieval.retriever.dense.index import DenseIndex
    from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex
    from hcmai.retrieval.retriever.segment.artifacts import transcript_lineage_files

    inputs = _inspect_inputs(config)
    _validate_projected_frames(inputs.projected, config.projected_frames_path)
    visual = DenseIndex.load(config.indexes.visual)
    context = DenseIndex.load(config.indexes.context)
    asr = SegmentDenseIndex.load(config.indexes.asr_segments)

    canonical_ids = set(inputs.projected["frame_id"].astype(str))
    visual_ids = set(visual.mapping["frame_id"].astype(str))
    context_ids = set(context.mapping["frame_id"].astype(str))
    if visual.metadata.vector_count != config.dataset.expected_frame_count:
        raise ValueError("Visual vector count does not match expected corpus coverage")
    if visual_ids != canonical_ids:
        raise ValueError("Visual frame_id coverage does not match canonical frames")
    if not context_ids.issubset(canonical_ids):
        raise ValueError("Context index contains non-canonical frame_id values")
    if "frame_id" in asr.mapping.columns:
        raise ValueError("ASR segment mapping must not contain frame_id")
    _validate_frame_identity(visual.mapping, inputs.projected, "Visual")
    _validate_frame_identity(context.mapping, inputs.projected, "Context")

    config_fingerprint = _config_fingerprint(config)
    _validate_index_metadata(
        visual,
        models.visual_embedding,
        "Visual",
        retrieval_source="visual",
        entity_kind="frame",
        source_fingerprint=_visual_source_fingerprint(config),
        config_fingerprint=config_fingerprint,
    )
    evidence_encoder = models.resolved_evidence_embedding
    _validate_index_metadata(
        context,
        evidence_encoder,
        "Context",
        retrieval_source="context",
        entity_kind="frame",
        source_fingerprint=fingerprint_files(
            [
                config.dataset.context_path,
                config.dataset.context_path.with_name("manifest.json"),
            ]
        ),
        config_fingerprint=config_fingerprint,
    )
    _validate_index_metadata(
        asr,
        evidence_encoder,
        "ASR",
        retrieval_source="asr",
        entity_kind="segment",
        source_fingerprint=fingerprint_files(
            transcript_lineage_files(config.dataset.transcripts_path)
        ),
        config_fingerprint=config_fingerprint,
    )
    if context.metadata.embedding_dim != asr.metadata.embedding_dim:
        raise ValueError("Context and ASR index dimensions must match")
    _validate_segment_identity(asr, config.dataset.transcripts_path)

    bundles = {
        "visual": (config.indexes.visual, visual),
        "context": (config.indexes.context, context),
        "asr_segments": (config.indexes.asr_segments, asr),
    }
    report = {
        "status": "passed",
        "dataset_version": config.dataset.version,
        "inputs": inputs.report(),
        "indexes": {
            name: {
                "path": str(path.resolve()),
                "vector_count": index.metadata.vector_count,
                "model_name": index.metadata.model_name,
                "model_revision": index.metadata.model_revision,
                "embedding_dim": index.metadata.embedding_dim,
                "normalization": index.metadata.normalization,
                "size_bytes": _bundle_size(path),
                "schema_version": index.metadata.schema_version,
                "entity_kind": index.metadata.entity_kind,
                "retrieval_source": index.metadata.retrieval_source,
                "source_fingerprint": index.metadata.source_fingerprint,
                "config_fingerprint": index.metadata.config_fingerprint,
                "checksums": dict(index.metadata.checksums or {}),
            }
            for name, (path, index) in bundles.items()
        },
    }
    atomic_write(config.report_path, lambda staged: write_json(report, staged))
    LOGGER.info("Validation passed; report written to %s", config.report_path)
    return config.report_path


def run(args: argparse.Namespace) -> None:
    """Dispatch one stage or the sequential single-process A6000 workflow."""

    config = load_offline_config(
        args.config,
        args.model_config,
        frames=args.frames,
        keyframes_root=args.keyframes_root,
        map_keyframes_root=args.map_keyframes_root,
        context=args.context,
        transcripts=args.transcripts,
        output_root=args.output_root,
    )
    if args.s3 and args.inference_url is not None:
        raise ValueError("--s3 uses local models and cannot be combined with --inference-url")

    s3_client: Any | None = None
    s3_bucket: str | None = None
    remote: RemoteEmbeddingAdapters | None = None
    try:
        if args.s3:
            s3_client, s3_bucket = _load_s3_transport(args.s3_config)
            _download_s3_inputs(s3_client, s3_bucket, config, args)
            if args.s3_dry_run:
                LOGGER.info("S3 dry-run complete; no local build was started")
                return

        models = load_model_config(args.model_config)
        require_visual, require_text = _remote_stage_requirements(args.stage)
        remote = (
            create_remote_embedding_adapters(
                args.inference_url,
                models,
                require_visual=require_visual,
                require_text=require_text,
            )
            if args.inference_url is not None and (require_visual or require_text)
            else None
        )

        if args.stage == "preflight":
            run_preflight(config)
        elif args.stage == "visual":
            projected_frames = _require_projected_frames(config)
            if remote is None:
                build_visual(config, models, projected_frames)
            else:
                assert remote.visual is not None
                build_visual(config, models, projected_frames, encoder=remote.visual)
        elif args.stage == "context":
            projected_frames = _require_projected_frames(config)
            if remote is None:
                build_context(config, models, projected_frames)
            else:
                assert remote.text is not None
                build_context(config, models, projected_frames, encoder=remote.text)
        elif args.stage == "asr":
            if remote is None:
                build_asr(config, models)
            else:
                assert remote.text is not None
                build_asr(config, models, encoder=remote.text)
        elif args.stage == "validate":
            run_validate(config, models)
        else:
            projected_frames = run_preflight(config)
            if remote is None:
                build_visual(config, models, projected_frames)
            else:
                assert remote.visual is not None
                build_visual(config, models, projected_frames, encoder=remote.visual)
            release_gpu_memory()
            text_encoder = remote.text if remote is not None else create_text_encoder(models)
            assert text_encoder is not None
            build_context(config, models, projected_frames, encoder=text_encoder)
            build_asr(config, models, encoder=text_encoder)
            run_validate(config, models)
        if args.s3 and args.stage in {"all", "validate"}:
            assert s3_client is not None
            assert s3_bucket is not None
            _publish_s3_bundle(s3_client, s3_bucket, config, args)
    finally:
        if remote is not None:
            remote.service.close()
        _close_s3_transport(s3_client)


def main(argv: list[str] | None = None) -> int:
    """Run the requested offline stage and return a shell-friendly status."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        run(parse_args(argv))
    except Exception:
        LOGGER.exception("Offline retrieval index workflow failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
