"""Generate visual embeddings and build their exact FAISS index."""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import numpy as np

from hcmai.common.utils.io import read_parquet, read_yaml
from hcmai.common.utils.logging import configure_logging, get_logger
from hcmai.retrieval.embedding.artifacts import EmbeddingArtifactBuilder
from hcmai.retrieval.embedding.models.metadata import EmbeddingMetadata
from hcmai.llm.pipeline import LLMServiceConfig
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.retrieval.retriever.artifacts import publish_directory
from hcmai.retrieval.retriever.dense.index import DenseIndex
from script_args import parse_arguments

logger = get_logger(__name__)


def _index_output_dir(config: dict[str, Any], output_dir: Path) -> Path:
    """Return the configured FAISS artifact directory."""
    configured = config.get("index", {}).get("path")
    if not configured:
        return output_dir / "indexes"
    path = Path(configured)
    if path.suffix:
        if path.name != RetrievalService.INDEX_FILENAME:
            raise ValueError(
                f"index.path filename must be {RetrievalService.INDEX_FILENAME!r}, "
                f"got {path.name!r}"
            )
        return path.parent
    return path


def _build_staged_index(
    embeddings_file: Path,
    mapping_file: Path,
    metadata: EmbeddingMetadata,
    config: dict[str, Any],
    index_dir: Path,
) -> None:
    """Write and validate an unpublished DenseIndex from complete embeddings."""
    if not metadata.successful_frames:
        raise RuntimeError("No embeddings were generated")
    embeddings = np.load(embeddings_file, mmap_mode="r")
    mapping = read_parquet(mapping_file)
    index = RetrievalService.build_index(
        embeddings,
        mapping,
        dataset_version=metadata.dataset_version,
        model_name=metadata.model_name,
        index_type=config.get("index", {}).get("type", "flat_ip"),
    )
    index_dir.mkdir(parents=True, exist_ok=False)
    index._write_bundle(index_dir)
    DenseIndex.load(index_dir)
    logger.info(
        "FAISS index saved to %s",
        index_dir / RetrievalService.INDEX_FILENAME,
    )


def _run(args: Any) -> None:
    """Stage, validate, and atomically publish visual embeddings and index."""
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = read_yaml(config_path)
    dataset = config.get("dataset", {})
    frames_value = args.frames or dataset.get("frames_path")
    root_value = args.dataset_root or dataset.get("root")
    if not frames_value or not root_value:
        raise ValueError("dataset root and frames path must be configured")
    frames_path = Path(frames_value)
    dataset_root = Path(root_value)
    if not frames_path.is_file():
        raise FileNotFoundError(f"Frames file not found: {frames_path}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    output_dir = Path(args.output).expanduser().resolve()
    index_dir = _index_output_dir(config, output_dir).resolve()
    try:
        staged_index_relative = index_dir.relative_to(output_dir)
    except ValueError as error:
        raise ValueError(
            "index.path must be inside --output so visual artifacts and their "
            "DenseIndex can be published together"
        ) from error

    model_config = LLMServiceConfig.from_yaml(args.model_config)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir.with_name(
        f".{output_dir.name}.visual-checkpoints"
    )
    staging_dir = Path(
        mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        builder = EmbeddingArtifactBuilder(
            frames_path=frames_path,
            dataset_root=dataset_root,
            output_dir=staging_dir,
            encoder_config=model_config.visual_embedding,
            dataset_version=dataset.get("version", "unknown"),
            strict=args.strict,
            resume=args.resume,
            shard_size=args.shard_size,
            # Keep checkpoints outside the staged publication so a strict
            # failure preserves completed canonical slices for repair/resume.
            checkpoint_dir=checkpoint_dir,
        )
        metadata = builder.run()
        if args.strict and metadata.successful_frames != metadata.total_frames:
            raise RuntimeError("Visual build does not have complete visual coverage")
        _build_staged_index(
            builder.embeddings_file,
            builder.mapping_file,
            metadata,
            config,
            staging_dir / staged_index_relative,
        )
        publish_directory(staging_dir, output_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def main() -> int:
    """Run the embedding and index build."""
    args = parse_arguments("build_embeddings")
    configure_logging(level=args.log_level, log_file=args.log_file)
    try:
        _run(args)
        return 0
    except Exception as error:
        logger.exception("Pipeline failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
