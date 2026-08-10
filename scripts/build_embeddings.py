"""Generate visual embeddings and build their exact FAISS index."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from hcmai.common.utils.io import read_parquet, read_yaml
from hcmai.common.utils.logging import configure_logging, get_logger
from hcmai.retrieval.embedding.pipeline import EmbeddingService
from hcmai.llm.pipeline import LLMServiceConfig
from hcmai.retrieval.retriever.pipeline import RetrievalService
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


def _build_index(
    run: Any,
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    """Build FAISS from one completed embedding run."""
    if not run.generated_count:
        raise RuntimeError("No embeddings were generated")
    embeddings = np.load(run.embeddings_file, mmap_mode="r")
    mapping = read_parquet(run.mapping_file)
    metadata = run.metadata
    index = RetrievalService.build_index(
        embeddings,
        mapping,
        dataset_version=metadata.dataset_version,
        model_name=metadata.model_name,
        index_type=config.get("index", {}).get("type", "flat_ip"),
    )
    index_dir = _index_output_dir(config, output_dir)
    index.save(index_dir)
    logger.info(
        "FAISS index saved to %s",
        index_dir / RetrievalService.INDEX_FILENAME,
    )


def _run(args: Any) -> None:
    """Generate embeddings and their exact index."""
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
    output_dir = Path(args.output)
    model_config = LLMServiceConfig.from_yaml(args.model_config)
    run = EmbeddingService.build_visual_artifacts(
        frames_path=frames_path,
        dataset_root=dataset_root,
        output_dir=output_dir,
        encoder_config=model_config.visual_embedding,
        dataset_version=dataset.get("version", "unknown"),
    )
    _build_index(run, config, output_dir)


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
