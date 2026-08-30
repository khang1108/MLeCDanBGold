"""Offline BTC frame-store preparation without a runtime Corpus dependency.

This module owns the organizer-import adapter used by corpus preparation.  It
does not provide online corpus reads or mutate an already-open ``Corpus``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from hcmai.common.utils.io import read_yaml_section
from offline.enrichment.dataset_cli import merge_dataset_values
from offline.ingestion import BTCIngestionConfig, import_btc_frame_store


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def prepare_btc_frame_store(
    config_path: str | Path,
    *,
    dataset: Mapping[str, object] | None = None,
) -> Path:
    """Import the configured BTC keyframe mapping into its canonical artifact."""

    resolved_config = Path(config_path).expanduser().resolve()
    raw_config = read_yaml_section(resolved_config, "enrichment")
    dataset_values = merge_dataset_values(
        raw_config,
        dict(dataset) if dataset else None,
    )
    required = {
        "version",
        "source",
        "btc_root",
        "mapping_root",
        "data_root",
        "frame_store_id",
        "frames_path",
        "frame_store_output",
    }
    missing = sorted(required.difference(dataset_values))
    if missing:
        raise ValueError("Missing dataset configuration: " + ", ".join(missing))
    if str(dataset_values["source"]) != "btc_keyframes":
        raise ValueError(
            "Unsupported dataset.source "
            f"{dataset_values['source']!r}; expected 'btc_keyframes'"
        )

    output_root = _project_path(str(dataset_values["frame_store_output"]))
    frames_path = _project_path(str(dataset_values["frames_path"]))
    if frames_path.resolve() != (output_root / "frames.parquet").resolve():
        raise ValueError(
            "dataset.frames_path must equal dataset.frame_store_output/frames.parquet"
        )
    return import_btc_frame_store(
        BTCIngestionConfig(
            btc_root=_project_path(str(dataset_values["btc_root"])),
            mapping_root=_project_path(str(dataset_values["mapping_root"])),
            data_root=_project_path(str(dataset_values["data_root"])),
            output_root=output_root,
            frame_store_id=str(dataset_values["frame_store_id"]),
        )
    )


def _project_path(value: str | Path) -> Path:
    """Resolve a preparation configuration path from the repository root."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else _PROJECT_ROOT / path


__all__ = ["prepare_btc_frame_store"]
