"""Shared command-line contract for selecting one preparation dataset.

This module owns runtime dataset identity and path arguments used by enrichment
CLIs. Model policies and stage output policies remain in ``prepare.yaml``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

_ConfigT = TypeVar("_ConfigT", bound="DataclassInstance")


_CLI_NAMES = {
    "version": "--version",
    "source": "--source",
    "frame_store_id": "--frame-store-id",
    "data_root": "--data-root",
    "frames_path": "--frames",
    "frame_store_output": "--frame-store-output",
}


def add_dataset_arguments(
    parser: argparse.ArgumentParser,
    *,
    required: bool = False,
) -> None:
    """Add the common runtime dataset selector and path arguments.

    When ``required`` is false, parsers remain usable for unit tests and for
    callers that inject a validated job. Production loaders still reject a
    missing dataset when the YAML no longer contains a dataset section.
    """

    parser.add_argument(
        "--version",
        dest="dataset_version",
        help="Dataset version recorded in every generated artifact.",
        required=required,
    )
    parser.add_argument(
        "--source",
        help="Dataset source label, for example custom_raw_video_1fps.",
        required=required,
    )
    parser.add_argument(
        "--frame-store-id",
        help="Canonical FrameStore lineage shared by every stage.",
        required=required,
    )
    parser.add_argument(
        "--data-root",
        "--dataset-root",
        dest="data_root",
        type=Path,
        help="Root used to resolve relative frame assets.",
        required=required,
    )
    parser.add_argument(
        "--frames",
        type=Path,
        help="Canonical FrameStore frames.parquet input.",
        required=required,
    )
    parser.add_argument(
        "--frame-store-output",
        type=Path,
        help="Canonical FrameStore artifact directory.",
        required=required,
    )


def dataset_overrides(args: argparse.Namespace) -> dict[str, Any] | None:
    """Return one complete dataset mapping from CLI values.

    Returns ``None`` when no dataset argument was provided, which lets tests
    inject a job object. Supplying only part of the dataset contract is an
    error so a stage cannot silently mix CLI and YAML identities.
    """

    values = {
        "version": getattr(args, "dataset_version", None),
        "source": getattr(args, "source", None),
        "frame_store_id": getattr(args, "frame_store_id", None),
        "data_root": getattr(args, "data_root", None),
        "frames_path": getattr(args, "frames", None),
        "frame_store_output": getattr(args, "frame_store_output", None),
    }
    provided = [value is not None for value in values.values()]
    if not any(provided):
        return None
    if not all(provided):
        missing = [
            _CLI_NAMES[name]
            for name, value in values.items()
            if value is None
        ]
        raise ValueError(
            "dataset arguments must be supplied together; missing: "
            + ", ".join(missing)
        )
    for name in ("version", "source", "frame_store_id"):
        value = values[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name.replace('_', '-')} must be non-empty")
        values[name] = value.strip()
    return values


def apply_overrides(config: _ConfigT, **values: Any) -> _ConfigT:
    """Replace only the stage-config fields whose CLI override was supplied."""

    supplied = {name: value for name, value in values.items() if value is not None}
    return replace(config, **supplied) if supplied else config


def merge_dataset_values(
    raw: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge optional CLI dataset values over a legacy YAML dataset mapping."""

    configured = raw.get("dataset", {})
    if configured is None:
        configured = {}
    if not isinstance(configured, dict):
        raise ValueError("YAML dataset section must be a mapping")
    values = dict(configured)
    if overrides:
        values.update(overrides)
    return values


__all__ = [
    "add_dataset_arguments",
    "apply_overrides",
    "dataset_overrides",
    "merge_dataset_values",
]
