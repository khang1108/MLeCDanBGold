"""Read-only diagnostics for configured local runtime artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from hcmai.common.config import AppConfig
from hcmai.common.utils.io import read_json
from hcmai.data.pipeline import DataService


def diagnose(
    config_path: str | Path,
    *,
    sample_size: int = 100,
    check_remote: bool = False,
) -> dict[str, Any]:
    """Return one safe snapshot without loading model weights."""
    settings = AppConfig.from_yaml(config_path)
    metadata_path = settings.dataset.frames_path
    report: dict[str, Any] = {
        "config": str(config_path),
        "metadata": _metadata_report(metadata_path),
        "frame_assets": _asset_report(settings, metadata_path, sample_size),
        "visual_index": _index_report(settings.index.path, metadata_path),
        "evidence": {
            source: _file_report(path)
            for source, path in {
                "caption": settings.dataset.enrichment.caption_path,
                "ocr": settings.dataset.enrichment.ocr_path,
                "asr": settings.dataset.enrichment.asr_path,
            }.items()
        },
    }
    if check_remote:
        report["remote_inference"] = _remote_report(settings)
    report["ready"] = all(
        report[name].get("ready", False)
        for name in ("metadata", "frame_assets", "visual_index")
    )
    return report


def _metadata_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"ready": False, "path": str(path), "reason": "missing"}
    try:
        table = pd.read_parquet(path, columns=["frame_id", "video_id"])
    except Exception as error:
        return {
            "ready": False,
            "path": str(path),
            "reason": type(error).__name__,
        }
    return {
        "ready": not table.empty and not table["frame_id"].duplicated().any(),
        "path": str(path),
        "frames": len(table),
        "videos": len(set(table["video_id"].astype(str).tolist())),
    }


def _asset_report(
    settings: AppConfig,
    metadata_path: Path,
    sample_size: int,
) -> dict[str, Any]:
    if not metadata_path.is_file():
        return {"ready": False, "checked": 0, "available": 0, "missing": 0}
    try:
        data = DataService.load(
            metadata_path,
            dataset_root=settings.dataset.root,
        )
        return data.frame_asset_status(sample_size=sample_size).as_dict()
    except Exception as error:
        return {
            "ready": False,
            "checked": 0,
            "available": 0,
            "missing": 0,
            "reason": type(error).__name__,
        }


def _index_report(index_dir: Path, metadata_path: Path) -> dict[str, Any]:
    metadata_file = index_dir / "metadata.json"
    mapping_file = index_dir / "frame_mapping.parquet"
    index_file = index_dir / "dense.index"
    required = (metadata_file, mapping_file, index_file)
    if not all(path.is_file() for path in required):
        return {"ready": False, "path": str(index_dir), "reason": "missing_files"}
    try:
        index_metadata = read_json(metadata_file)
        mapping = pd.read_parquet(mapping_file, columns=["frame_id"])
        canonical = pd.read_parquet(metadata_path, columns=["frame_id"])
        aligned = mapping["frame_id"].tolist() == canonical["frame_id"].tolist()
    except Exception as error:
        return {
            "ready": False,
            "path": str(index_dir),
            "reason": type(error).__name__,
        }
    vector_count = int(index_metadata.get("vector_count", -1))
    return {
        "ready": aligned and vector_count == len(mapping),
        "path": str(index_dir),
        "vectors": vector_count,
        "mapping_rows": len(mapping),
        "canonical_alignment": aligned,
        "embedding_dim": index_metadata.get("embedding_dim"),
    }


def _file_report(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"configured": False, "available": False}
    return {
        "configured": True,
        "available": path.is_file() and path.stat().st_size > 0,
        "path": str(path),
    }


def _remote_report(settings: AppConfig) -> dict[str, Any]:
    base_url = os.getenv("HCMAI_INFERENCE_BASE_URL", settings.inference.base_url)
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/ready", timeout=5.0)
        response.raise_for_status()
        payload = response.json()
        return {
            "ready": bool(payload.get("ready")),
            "capabilities": payload.get("capabilities", {}),
        }
    except Exception as error:
        return {"ready": False, "reason": type(error).__name__}
