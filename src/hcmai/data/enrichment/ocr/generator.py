"""Bộ sinh dữ liệu (Generator) cho OCR.

Quản lý luồng thực thi mô hình nhận diện chữ trên các khung hình video (OCR pipeline).

Các tính năng chính:
1. Nhận diện (Detection): Quét ảnh để tìm các vùng chữ (Bounding Box).
2. Trích xuất (Recognition): Chuyển đổi vùng ảnh chứa chữ thành chuỗi ký tự (Text).
3. Cơ chế Resume: Khôi phục tiến trình tự động bằng cách bỏ qua các frames đã có kết quả OCR."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import pandas as pd
from PIL import Image
from tqdm import tqdm

from hcmai.common.schemas import FrameEnrichment, validate_frame_enrichment
from hcmai.common.utils.image import load_image
from hcmai.common.utils.io import atomic_write, read_json, write_json

from .artifacts import (
    failure_row,
    parsed_row,
    valid_ocr,
    write_ocr_artifacts,
)
from .adapters.florence import FlorenceAdapter
from .config import OCRConfig
from .models.contracts import OCRAdapter
from .models.entities import Evidence, FailureDetail, FrameRow
from .report import build_ocr_report


def _resume(
    frames: list[FrameRow], path: Path, config: OCRConfig, frame_store_id: str | None = None
) -> tuple[dict[str, FrameEnrichment], list[FrameRow], int, int]:
    groups: dict[str, list[FrameRow]] = {}
    if path.exists():
        prior = cast(
            list[FrameRow], pd.read_parquet(path).to_dict(orient="records")
        )
        for row in prior:
            if row.get("enrichment_version") == config.enrichment_version and (
                frame_store_id is None or row.get("frame_store_id") == frame_store_id
            ):
                groups.setdefault(str(row.get("frame_id")), []).append(row)
    rows: dict[str, FrameEnrichment] = {}
    todo: list[FrameRow] = []
    skipped, retried = 0, 0
    for frame in frames:
        frame_id = str(frame["frame_id"])
        old = groups.get(frame_id, [])
        row = valid_ocr(old[0], config) if len(old) == 1 else None
        if row:
            rows[frame_id], skipped = row, skipped + 1
        else:
            retried += bool(old)
            todo.append(frame)
    return rows, todo, skipped, retried


def _process(
    todo: list[FrameRow],
    rows: dict[str, FrameEnrichment],
    failures: dict[str, FailureDetail],
    evidence: dict[str, Evidence],
    engine: OCRAdapter,
    config: OCRConfig,
    root: Path,
) -> None:
    for start in tqdm(range(0, len(todo), config.batch_size), desc="Generating OCR", unit="batch"):
        valid: list[tuple[str, Image.Image]] = []
        for frame in todo[start : start + config.batch_size]:
            frame_id = str(frame["frame_id"])
            try:
                path = Path(str(frame["image_path"])).expanduser()
                image_path = path if path.is_absolute() else root / path
                image = load_image(image_path, mode="RGB")
                if config.image_size:
                    image.thumbnail((config.image_size, config.image_size))
                valid.append((frame_id, image))
            except Exception as error:
                rows[frame_id], failures[frame_id] = failure_row(
                    frame_id, config, "image_load", error
                )
        if not valid:
            continue
        try:
            results: list[object] = list(
                engine.recognize_batch([image for _, image in valid])
            )
            if len(results) != len(valid):
                raise ValueError("OCR backend returned the wrong result count")
        except Exception as error:
            results = [error] * len(valid)
        finally:
            for _, image in valid:
                image.close()
        for (frame_id, _), result in zip(valid, results):
            try:
                rows[frame_id], evidence[frame_id] = parsed_row(
                    frame_id, result, config
                )
            except Exception as error:
                rows[frame_id], failures[frame_id] = failure_row(
                    frame_id, config, "backend", error
                )


def generate_ocr(
    frames_path: str | Path,
    output_dir: str | Path,
    config: OCRConfig,
    engine: OCRAdapter | None ,
    engine_factory: Callable[[OCRConfig], OCRAdapter] | None = None,
    *,
    dataset_root: str | Path = ".",
    frame_store_id: str | None = None,
) -> dict[str, Any]:
    """Generate or resume one deterministic independent OCR artifact."""
    started, began = datetime.now(timezone.utc), perf_counter()
    path, root = Path(frames_path), Path(dataset_root).expanduser().resolve()
    frames = cast(
        list[FrameRow], pd.read_parquet(path).to_dict(orient="records")
    )
    order = [str(frame["frame_id"]) for frame in frames]
    if len(order) != len(set(order)):
        raise ValueError("input frames contain duplicate frame_id values")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "ocr_report.json"
    old = (
        cast(dict[str, Any], read_json(report_path))
        if report_path.exists()
        else {}
    )
    rows, todo, skipped, retried = (
        _resume(frames, output / "frame_enrichment.parquet", config, frame_store_id)
        if config.enabled
        else ({}, [], 0, 0)
    )
    prior = old.get("raw_evidence", [])
    evidence: dict[str, Evidence] = {
        str(item["frame_id"]): item
        for item in prior
        if isinstance(item, dict) and item.get("frame_id") in rows
    }
    failures: dict[str, FailureDetail] = {}
    if todo:
        if engine is None and engine_factory is None:
            if config.backend == "remote":
                raise NotImplementedError("Remote OCR adapter is not implemented.")
            engine_factory = FlorenceAdapter

        engine = engine or engine_factory(config)
        _process(todo, rows, failures, evidence, engine, config, root)

    for row in rows.values():
        if frame_store_id is not None:
            row.frame_store_id = frame_store_id

    validate_frame_enrichment(rows, order, frame_store_id)
    write_ocr_artifacts(output, order, rows, failures)
    revision = (
        getattr(engine, "resolved_revision", None)
        or old.get("resolved_revision")
        or config.revision
    )
    report = build_ocr_report(
        config, path, root, rows, evidence, failures, old, started,
        perf_counter() - began, len(frames), len(todo), skipped, retried,
        revision, len(frames) if not config.enabled else 0,
    )
    atomic_write(report_path, lambda target: write_json(report, target))
    manifest = {key: value for key, value in report.items() if key != "raw_evidence"}
    atomic_write(
        output / "manifest.json",
        lambda target: write_json(manifest, target),
    )
    return report
