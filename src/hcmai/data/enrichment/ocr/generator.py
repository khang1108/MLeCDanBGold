"""Generate resumable OCR frame and region artifacts from canonical frames.

The canonical frame identity and backend region order are preserved. Only
completed, lineage-matching, region-consistent rows are reused.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import pandas as pd
from PIL import Image
from tqdm import tqdm

from hcmai.common.schemas import OCREvidence, OCRRegion
from hcmai.common.utils.image import load_image
from hcmai.common.utils.io import atomic_write, read_json, write_json

from .adapters.florence import FlorenceAdapter
from .artifacts import failure_row, parsed_row, valid_ocr, write_ocr_artifacts
from .config import OCRConfig
from .models.contracts import OCRAdapter
from .models.entities import Evidence, FailureDetail, FrameRow
from .report import build_ocr_report


def _read_rows(path: Path) -> list[FrameRow]:
    """Read a Parquet table as record dictionaries when it exists."""

    if not path.exists():
        return []
    return cast(list[FrameRow], pd.read_parquet(path).to_dict(orient="records"))


def _consistent_regions(
    row: OCREvidence, candidates: list[FrameRow]
) -> list[OCRRegion] | None:
    """Validate the exact region identity/order promised by one frame row."""

    if len(candidates) != row.region_count:
        return None
    parsed: list[OCRRegion] = []
    try:
        for candidate in candidates:
            values = {
                key: None
                if isinstance(value, float) and pd.isna(value)
                else value
                for key, value in candidate.items()
            }
            parsed.append(OCRRegion.model_validate(values))
    except Exception:
        return None

    parsed.sort(key=lambda region: region.region_order)
    expected_orders = list(range(row.region_count))
    expected_ids = [f"{row.frame_id}:{order}" for order in expected_orders]
    return parsed if (
        [region.region_order for region in parsed] == expected_orders
        and [region.region_id for region in parsed] == expected_ids
        and all(region.frame_id == row.frame_id for region in parsed)
        and all(region.frame_idx == row.frame_idx for region in parsed)
    ) else None


def _resume(
    frames: list[FrameRow],
    frames_path: Path,
    regions_path: Path,
    config: OCRConfig,
    *,
    frame_store_id: str | None,
    model_revision: str | None,
) -> tuple[
    dict[str, OCREvidence],
    dict[str, list[OCRRegion]],
    list[FrameRow],
    int,
    int,
]:
    """Reuse only valid frame rows whose structured region table is consistent."""

    old_frames: dict[str, list[FrameRow]] = {}
    for row in _read_rows(frames_path):
        old_frames.setdefault(str(row.get("frame_id")), []).append(row)
    old_regions: dict[str, list[FrameRow]] = {}
    for row in _read_rows(regions_path):
        old_regions.setdefault(str(row.get("frame_id")), []).append(row)

    rows: dict[str, OCREvidence] = {}
    regions: dict[str, list[OCRRegion]] = {}
    todo: list[FrameRow] = []
    skipped = retried = 0
    for frame in frames:
        frame_id = str(frame["frame_id"])
        candidates = old_frames.get(frame_id, [])
        row = (
            valid_ocr(
                candidates[0],
                config,
                frame_store_id=frame_store_id,
                model_revision=model_revision,
            )
            if len(candidates) == 1
            else None
        )
        region_rows = _consistent_regions(row, old_regions.get(frame_id, [])) if row else None
        if row is not None and (
            row.video_id != str(frame["video_id"])
            or row.frame_idx != int(frame["frame_idx"])
        ):
            region_rows = None
        if row is not None and region_rows is not None:
            rows[frame_id] = row
            regions[frame_id] = region_rows
            skipped += 1
        else:
            retried += bool(candidates)
            todo.append(frame)
    return rows, regions, todo, skipped, retried


def _process(
    todo: list[FrameRow],
    rows: dict[str, OCREvidence],
    regions: dict[str, list[OCRRegion]],
    failures: dict[str, FailureDetail],
    evidence: dict[str, Evidence],
    engine: OCRAdapter,
    config: OCRConfig,
    root: Path,
    *,
    frame_store_id: str | None,
    model_revision: str | None,
) -> None:
    """Process independent batches while containing per-frame failures."""

    for start in tqdm(
        range(0, len(todo), config.batch_size),
        desc="Generating OCR",
        unit="batch",
    ):
        valid: list[tuple[FrameRow, Image.Image]] = []
        for frame in todo[start : start + config.batch_size]:
            frame_id = str(frame["frame_id"])
            try:
                path = Path(str(frame["image_path"])).expanduser()
                image_path = path if path.is_absolute() else root / path
                image = load_image(image_path, mode="RGB")
                if config.image_size:
                    image.thumbnail((config.image_size, config.image_size))
                valid.append((frame, image))
            except Exception as error:
                rows[frame_id], failures[frame_id] = failure_row(
                    frame,
                    config,
                    "image_load",
                    error,
                    frame_store_id=frame_store_id,
                    model_revision=model_revision,
                )
                regions[frame_id] = []
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

        for (frame, _), result in zip(valid, results):
            frame_id = str(frame["frame_id"])
            try:
                rows[frame_id], regions[frame_id], evidence[frame_id] = parsed_row(
                    frame,
                    result,
                    config,
                    frame_store_id=frame_store_id,
                    model_revision=model_revision,
                )
            except Exception as error:
                rows[frame_id], failures[frame_id] = failure_row(
                    frame,
                    config,
                    "backend",
                    error,
                    frame_store_id=frame_store_id,
                    model_revision=model_revision,
                )
                regions[frame_id] = []


def generate_ocr(
    frames_path: str | Path,
    output_dir: str | Path,
    config: OCRConfig,
    engine: OCRAdapter | None = None,
    engine_factory: Callable[[OCRConfig], OCRAdapter] | None = None,
    *,
    dataset_root: str | Path = ".",
    frame_store_id: str | None = None,
) -> dict[str, Any]:
    """Generate or resume deterministic structured OCR artifacts."""

    started, began = datetime.now(timezone.utc), perf_counter()
    path, root = Path(frames_path), Path(dataset_root).expanduser().resolve()
    frames = _read_rows(path)
    order = [str(frame["frame_id"]) for frame in frames]
    if len(order) != len(set(order)):
        raise ValueError("input frames contain duplicate frame_id values")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "ocr_report.json"
    old = cast(dict[str, Any], read_json(report_path)) if report_path.exists() else {}
    expected_revision = old.get("resolved_revision") or config.revision

    if config.enabled:
        rows, regions, todo, skipped, retried = _resume(
            frames,
            output / "frames.parquet",
            output / "regions.parquet",
            config,
            frame_store_id=frame_store_id,
            model_revision=expected_revision,
        )
    else:
        rows, regions, todo, skipped, retried = {}, {}, [], 0, 0

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
                raise NotImplementedError("Remote OCR adapter requires a client")
            engine_factory = FlorenceAdapter
        if engine is None:
            assert engine_factory is not None
            engine = engine_factory(config)
        _process(
            todo,
            rows,
            regions,
            failures,
            evidence,
            engine,
            config,
            root,
            frame_store_id=frame_store_id,
            model_revision=expected_revision,
        )

    revision = getattr(engine, "resolved_revision", None) or expected_revision
    if revision != expected_revision:
        processed_ids = {str(frame["frame_id"]) for frame in todo}
        for frame_id in processed_ids:
            if frame_id in rows:
                rows[frame_id] = rows[frame_id].model_copy(
                    update={"model_revision": revision}
                )

    if len(rows) != len(order) or any(frame_id not in rows for frame_id in order):
        if config.enabled:
            raise ValueError("OCR artifact does not cover every canonical frame")
    write_ocr_artifacts(output, order, rows, regions, failures, config)

    report = build_ocr_report(
        config,
        path,
        root,
        rows,
        regions,
        evidence,
        failures,
        old,
        started,
        perf_counter() - began,
        len(frames),
        len(todo),
        skipped,
        retried,
        revision,
        len(frames) if not config.enabled else 0,
    )
    atomic_write(report_path, lambda target: write_json(report, target))
    manifest = {key: value for key, value in report.items() if key != "raw_evidence"}
    atomic_write(output / "manifest.json", lambda target: write_json(manifest, target))
    return report
