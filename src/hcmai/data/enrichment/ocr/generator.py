"""Generate resumable OCR frame and region artifacts from canonical frames.

The canonical frame identity and backend region order are preserved. Only
completed, lineage-matching, region-consistent rows are reused.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import pandas as pd
from PIL import Image
from tqdm import tqdm

from hcmai.common.schemas import OCREvidence, OCRRegion
from hcmai.common.utils.image import load_image
from hcmai.common.utils.io import read_json
from hcmai.data.stores.frame import FrameStore

from .adapters.florence import FlorenceAdapter
from .artifacts import failure_row, parsed_row, valid_ocr, write_ocr_artifacts
from .config import OCRConfig
from .models.contracts import OCRAdapter
from .models.entities import Evidence, FailureDetail, FrameRow
from .report import build_ocr_report


def _read_rows(path: Path, *, required: bool = False) -> list[FrameRow]:
    """Read Parquet records, optionally requiring the canonical source."""

    if not path.exists():
        if required:
            raise FileNotFoundError(f"required canonical frames not found: {path}")
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
            if (
                not isinstance(candidate.get("frame_id"), str)
                or not candidate["frame_id"]
                or candidate["frame_id"].strip() != candidate["frame_id"]
                or not isinstance(candidate.get("video_id"), str)
                or not candidate["video_id"]
                or candidate["video_id"].strip() != candidate["video_id"]
                or isinstance(candidate.get("frame_idx"), bool)
                or not isinstance(candidate.get("frame_idx"), Integral)
                or isinstance(candidate.get("timestamp_ms"), bool)
                or not isinstance(candidate.get("timestamp_ms"), Integral)
            ):
                return None
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
        and all(region.video_id == row.video_id for region in parsed)
        and all(region.frame_idx == row.frame_idx for region in parsed)
        and all(region.timestamp_ms == row.timestamp_ms for region in parsed)
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
            or row.timestamp_ms != int(frame["timestamp_ms"])
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


def _load_ocr_image(frame: FrameRow, config: OCRConfig, root: Path) -> Any:
    """Load and thumbnail one frame's OCR image, or return the raised exception."""

    try:
        path = Path(str(frame["image_path"])).expanduser()
        image_path = path if path.is_absolute() else root / path
        image = load_image(image_path, mode="RGB")
        if config.image_size:
            image.thumbnail((config.image_size, config.image_size))
        return image
    except Exception as error:  # noqa: BLE001 - surfaced as a per-frame failure row
        return error


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
    image_workers: int = 1,
) -> None:
    """Process independent batches while containing per-frame failures.

    ``image_workers`` only parallelizes local disk image decoding/thumbnailing;
    it never changes image content, order, or the resulting OCR rows.
    """

    for start in tqdm(
        range(0, len(todo), config.batch_size),
        desc="Generating OCR",
        unit="batch",
    ):
        chunk = todo[start : start + config.batch_size]
        valid: list[tuple[FrameRow, Image.Image]] = []
        if image_workers > 1 and len(chunk) > 1:
            with ThreadPoolExecutor(max_workers=image_workers) as pool:
                loaded = list(pool.map(lambda frame: _load_ocr_image(frame, config, root), chunk))
        else:
            loaded = [_load_ocr_image(frame, config, root) for frame in chunk]
        for frame, outcome in zip(chunk, loaded):
            frame_id = str(frame["frame_id"])
            if isinstance(outcome, Exception):
                rows[frame_id], failures[frame_id] = failure_row(
                    frame,
                    config,
                    "image_load",
                    outcome,
                    frame_store_id=frame_store_id,
                    model_revision=model_revision,
                )
                regions[frame_id] = []
            else:
                valid.append((frame, outcome))
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
    image_workers: int = 1,
) -> dict[str, Any]:
    """Generate or resume deterministic structured OCR artifacts."""

    started, began = datetime.now(timezone.utc), perf_counter()
    path, root = Path(frames_path), Path(dataset_root).expanduser().resolve()
    frames = [
        frame.model_dump(mode="python")
        for frame in FrameStore.load(path).iter_frames()
    ]
    order = [str(frame["frame_id"]) for frame in frames]
    if len(order) != len(set(order)):
        raise ValueError("input frames contain duplicate frame_id values")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "ocr_report.json"
    old = cast(dict[str, Any], read_json(report_path)) if report_path.exists() else {}
    # Requested configuration is authoritative over stale report metadata.
    expected_revision = config.revision
    if engine is not None:
        expected_revision = (
            getattr(engine, "resolved_revision", None) or expected_revision
        )

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
    prior_row_count = skipped + retried

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
            image_workers=image_workers,
        )

    revision = getattr(engine, "resolved_revision", None) or expected_revision
    if revision != expected_revision and skipped:
        # A lazy runtime may resolve a different immutable revision only after
        # the partial batch runs. Reprocess reused rows to prevent mixed lineage.
        assert engine is not None
        rows.clear()
        regions.clear()
        failures.clear()
        evidence.clear()
        _process(
            frames,
            rows,
            regions,
            failures,
            evidence,
            engine,
            config,
            root,
            frame_store_id=frame_store_id,
            model_revision=revision,
        )
        if getattr(engine, "resolved_revision", None) not in {None, revision}:
            raise RuntimeError("OCR runtime revision changed during generation")
        todo = frames
        skipped = 0
        retried = prior_row_count
    elif revision != expected_revision:
        for frame_id in rows:
            rows[frame_id] = rows[frame_id].model_copy(
                update={"model_revision": revision}
            )

    if len(rows) != len(order) or any(frame_id not in rows for frame_id in order):
        if config.enabled:
            raise ValueError("OCR artifact does not cover every canonical frame")
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
        frame_store_id=frame_store_id,
    )
    manifest = {key: value for key, value in report.items() if key != "raw_evidence"}
    write_ocr_artifacts(
        output,
        order,
        rows,
        regions,
        failures,
        config,
        report,
        manifest,
    )
    return report
