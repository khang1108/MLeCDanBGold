"""Generate resumable OCR frame and region artifacts from canonical frames.

The canonical frame identity and backend region order are preserved. Only
completed, lineage-matching, region-consistent rows are reused.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import math
from numbers import Integral
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import pandas as pd
from PIL import Image
from tqdm import tqdm

from hcmai.common.utils.image import load_image
from hcmai.common.utils.io import (
    atomic_write,
    read_json,
    write_json,
    write_parquet,
)
from offline.enrichment.bundle import publish_staged_bundle
from offline.enrichment.models import FrameEnrichment, ProcessingStatus

from .adapter import FlorenceAdapter
from .models import (
    Evidence,
    FailureDetail,
    FrameRow,
    NormalizedRegions,
    OCRAdapter,
    OCRConfig,
    OCREvidence,
    OCRRegion,
    OCRRegionResult,
    OCRResult,
    json_safe_ocr_raw,
    normalize_regions,
)


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


def valid_ocr(
    data: FrameRow,
    config: OCRConfig,
    *,
    frame_store_id: str | None,
    model_revision: str | None,
) -> OCREvidence | None:
    """Return a completed frame row only when all reusable lineage matches."""
    try:
        if (
            not isinstance(data.get("frame_id"), str)
            or not data["frame_id"]
            or data["frame_id"].strip() != data["frame_id"]
            or not isinstance(data.get("video_id"), str)
            or not data["video_id"]
            or data["video_id"].strip() != data["video_id"]
            or isinstance(data.get("frame_idx"), bool)
            or not isinstance(data.get("frame_idx"), Integral)
            or isinstance(data.get("timestamp_ms"), bool)
            or not isinstance(data.get("timestamp_ms"), Integral)
        ):
            return None
        values = {
            key: None if isinstance(value, float) and math.isnan(value) else value
            for key, value in dict(data).items()
        }
        row = OCREvidence.model_validate(values)
    except Exception:
        return None

    valid = (
        row.status == ProcessingStatus.COMPLETED
        and row.error_code is None
        and row.error_message is None
        and row.artifact_version == config.artifact_version
        and row.model_name == config.model_name
        and row.model_revision == model_revision
        and row.frame_store_id == frame_store_id
    )
    return row if valid else None


def failure_row(
    frame: FrameRow,
    config: OCRConfig,
    stage: str,
    error: Exception,
    *,
    frame_store_id: str | None,
    model_revision: str | None,
) -> tuple[OCREvidence, FailureDetail]:
    """Build one bounded failed OCR row and machine-readable diagnostic."""
    message = " ".join(str(error).split())[:300] or type(error).__name__
    code = type(error).__name__
    frame_id = str(frame["frame_id"])
    row = OCREvidence(
        frame_id=frame_id,
        video_id=str(frame["video_id"]),
        frame_idx=int(frame["frame_idx"]),
        timestamp_ms=int(frame["timestamp_ms"]),
        frame_store_id=frame_store_id,
        artifact_version=config.artifact_version,
        model_name=config.model_name,
        model_revision=model_revision,
        status=ProcessingStatus.FAILED,
        error_code=code,
        error_message=message,
    )
    return row, {
        "frame_id": frame_id,
        "artifact_version": config.artifact_version,
        "processing_stage": stage,
        "exception_category": code,
        "error_message": message,
    }


def parsed_row(
    frame: FrameRow,
    result: object,
    config: OCRConfig,
    *,
    frame_store_id: str | None,
    model_revision: str | None,
) -> tuple[OCREvidence, list[OCRRegion], Evidence]:
    """Validate a structured backend result and preserve every raw region."""
    if not isinstance(result, OCRResult) or not isinstance(result.text, str):
        raise TypeError("OCR backend returned a malformed result")
    if not isinstance(result.regions, tuple) or any(
        not isinstance(region, OCRRegionResult) for region in result.regions
    ):
        raise TypeError("OCR backend returned malformed regions")

    frame_id = str(frame["frame_id"])
    video_id = str(frame["video_id"])
    frame_idx = int(frame["frame_idx"])
    timestamp_ms = int(frame["timestamp_ms"])
    normalized = normalize_regions(
        result.regions, min_confidence=config.min_region_confidence
    )
    region_rows = [
        OCRRegion(
            frame_id=frame_id,
            video_id=video_id,
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            region_id=f"{frame_id}:{order}",
            region_order=order,
            text=region.text,
            confidence=region.confidence,
            x_min=region.x_min,
            y_min=region.y_min,
            x_max=region.x_max,
            y_max=region.y_max,
        )
        for order, region in enumerate(result.regions)
    ]
    raw_text = "\n".join(
        region.text for region in result.regions if region.text != ""
    ) or None
    row = OCREvidence(
        frame_id=frame_id,
        video_id=str(frame["video_id"]),
        frame_idx=frame_idx,
        timestamp_ms=timestamp_ms,
        raw_text=raw_text,
        normalized_text=normalized.text,
        quality_score=normalized.quality_score,
        region_count=len(region_rows),
        frame_store_id=frame_store_id,
        artifact_version=config.artifact_version,
        model_name=config.model_name,
        model_revision=model_revision,
    )
    evidence: Evidence = {
        "frame_id": frame_id,
        "raw_output": json_safe_ocr_raw(result.raw_output),
        "usable_region_count": normalized.usable_region_count,
    }
    return row, region_rows, evidence


def _legacy_projection(row: OCREvidence, config: OCRConfig) -> FrameEnrichment:
    """Build the temporary flat OCR view required by existing consumers."""
    return FrameEnrichment(
        frame_id=row.frame_id,
        frame_store_id=row.frame_store_id,
        ocr_text=row.normalized_text,
        enrichment_version=config.enrichment_version,
        model_name=row.model_name,
        status=row.status,
        error_message=row.error_message,
    )


def write_ocr_artifacts(
    output: Path,
    order: list[str],
    rows: dict[str, OCREvidence],
    regions: dict[str, list[OCRRegion]],
    failures: dict[str, FailureDetail],
    config: OCRConfig,
    report: dict[str, object],
    manifest: dict[str, object],
) -> None:
    """Stage, validate, and publish the complete structured OCR bundle."""
    frame_table = pd.DataFrame(
        [rows[key].model_dump(mode="json") for key in order if key in rows],
        columns=list(OCREvidence.model_fields),
    )
    region_table = pd.DataFrame(
        [
            region.model_dump(mode="json")
            for frame_id in order
            for region in regions.get(frame_id, [])
        ],
        columns=list(OCRRegion.model_fields),
    )
    projection = pd.DataFrame(
        [
            _legacy_projection(rows[key], config).model_dump(mode="json")
            for key in order
            if key in rows
        ],
        columns=list(FrameEnrichment.model_fields),
    )

    failure_rows = [failures[key] for key in order if key in failures]
    output.mkdir(parents=True, exist_ok=True)
    published = (
        output / "frames.parquet",
        output / "regions.parquet",
        output / "failures.json",
        output / "frame_enrichment.parquet",
        output / "ocr_report.json",
        output / "manifest.json",
    )
    staged = tuple(
        path.with_name(f".{path.name}.staged") for path in published
    )
    try:
        atomic_write(
            staged[0],
            lambda path: write_parquet(frame_table, path, index=False),
        )
        atomic_write(
            staged[1],
            lambda path: write_parquet(region_table, path, index=False),
        )
        atomic_write(staged[2], lambda path: write_json(failure_rows, path))
        atomic_write(
            staged[3],
            lambda path: write_parquet(projection, path, index=False),
        )
        atomic_write(staged[4], lambda path: write_json(report, path))
        atomic_write(staged[5], lambda path: write_json(manifest, path))

        staged_frames = pd.read_parquet(staged[0])
        staged_regions = pd.read_parquet(staged[1])
        staged_projection = pd.read_parquet(staged[3])
        if staged_frames.columns.tolist() != list(OCREvidence.model_fields):
            raise ValueError("staged OCR frames have an invalid schema")
        if staged_regions.columns.tolist() != list(OCRRegion.model_fields):
            raise ValueError("staged OCR regions have an invalid schema")
        if staged_projection.columns.tolist() != list(FrameEnrichment.model_fields):
            raise ValueError("staged OCR projection has an invalid schema")

        expected_order = [frame_id for frame_id in order if frame_id in rows]
        if staged_frames["frame_id"].tolist() != expected_order:
            raise ValueError("staged OCR frames changed canonical order")
        if staged_projection["frame_id"].tolist() != expected_order:
            raise ValueError("staged OCR projection changed canonical order")

        parsed_frames: dict[str, OCREvidence] = {}
        for data in staged_frames.astype(object).where(
            staged_frames.notna(), None
        ).to_dict(orient="records"):
            row = OCREvidence.model_validate(data)
            parsed_frames[row.frame_id] = row
        parsed_regions: dict[str, list[OCRRegion]] = {}
        for data in staged_regions.astype(object).where(
            staged_regions.notna(), None
        ).to_dict(orient="records"):
            region = OCRRegion.model_validate(data)
            parsed_regions.setdefault(region.frame_id, []).append(region)
        for frame_id, row in parsed_frames.items():
            frame_regions = parsed_regions.get(frame_id, [])
            if len(frame_regions) != row.region_count:
                raise ValueError(
                    f"staged OCR region_count mismatch for {frame_id}"
                )
            for region_order, region in enumerate(frame_regions):
                if (
                    region.region_order != region_order
                    or region.region_id != f"{frame_id}:{region_order}"
                    or region.video_id != row.video_id
                    or region.frame_idx != row.frame_idx
                    or region.timestamp_ms != row.timestamp_ms
                ):
                    raise ValueError(
                        f"staged OCR region identity mismatch for {frame_id}"
                    )
        if set(parsed_regions).difference(parsed_frames):
            raise ValueError("staged OCR regions reference an unknown frame")

        for data in staged_projection.astype(object).where(
            staged_projection.notna(), None
        ).to_dict(orient="records"):
            objects = data.get("objects")
            to_list = getattr(objects, "tolist", None)
            if callable(to_list):
                data["objects"] = to_list()
            FrameEnrichment.model_validate(data)
        if read_json(staged[2]) != failure_rows:
            raise ValueError("staged OCR failures failed validation")
        if read_json(staged[4]) != report:
            raise ValueError("staged OCR report failed validation")
        if read_json(staged[5]) != manifest:
            raise ValueError("staged OCR manifest failed validation")

        versions = {row.artifact_version for row in rows.values()}
        lineages = {row.frame_store_id for row in rows.values()}
        if len(versions) > 1 or len(lineages) > 1:
            raise ValueError("OCR bundle has mixed version or lineage")
        if versions and manifest.get("artifact_version") not in versions:
            raise ValueError("OCR manifest artifact_version mismatch")
        if lineages and manifest.get("frame_store_id") not in lineages:
            raise ValueError("OCR manifest frame_store_id mismatch")

        publish_staged_bundle(staged, published)
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def build_ocr_report(
    config: OCRConfig,
    path: Path,
    root: Path,
    rows: dict[str, OCREvidence],
    regions: dict[str, list[OCRRegion]],
    evidence: dict[str, Evidence],
    failures: dict[str, FailureDetail],
    old: dict[str, Any],
    started: datetime,
    elapsed: float,
    input_count: int,
    processed: int,
    skipped: int,
    retried: int,
    revision: str | None,
    disabled: int,
    *,
    frame_store_id: str | None,
) -> dict[str, Any]:
    """Summarize raw, normalized, and region OCR evidence independently."""
    complete = sum(row.status == ProcessingStatus.COMPLETED for row in rows.values())
    raw_text_count = sum(row.raw_text is not None for row in rows.values())
    normalized_text_count = sum(row.normalized_text is not None for row in rows.values())
    frames_with_regions = sum(row.region_count > 0 for row in rows.values())
    raw_region_count = sum(row.region_count for row in rows.values())
    usable_region_count = sum(
        normalize_regions(
            tuple(
                OCRRegionResult(
                    text=region.text,
                    confidence=region.confidence,
                    x_min=region.x_min,
                    y_min=region.y_min,
                    x_max=region.x_max,
                    y_max=region.y_max,
                )
                for region in regions.get(frame_id, [])
            ),
            min_confidence=config.min_region_confidence,
        ).usable_region_count
        for frame_id in rows
    )
    quality_scores = [
        row.quality_score
        for row in rows.values()
        if row.status == ProcessingStatus.COMPLETED
    ]
    ratio = lambda count: count / input_count if input_count else 0.0

    return {
        "report_version": "ocr_report.v2",
        "artifact_version": config.artifact_version,
        "enrichment_version": config.enrichment_version,
        "dataset_version": config.dataset_version,
        "frame_store_id": frame_store_id,
        "input_parquet_path": str(path),
        "dataset_root": str(root),
        "backend": config.backend,
        "checkpoint": config.checkpoint,
        "resolved_revision": revision,
        "enabled": config.enabled,
        "device": config.device,
        "dtype": config.dtype,
        "batch_size": config.batch_size,
        "runtime_settings": asdict(config),
        "total_frames": input_count,
        "processed_frames": processed,
        "completed_frames": complete,
        "failed_frames": len(rows) - complete,
        "skipped_frames": skipped,
        "retried_frames": retried,
        "disabled_frames": disabled,
        "frames_with_raw_text": raw_text_count,
        "frames_with_normalized_text": normalized_text_count,
        "frames_with_regions": frames_with_regions,
        "raw_region_count": raw_region_count,
        "usable_region_count": usable_region_count,
        "mean_quality_score": (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        ),
        "raw_text_coverage_rate": ratio(raw_text_count),
        "normalized_text_coverage_rate": ratio(normalized_text_count),
        "region_coverage_rate": ratio(frames_with_regions),
        "failure_rate": ratio(len(rows) - complete),
        "error_counts": dict(
            Counter(item["exception_category"] for item in failures.values())
        ),
        "raw_output_available": any(
            item.get("raw_output") is not None for item in evidence.values()
        ),
        "raw_evidence": [evidence[key] for key in rows if key in evidence],
        "normalization_policy": (
            "Unicode NFC; collapse whitespace; confidence filter; require Unicode "
            "alphanumeric; case-insensitive ordered deduplication; newline join."
        ),
        "start_time": started.isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "elapsed_time_sec": elapsed,
        "manual_review": old.get(
            "manual_review",
            {"sample_count": 0, "status": "pending", "summary": "Human review pending."},
        ),
        "known_limitations": [
            "Coverage and quality heuristics are not OCR accuracy.",
            "Florence-2 has no calibrated OCR confidence.",
        ],
    }


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
    except Exception as error:  # noqa: BLE001
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
    """Process independent batches while containing per-frame failures."""
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
    from offline.artifact_readers import FrameArtifactReader

    started, began = datetime.now(timezone.utc), perf_counter()
    path, root = Path(frames_path), Path(dataset_root).expanduser().resolve()
    frames = [
        asdict(frame)
        for frame in FrameArtifactReader.load(path).iter_frames()
    ]
    order = [str(frame["frame_id"]) for frame in frames]
    if len(order) != len(set(order)):
        raise ValueError("input frames contain duplicate frame_id values")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "ocr_report.json"
    old = cast(dict[str, Any], read_json(report_path)) if report_path.exists() else {}
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


__all__ = ["build_ocr_report", "generate_ocr", "write_ocr_artifacts"]
