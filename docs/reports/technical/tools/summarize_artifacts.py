#!/usr/bin/env python3
"""Stream finalized Parquet artifacts into report-ready corpus statistics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable

import pyarrow.parquet as pq

DEFAULT_BATCH_SIZE = 8_192
SAMPLE_PERIOD_MS = 1_000
_COLLECTION_PATTERN = re.compile(r"^(L\d+)_")


@dataclass(frozen=True)
class ArtifactSpec:
    """One finalized Parquet artifact included in the report summary."""

    key: str
    relative_path: str
    identity_column: str
    entity: str
    frame_aligned: bool = False


ARTIFACT_SPECS = (
    ArtifactSpec("canonical_frames", "frame_store/frames.parquet", "video_id", "frames"),
    ArtifactSpec("captions", "corpus/caption.parquet", "frame_id", "caption rows", True),
    ArtifactSpec(
        "translated_captions",
        "corpus/caption_vi.parquet",
        "frame_id",
        "translated caption rows",
        True,
    ),
    ArtifactSpec("ocr_frames", "corpus/ocr_frames.parquet", "frame_id", "OCR frame summaries", True),
    ArtifactSpec("ocr_regions", "corpus/ocr_regions.parquet", "region_id", "OCR regions"),
    ArtifactSpec("object_frames", "corpus/object_frames.parquet", "frame_id", "object frame summaries", True),
    ArtifactSpec(
        "object_detections",
        "corpus/object_detections.parquet",
        "frame_id",
        "object detections",
    ),
    ArtifactSpec("frame_context", "corpus/context.parquet", "frame_id", "FrameContext rows", True),
    ArtifactSpec(
        "asr_frame_enrichment",
        "enrichment/asr/frame_enrichment.parquet",
        "frame_id",
        "ASR frame-enrichment rows",
        True,
    ),
    ArtifactSpec(
        "visual_index_mapping",
        "indexes/visual/frame_mapping.parquet",
        "frame_id",
        "visual index mappings",
        True,
    ),
    ArtifactSpec(
        "context_index_mapping",
        "indexes/context/frame_mapping.parquet",
        "frame_id",
        "context index mappings",
        True,
    ),
    ArtifactSpec(
        "translated_context_index_mapping",
        "indexes/context_vi/frame_mapping.parquet",
        "frame_id",
        "translated context index mappings",
        True,
    ),
    ArtifactSpec(
        "bm25_document_mapping",
        "indexes/bm25/frame_mapping.parquet",
        "frame_id",
        "BM25 document mappings",
        True,
    ),
    ArtifactSpec(
        "asr_segments",
        "indexes/asr_segments/segment_mapping.parquet",
        "segment_id",
        "ASR segments",
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def _open_parquet(path: Path, required_column: str) -> pq.ParquetFile:
    if not path.is_file():
        raise FileNotFoundError(f"required finalized artifact not found: {path}")
    parquet = pq.ParquetFile(path)
    if required_column not in parquet.schema_arrow.names:
        raise ValueError(f"{path} does not contain required column {required_column!r}")
    return parquet


def stream_row_count(path: Path, column: str, batch_size: int) -> tuple[int, int, int]:
    """Count rows in bounded record batches and return rows, batches, footer rows."""

    parquet = _open_parquet(path, column)
    rows = 0
    batches = 0
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=[column],
        use_threads=False,
    ):
        rows += batch.num_rows
        batches += 1
    footer_rows = parquet.metadata.num_rows
    if rows != footer_rows:
        raise ValueError(f"streamed {rows} rows from {path}, but footer reports {footer_rows}")
    return rows, batches, footer_rows


def stream_frame_distribution(
    path: Path,
    batch_size: int,
) -> tuple[int, int, dict[str, dict[str, int]]]:
    """Aggregate frame/video counts while retaining only small counters and sets."""

    parquet = _open_parquet(path, "video_id")
    frame_count = 0
    batch_count = 0
    per_video: Counter[str] = Counter()
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=["video_id"],
        use_threads=False,
    ):
        batch_count += 1
        frame_count += batch.num_rows
        for value in batch.column(0).to_pylist():
            if not isinstance(value, str) or not value:
                raise ValueError(f"invalid video_id in {path}: {value!r}")
            per_video[value] += 1

    footer_rows = parquet.metadata.num_rows
    if frame_count != footer_rows:
        raise ValueError(
            f"streamed {frame_count} rows from {path}, but footer reports {footer_rows}"
        )

    collection_videos: defaultdict[str, int] = defaultdict(int)
    collection_frames: defaultdict[str, int] = defaultdict(int)
    for video_id, count in per_video.items():
        match = _COLLECTION_PATTERN.match(video_id)
        if match is None:
            raise ValueError(f"video_id has no collection prefix: {video_id!r}")
        collection = match.group(1)
        collection_videos[collection] += 1
        collection_frames[collection] += count

    collections = {
        collection: {
            "videos": collection_videos[collection],
            "frames": collection_frames[collection],
        }
        for collection in sorted(collection_frames, key=lambda value: int(value[1:]))
    }
    return frame_count, batch_count, collections


def _manifest_validation(
    artifacts_root: Path,
    frame_count: int,
    video_count: int,
) -> dict[str, Any]:
    path = artifacts_root / "frame_store/manifest.json"
    if not path.is_file():
        return {"path": "frame_store/manifest.json", "present": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_frames = int(payload["frame_count"])
    manifest_videos = int(payload["video_count"])
    if (manifest_frames, manifest_videos) != (frame_count, video_count):
        raise ValueError(
            "streamed frame distribution disagrees with frame-store manifest: "
            f"stream={frame_count} frames/{video_count} videos, "
            f"manifest={manifest_frames} frames/{manifest_videos} videos"
        )
    return {
        "path": "frame_store/manifest.json",
        "present": True,
        "frame_count_matches": True,
        "video_count_matches": True,
    }


def summarize_artifacts(
    artifacts_root: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    specs: Iterable[ArtifactSpec] = ARTIFACT_SPECS,
) -> dict[str, Any]:
    """Build report statistics without materializing any complete Parquet table."""

    root = artifacts_root.resolve()
    spec_list = tuple(specs)
    canonical = next((spec for spec in spec_list if spec.key == "canonical_frames"), None)
    if canonical is None:
        raise ValueError("artifact registry must include canonical_frames")

    frame_count, frame_batches, collections = stream_frame_distribution(
        root / canonical.relative_path,
        batch_size,
    )
    video_count = sum(item["videos"] for item in collections.values())
    rows: dict[str, dict[str, Any]] = {
        canonical.key: {
            "entity": canonical.entity,
            "path": canonical.relative_path,
            "rows": frame_count,
            "streamed_batches": frame_batches,
            "footer_rows": frame_count,
        }
    }

    for spec in spec_list:
        if spec is canonical:
            continue
        count, batches, footer_rows = stream_row_count(
            root / spec.relative_path,
            spec.identity_column,
            batch_size,
        )
        if spec.frame_aligned and count != frame_count:
            raise ValueError(
                f"frame-aligned artifact {spec.relative_path} has {count} rows; "
                f"expected {frame_count}"
            )
        rows[spec.key] = {
            "entity": spec.entity,
            "path": spec.relative_path,
            "rows": count,
            "streamed_batches": batches,
            "footer_rows": footer_rows,
        }

    collection_rows = []
    for collection, values in collections.items():
        frames = values["frames"]
        collection_rows.append(
            {
                "collection": collection,
                "videos": values["videos"],
                "frames": frames,
                "sampled_timeline_hours": round(
                    frames * SAMPLE_PERIOD_MS / 3_600_000,
                    1,
                ),
            }
        )

    return {
        "schema_version": "hcmai-report-artifact-statistics-v1",
        "method": {
            "reader": "pyarrow.parquet.ParquetFile.iter_batches",
            "batch_size": batch_size,
            "use_threads": False,
            "scope": "finalized artifacts only; batch shards excluded",
            "sample_period_ms": SAMPLE_PERIOD_MS,
        },
        "summary": {
            "videos": video_count,
            "frames": frame_count,
            "sampled_timeline_hours": round(
                frame_count * SAMPLE_PERIOD_MS / 3_600_000,
                1,
            ),
            "ocr_regions": rows["ocr_regions"]["rows"],
            "object_detections": rows["object_detections"]["rows"],
            "asr_segments": rows["asr_segments"]["rows"],
        },
        "collections": collection_rows,
        "artifacts": rows,
        "manifest_validation": _manifest_validation(root, frame_count, video_count),
    }


def _format_integer(value: int) -> str:
    return f"{value:,}"


def render_tex(statistics: dict[str, Any]) -> str:
    """Render deterministic LaTeX macros and data rows."""

    summary = statistics["summary"]
    artifacts = statistics["artifacts"]
    collections = statistics["collections"]
    max_frames = max(item["frames"] for item in collections)
    lines = [
        "% Generated by tools/summarize_artifacts.py; do not edit manually.",
        f"\\newcommand{{\\ArtifactVideoCount}}{{{_format_integer(summary['videos'])}}}",
        f"\\newcommand{{\\ArtifactFrameCount}}{{{_format_integer(summary['frames'])}}}",
        f"\\newcommand{{\\ArtifactSampledHours}}{{{summary['sampled_timeline_hours']:.1f}}}",
        f"\\newcommand{{\\ArtifactCaptionCount}}{{{_format_integer(artifacts['captions']['rows'])}}}",
        f"\\newcommand{{\\ArtifactOCRFrameCount}}{{{_format_integer(artifacts['ocr_frames']['rows'])}}}",
        f"\\newcommand{{\\ArtifactOCRRegionCount}}{{{_format_integer(summary['ocr_regions'])}}}",
        f"\\newcommand{{\\ArtifactObjectFrameCount}}{{{_format_integer(artifacts['object_frames']['rows'])}}}",
        f"\\newcommand{{\\ArtifactObjectDetectionCount}}{{{_format_integer(summary['object_detections'])}}}",
        f"\\newcommand{{\\ArtifactContextCount}}{{{_format_integer(artifacts['frame_context']['rows'])}}}",
        f"\\newcommand{{\\ArtifactASRSegmentCount}}{{{_format_integer(summary['asr_segments'])}}}",
        "\\newcommand{\\ArtifactCollectionBars}{%",
    ]
    top_y = 45
    for index, item in enumerate(collections):
        bar_width = 75 * item["frames"] / max_frames
        lines.append(
            "  \\ArtifactCollectionBar"
            f"{{{top_y - 5 * index}}}"
            f"{{{item['collection']}}}"
            f"{{{item['videos']}}}"
            f"{{{bar_width:.4f}}}"
            f"{{{_format_integer(item['frames'])}}}%"
        )
    lines.extend(("}", ""))
    return "\n".join(lines)


def render_json(statistics: dict[str, Any]) -> str:
    """Render deterministic, human-reviewable JSON."""

    return json.dumps(statistics, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _stage_text(destination: Path, text: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def write_outputs(statistics: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Stage complete outputs before atomically replacing either destination."""

    json_path = output_dir / "artifact-statistics.json"
    tex_path = output_dir / "artifact-statistics.tex"
    staged_json = _stage_text(json_path, render_json(statistics))
    staged_tex = _stage_text(tex_path, render_tex(statistics))
    try:
        os.replace(staged_json, json_path)
        os.replace(staged_tex, tex_path)
    finally:
        staged_json.unlink(missing_ok=True)
        staged_tex.unlink(missing_ok=True)
    return json_path, tex_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    statistics = summarize_artifacts(
        args.artifacts_root,
        batch_size=args.batch_size,
    )
    json_path, tex_path = write_outputs(statistics, args.output_dir)
    print(
        f"Wrote {json_path} and {tex_path}: "
        f"{statistics['summary']['videos']:,} videos, "
        f"{statistics['summary']['frames']:,} frames"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
