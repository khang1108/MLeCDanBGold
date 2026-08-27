"""Prepare deterministic metadata inputs for the native custom extractor.

This module owns organizer media-info validation, native JSONL generation, and
the normalized extraction configuration hash. It intentionally does not invoke
yt-dlp, FFmpeg, models, or any remote compute service.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from hcmai.common.utils.io import atomic_write


_EXTRACTOR_VERSION = "hcmai-keyframes-extractor/0.1.0"
_NATIVE_CONFIG_DEFAULTS: dict[str, object] = {
    "sample_period_ms": 1_000,
    "durable_long_edge": 1_024,
    "durable_jpeg_quality": 92,
    "enrichment_jpeg_quality": 95,
    "write_enrichment_images": True,
    "extractor_version": _EXTRACTOR_VERSION,
}


def _canonical_json(value: object) -> str:
    """Serialize a value in the stable representation used for config hashing.

    Args:
        value: JSON-compatible value whose ordering and whitespace must be
            deterministic.

    Returns:
        Compact canonical JSON with sorted keys and UTF-8-safe characters.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _require_non_blank_string(value: object, field_name: str, source: Path) -> str:
    """Return a normalized non-blank string from one media-info field.

    Args:
        value: Candidate JSON field value.
        field_name: Field name used in an actionable validation error.
        source: Media-info source file that supplied the value.

    Returns:
        Trimmed non-blank string.

    Raises:
        ValueError: If value is not a string or becomes blank after trimming.
    """

    if not isinstance(value, str) or not (normalized := value.strip()):
        raise ValueError(f"{source} has a blank or invalid {field_name}")
    return normalized


def _require_metadata_length(value: object, source: Path) -> int:
    """Validate a whole, non-negative organizer duration estimate.

    Args:
        value: Candidate ``length`` value loaded from media-info JSON.
        source: Media-info source file that supplied the value.

    Returns:
        Exact non-negative integer duration in seconds.

    Raises:
        ValueError: If length is boolean, non-integral, or negative.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{source} length must be an integer")
    if value < 0:
        raise ValueError(f"{source} length must be non-negative")
    return value


def _load_media_record(path: Path) -> dict[str, object]:
    """Load one JSON object without accepting an ambiguous non-object root.

    Args:
        path: Existing organizer media-info JSON file.

    Returns:
        Parsed JSON object as a mutable plain dictionary.

    Raises:
        ValueError: If the document is invalid JSON or does not contain an object.
    """

    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid media-info JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"media-info root must be an object: {path}")
    return value


def build_native_input_manifest(
    media_info_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Build a sorted, strict metadata-only JSONL manifest for native extraction.

    Args:
        media_info_dir: Directory containing one ``{video_id}.json`` organizer
            media-info record per source video.
        output_path: JSONL path that receives sorted native input rows through a
            sibling temporary file and atomic replacement.

    Returns:
        The final manifest path.

    Raises:
        FileNotFoundError: If ``media_info_dir`` does not exist.
        NotADirectoryError: If ``media_info_dir`` is not a directory.
        ValueError: If records are malformed, IDs/URLs repeat, or no JSON files
            are present.
    """

    source_dir = Path(media_info_dir)
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)

    source_paths = sorted(
        path for path in source_dir.iterdir() if path.is_file() and path.suffix == ".json"
    )
    if not source_paths:
        raise ValueError(f"media-info directory contains no JSON files: {source_dir}")

    rows: list[dict[str, object]] = []
    seen_video_ids: set[str] = set()
    seen_watch_urls: set[str] = set()
    for source_path in source_paths:
        video_id = _require_non_blank_string(
            source_path.stem,
            "video_id filename stem",
            source_path,
        )
        if video_id in seen_video_ids:
            raise ValueError(f"duplicate video_id: {video_id}")
        record = _load_media_record(source_path)
        watch_url = _require_non_blank_string(
            record.get("watch_url"),
            "watch_url",
            source_path,
        )
        if watch_url in seen_watch_urls:
            raise ValueError(f"duplicate watch_url: {watch_url}")
        metadata_length_s = _require_metadata_length(record.get("length"), source_path)

        seen_video_ids.add(video_id)
        seen_watch_urls.add(watch_url)
        rows.append(
            {
                "video_id": video_id,
                "watch_url": watch_url,
                "metadata_length_s": metadata_length_s,
            }
        )

    destination = Path(output_path)

    def write_manifest(temporary_path: Path) -> None:
        """Write one complete deterministic JSONL document to a temporary sibling.

        Args:
            temporary_path: Temporary file path provided by ``atomic_write``.

        Returns:
            None; writes each sorted manifest record once.
        """

        temporary_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    atomic_write(destination, write_manifest)
    return destination


def write_extraction_config(
    config_path: str | Path,
    *,
    run_root: str | Path,
    native_executable: str | Path,
    frame_store_id: str,
    yt_dlp_binary: str | Path,
    yt_dlp_cookies_path: str | Path | None = None,
    yt_dlp_js_runtime: str | None = None,
) -> Path:
    """Write native extraction settings and a reproducible configuration hash.

    Args:
        config_path: JSON destination consumed by the C++ extractor.
        run_root: Isolated lifecycle root recorded for operator provenance.
        native_executable: Native binary path recorded for operator provenance.
        frame_store_id: Separate custom-corpus lineage identifier.
        yt_dlp_binary: Explicit downloader executable path or command name.
        yt_dlp_cookies_path: Optional Netscape cookie file path passed to yt-dlp.
        yt_dlp_js_runtime: Optional yt-dlp runtime token such as ``deno`` or ``node``.

    Returns:
        The final JSON config path.

    Raises:
        ValueError: If required path-like values or ``frame_store_id`` are blank.
    """

    normalized_run_root = _require_non_blank_string(
        str(run_root),
        "run_root",
        Path(config_path),
    )
    normalized_native_executable = _require_non_blank_string(
        str(native_executable),
        "native_executable",
        Path(config_path),
    )
    normalized_frame_store_id = _require_non_blank_string(
        frame_store_id,
        "frame_store_id",
        Path(config_path),
    )
    normalized_yt_dlp_binary = _require_non_blank_string(
        str(yt_dlp_binary),
        "yt_dlp_binary",
        Path(config_path),
    )
    payload = {
        **_NATIVE_CONFIG_DEFAULTS,
        "yt_dlp_binary": normalized_yt_dlp_binary,
        "run_root": normalized_run_root,
        "native_executable": normalized_native_executable,
        "frame_store_id": normalized_frame_store_id,
    }
    # Authentication and runtime locations are operational inputs, not frame
    # lineage. Excluding them from the hash permits a failed download to retry
    # with refreshed cookies without invalidating its canonical extraction state.
    config_hash = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    normalized_cookies = (
        _require_non_blank_string(
            str(yt_dlp_cookies_path),
            "yt_dlp_cookies_path",
            Path(config_path),
        )
        if yt_dlp_cookies_path is not None
        else None
    )
    normalized_js_runtime = (
        _require_non_blank_string(
            yt_dlp_js_runtime,
            "yt_dlp_js_runtime",
            Path(config_path),
        )
        if yt_dlp_js_runtime is not None
        else None
    )
    config = {
        **payload,
        "yt_dlp_cookies_path": normalized_cookies,
        "yt_dlp_js_runtime": normalized_js_runtime,
        "config_hash": config_hash,
    }
    destination = Path(config_path)

    def write_config(temporary_path: Path) -> None:
        """Serialize the complete normalized config through the atomic writer.

        Args:
            temporary_path: Temporary file path provided by ``atomic_write``.

        Returns:
            None; writes canonical JSON plus one trailing newline.
        """

        temporary_path.write_text(
            _canonical_json(config) + "\n",
            encoding="utf-8",
        )

    atomic_write(destination, write_config)
    return destination


__all__ = ["build_native_input_manifest", "write_extraction_config"]
