"""Estimate offline video-pipeline work from YouTube metadata.

This module reads the AIC media-info JSON files and optionally probes each
``watch_url`` with ``yt-dlp`` metadata-only requests. It does not download
video or image bytes. It calculates a deterministic 1-FPS frame workload and
turns measured stage throughputs into sequential wall-time and ThunderCompute
cost estimates.

The estimator intentionally does not invent model throughput. A cost profile
must provide measured rates for the stages that should receive time estimates;
without one, the script still reports durations and frame counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


DEFAULT_MEDIA_INFO_DIR = Path("data/media-info-aic25-b1/media-info")
DEFAULT_OUTPUT_DIR = Path("artifacts/cost_estimates/aic25_b1")
DEFAULT_HOURLY_RATE_USD = 0.38

STAGE_UNITS: dict[str, str] = {
    "download": "duration_s",
    "decode": "duration_s",
    "caption": "frames_1fps",
    "ocr": "frames_1fps",
    "objects": "frames_1fps",
    "asr": "duration_s",
    "visual_embedding": "frames_1fps",
    "context_embedding": "frames_1fps",
    "index": "frames_1fps",
}


@dataclass(frozen=True)
class MediaRecord:
    """Identify one media-info JSON and its locally recorded metadata."""

    video_id: str
    source_path: Path
    watch_url: str
    local_length_s: float | None


@dataclass(frozen=True)
class ProbeResult:
    """Store the metadata result without retaining downloaded media."""

    duration_s: float | None
    fps: float | None
    status: str
    error: str | None = None


@dataclass(frozen=True)
class VideoEstimate:
    """Resolve one video duration and its resulting 1-FPS workload."""

    video_id: str
    source_path: Path
    watch_url: str
    local_length_s: float | None
    duration_s: float | None
    duration_source: str
    fps: float | None
    frames_1fps: int | None
    probe_status: str
    probe_error: str | None = None


@dataclass(frozen=True)
class StageRate:
    """Describe measured work units processed per wall-clock second."""

    units_per_second: float
    workers: int = 1

    def __post_init__(self) -> None:
        """Reject rates that would make an estimate undefined or misleading."""

        if not math.isfinite(self.units_per_second) or self.units_per_second <= 0:
            raise ValueError("units_per_second must be a finite positive number")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")


@dataclass(frozen=True)
class CostProfile:
    """Hold measured stage rates and the ThunderCompute billing assumptions."""

    stages: Mapping[str, StageRate]
    hourly_rate_usd: float = DEFAULT_HOURLY_RATE_USD
    billed_vm_count: int = 1
    overhead_fraction: float = 0.0

    def __post_init__(self) -> None:
        """Validate billing inputs before they affect a reported estimate."""

        if not math.isfinite(self.hourly_rate_usd) or self.hourly_rate_usd < 0:
            raise ValueError("hourly_rate_usd must be finite and non-negative")
        if self.billed_vm_count < 1:
            raise ValueError("billed_vm_count must be at least 1")
        if not math.isfinite(self.overhead_fraction) or self.overhead_fraction < 0:
            raise ValueError("overhead_fraction must be finite and non-negative")


def _optional_nonnegative_number(value: Any, *, field_name: str) -> float | None:
    """Convert an optional numeric metadata field while preserving missingness."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric or null")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return converted


def load_media_records(media_info_dir: Path) -> list[MediaRecord]:
    """Load sorted video URLs and local duration fallbacks from JSON files.

    Missing ``length`` values remain unresolved so a successful metadata probe
    can still provide the duration. Invalid URLs or malformed durations fail
    early because silently dropping a video would undercount cost.
    """

    if not media_info_dir.is_dir():
        raise FileNotFoundError(f"Media-info directory does not exist: {media_info_dir}")

    records: list[MediaRecord] = []
    for source_path in sorted(media_info_dir.glob("*.json")):
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected an object in {source_path}")

        watch_url = payload.get("watch_url")
        if not isinstance(watch_url, str) or not watch_url.strip():
            raise ValueError(f"Missing watch_url in {source_path}")

        records.append(
            MediaRecord(
                video_id=source_path.stem,
                source_path=source_path,
                watch_url=watch_url,
                local_length_s=_optional_nonnegative_number(
                    payload.get("length"),
                    field_name=f"length in {source_path}",
                ),
            )
        )
    return records


def estimate_one_fps_frames(duration_s: float) -> int:
    """Count samples at timestamps 0, 1, 2, ... before video end."""

    checked_duration = _optional_nonnegative_number(
        duration_s,
        field_name="duration_s",
    )
    assert checked_duration is not None
    return math.ceil(checked_duration)


def resolve_video_estimate(
    record: MediaRecord,
    probe: ProbeResult,
) -> VideoEstimate:
    """Prefer probed duration and fall back to the local JSON duration."""

    if probe.duration_s is not None:
        duration_s = probe.duration_s
        duration_source = "yt-dlp"
    elif record.local_length_s is not None:
        duration_s = record.local_length_s
        duration_source = "local_json"
    else:
        duration_s = None
        duration_source = "unavailable"

    return VideoEstimate(
        video_id=record.video_id,
        source_path=record.source_path,
        watch_url=record.watch_url,
        local_length_s=record.local_length_s,
        duration_s=duration_s,
        duration_source=duration_source,
        fps=probe.fps,
        frames_1fps=(
            estimate_one_fps_frames(duration_s)
            if duration_s is not None
            else None
        ),
        probe_status=probe.status,
        probe_error=probe.error,
    )


def build_yt_dlp_command(url: str, *, executable: str = "yt-dlp") -> list[str]:
    """Build a metadata-only command that cannot write video output."""

    return [
        executable,
        "--dump-single-json",
        "--skip-download",
        "--no-playlist",
        "--no-warnings",
        "--no-cache-dir",
        url,
    ]


def _extract_probe_fps(payload: Mapping[str, Any]) -> float | None:
    """Read top-level FPS, then the first usable video-format FPS."""

    fps = _optional_nonnegative_number(payload.get("fps"), field_name="fps")
    if fps is not None and fps > 0:
        return fps

    formats = payload.get("formats")
    if not isinstance(formats, list):
        return None

    candidates: list[float] = []
    for media_format in formats:
        if not isinstance(media_format, dict):
            continue
        if media_format.get("vcodec") == "none":
            continue
        candidate = _optional_nonnegative_number(
            media_format.get("fps"),
            field_name="format fps",
        )
        if candidate is not None and candidate > 0:
            candidates.append(candidate)
    return max(candidates) if candidates else None


def parse_probe_output(stdout: str) -> ProbeResult:
    """Parse duration and optional FPS from one ``yt-dlp`` JSON response."""

    payload = json.loads(stdout)
    if not isinstance(payload, dict):
        raise ValueError("yt-dlp output must be a JSON object")
    duration_s = _optional_nonnegative_number(
        payload.get("duration"),
        field_name="duration",
    )
    if duration_s is None:
        raise ValueError("yt-dlp output did not contain a duration")
    return ProbeResult(
        duration_s=duration_s,
        fps=_extract_probe_fps(payload),
        status="probed",
    )


def probe_video_url(
    url: str,
    *,
    executable: str = "yt-dlp",
    timeout_s: float = 60.0,
    runner: Callable[..., Any] | None = None,
) -> ProbeResult:
    """Query one URL's metadata without downloading media bytes."""

    command = build_yt_dlp_command(url, executable=executable)
    run = runner or subprocess.run
    try:
        completed = run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ProbeResult(
            duration_s=None,
            fps=None,
            status="error",
            error=f"timeout after {timeout_s:g}s",
        )
    except OSError as error:
        return ProbeResult(
            duration_s=None,
            fps=None,
            status="error",
            error=str(error),
        )

    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout or "yt-dlp failed").strip()
        return ProbeResult(
            duration_s=None,
            fps=None,
            status="error",
            error=error,
        )

    try:
        return parse_probe_output(completed.stdout)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        return ProbeResult(
            duration_s=None,
            fps=None,
            status="error",
            error=f"invalid yt-dlp metadata: {error}",
        )


def load_probe_cache(cache_path: Path) -> dict[str, ProbeResult]:
    """Load URL probe results, returning an empty cache when none exists."""

    if not cache_path.exists():
        return {}
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Probe cache must be an object: {cache_path}")

    cache: dict[str, ProbeResult] = {}
    for url, raw_result in payload.items():
        if not isinstance(raw_result, dict):
            raise ValueError(f"Invalid probe cache entry for {url}")
        cache[url] = ProbeResult(
            duration_s=raw_result.get("duration_s"),
            fps=raw_result.get("fps"),
            status=str(raw_result.get("status", "error")),
            error=raw_result.get("error"),
        )
    return cache


def save_probe_cache(cache_path: Path, cache: Mapping[str, ProbeResult]) -> None:
    """Persist probe results after each URL so interrupted runs can resume."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {url: asdict(result) for url, result in sorted(cache.items())}
    cache_path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_cost_profile(profile_path: Path | None) -> CostProfile:
    """Load measured stage rates from a JSON profile, if one was supplied."""

    if profile_path is None:
        return CostProfile(stages={})
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Cost profile must be a JSON object")

    raw_stages = payload.get("stages", {})
    if not isinstance(raw_stages, dict):
        raise ValueError("Cost profile stages must be an object")

    stages: dict[str, StageRate] = {}
    for stage_name, raw_rate in raw_stages.items():
        if stage_name not in STAGE_UNITS:
            raise ValueError(f"Unknown stage in cost profile: {stage_name}")
        if not isinstance(raw_rate, dict):
            raise ValueError(f"Stage profile must be an object: {stage_name}")
        raw_throughput = raw_rate.get("units_per_second")
        if raw_throughput is None:
            raise ValueError(f"Missing units_per_second for stage: {stage_name}")
        stages[stage_name] = StageRate(
            units_per_second=float(raw_throughput),
            workers=int(raw_rate.get("workers", 1)),
        )

    return CostProfile(
        stages=stages,
        hourly_rate_usd=float(
            payload.get("hourly_rate_usd", DEFAULT_HOURLY_RATE_USD)
        ),
        billed_vm_count=int(payload.get("billed_vm_count", 1)),
        overhead_fraction=float(payload.get("overhead_fraction", 0.0)),
    )


def estimate_pipeline(
    estimates: Sequence[VideoEstimate],
    profile: CostProfile,
) -> dict[str, Any]:
    """Calculate workload, stage time, sequential wall time, and VM cost.

    Stages are deliberately modeled as sequential in this first estimator.
    ``workers`` changes effective throughput; ``billed_vm_count`` changes cost
    independently so one VM with multiple processes is not confused with
    multiple billed VMs.
    """

    total_duration_s = sum(
        estimate.duration_s
        for estimate in estimates
        if estimate.duration_s is not None
    )
    total_frames_1fps = sum(
        estimate.frames_1fps
        for estimate in estimates
        if estimate.frames_1fps is not None
    )
    unresolved_videos = sum(
        estimate.duration_s is None for estimate in estimates
    )

    totals = {
        "duration_s": total_duration_s,
        "frames_1fps": total_frames_1fps,
    }
    stage_estimates: dict[str, dict[str, Any]] = {}
    known_stage_seconds: list[float] = []
    for stage_name, unit_name in STAGE_UNITS.items():
        rate = profile.stages.get(stage_name)
        work_units = totals[unit_name]
        if rate is None:
            stage_estimates[stage_name] = {
                "work_units": work_units,
                "unit": unit_name,
                "units_per_second": None,
                "workers": None,
                "seconds": None,
                "hours": None,
            }
            continue

        seconds = work_units / (rate.units_per_second * rate.workers)
        known_stage_seconds.append(seconds)
        stage_estimates[stage_name] = {
            "work_units": work_units,
            "unit": unit_name,
            "units_per_second": rate.units_per_second,
            "workers": rate.workers,
            "seconds": seconds,
            "hours": seconds / 3600,
        }

    estimated_wall_hours: float | None
    estimated_cost_usd: float | None
    if known_stage_seconds:
        wall_seconds = sum(known_stage_seconds) * (1 + profile.overhead_fraction)
        estimated_wall_hours = wall_seconds / 3600
        estimated_cost_usd = (
            estimated_wall_hours
            * profile.billed_vm_count
            * profile.hourly_rate_usd
        )
    else:
        estimated_wall_hours = None
        estimated_cost_usd = None

    return {
        "video_count": len(estimates),
        "unique_watch_url_count": len({estimate.watch_url for estimate in estimates}),
        "unresolved_duration_video_count": unresolved_videos,
        "total_duration_s": total_duration_s,
        "total_duration_hours": total_duration_s / 3600,
        "total_frames_1fps": total_frames_1fps,
        "stage_estimates": stage_estimates,
        "estimated_wall_hours": estimated_wall_hours,
        "estimated_cost_usd": estimated_cost_usd,
        "execution_model": "sequential_stages",
        "overhead_fraction": profile.overhead_fraction,
        "hourly_rate_usd": profile.hourly_rate_usd,
        "billed_vm_count": profile.billed_vm_count,
    }


def _video_estimate_row(estimate: VideoEstimate) -> dict[str, Any]:
    """Convert a dataclass estimate into CSV/JSON-safe scalar values."""

    row = asdict(estimate)
    row["source_path"] = str(estimate.source_path)
    return row


def write_outputs(
    output_dir: Path,
    estimates: Sequence[VideoEstimate],
    summary: Mapping[str, Any],
) -> None:
    """Write per-video and aggregate estimates in inspectable formats."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_video_estimate_row(estimate) for estimate in estimates]
    fieldnames = [
        "video_id",
        "source_path",
        "watch_url",
        "local_length_s",
        "duration_s",
        "duration_source",
        "fps",
        "frames_1fps",
        "probe_status",
        "probe_error",
    ]
    with (output_dir / "video_estimates.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "video_estimates.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_estimator(
    *,
    media_info_dir: Path,
    output_dir: Path,
    profile: CostProfile,
    probe: bool = True,
    refresh_probe: bool = False,
    cache_path: Path | None = None,
    yt_dlp_executable: str = "yt-dlp",
    probe_timeout_s: float = 60.0,
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run the resumable metadata probe and write the estimate artifacts."""

    records = load_media_records(media_info_dir)
    resolved_cache_path = cache_path or (output_dir / "probe_cache.json")
    cache = load_probe_cache(resolved_cache_path) if probe else {}
    estimates: list[VideoEstimate] = []

    for record in records:
        if not probe:
            probe_result = ProbeResult(None, None, "disabled")
        elif not refresh_probe and record.watch_url in cache:
            probe_result = cache[record.watch_url]
        else:
            probe_result = probe_video_url(
                record.watch_url,
                executable=yt_dlp_executable,
                timeout_s=probe_timeout_s,
                runner=runner,
            )
            cache[record.watch_url] = probe_result
            save_probe_cache(resolved_cache_path, cache)
        estimates.append(resolve_video_estimate(record, probe_result))

    summary = estimate_pipeline(estimates, profile)
    write_outputs(output_dir, estimates, summary)
    if probe:
        save_probe_cache(resolved_cache_path, cache)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the metadata-only estimator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--media-info-dir",
        type=Path,
        default=DEFAULT_MEDIA_INFO_DIR,
        help="Directory containing one media-info JSON per video.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV, JSON, and probe-cache outputs.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="JSON file containing measured stage throughputs and billing inputs.",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="Use only local JSON lengths; never call yt-dlp.",
    )
    parser.add_argument(
        "--refresh-probe",
        action="store_true",
        help="Ignore cached URL probe results and query metadata again.",
    )
    parser.add_argument(
        "--yt-dlp-executable",
        default="yt-dlp",
        help="yt-dlp executable name or path.",
    )
    parser.add_argument(
        "--probe-timeout-s",
        type=float,
        default=60.0,
        help="Timeout for each metadata request.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the estimator CLI and print the aggregate summary."""

    args = build_parser().parse_args(argv)
    summary = run_estimator(
        media_info_dir=args.media_info_dir,
        output_dir=args.output_dir,
        profile=load_cost_profile(args.profile),
        probe=not args.no_probe,
        refresh_probe=args.refresh_probe,
        yt_dlp_executable=args.yt_dlp_executable,
        probe_timeout_s=args.probe_timeout_s,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
