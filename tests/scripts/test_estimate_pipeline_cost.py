"""Tests for the metadata-only pipeline cost estimator."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.estimate_pipeline_cost import (
    CostProfile,
    MediaRecord,
    ProbeResult,
    StageRate,
    build_yt_dlp_command,
    estimate_one_fps_frames,
    estimate_pipeline,
    load_cost_profile,
    load_media_records,
    load_probe_cache,
    resolve_video_estimate,
    run_estimator,
    save_probe_cache,
)


def test_load_media_records_reads_urls_and_local_lengths(tmp_path: Path) -> None:
    """Read one deterministic record per media-info JSON file."""

    media_info = tmp_path / "media-info"
    media_info.mkdir()
    (media_info / "L21_V002.json").write_text(
        json.dumps(
            {
                "length": 12,
                "watch_url": "https://youtube.com/watch?v=two",
            }
        ),
        encoding="utf-8",
    )
    (media_info / "L21_V001.json").write_text(
        json.dumps(
            {
                "length": 8.5,
                "watch_url": "https://youtube.com/watch?v=one",
            }
        ),
        encoding="utf-8",
    )

    records = load_media_records(media_info)

    assert records == [
        MediaRecord(
            video_id="L21_V001",
            source_path=media_info / "L21_V001.json",
            watch_url="https://youtube.com/watch?v=one",
            local_length_s=8.5,
        ),
        MediaRecord(
            video_id="L21_V002",
            source_path=media_info / "L21_V002.json",
            watch_url="https://youtube.com/watch?v=two",
            local_length_s=12.0,
        ),
    ]


def test_resolve_video_estimate_prefers_probed_duration() -> None:
    """Use fresh YouTube metadata when it is available."""

    record = MediaRecord(
        video_id="L21_V001",
        source_path=Path("L21_V001.json"),
        watch_url="https://youtube.com/watch?v=one",
        local_length_s=8.0,
    )

    estimate = resolve_video_estimate(
        record,
        ProbeResult(duration_s=8.75, fps=29.97, status="probed"),
    )

    assert estimate.duration_s == 8.75
    assert estimate.duration_source == "yt-dlp"
    assert estimate.frames_1fps == 9
    assert estimate.fps == 29.97


def test_resolve_video_estimate_falls_back_to_local_length() -> None:
    """Keep a usable local duration when metadata probing fails."""

    record = MediaRecord(
        video_id="L21_V001",
        source_path=Path("L21_V001.json"),
        watch_url="https://youtube.com/watch?v=one",
        local_length_s=8.0,
    )

    estimate = resolve_video_estimate(
        record,
        ProbeResult(duration_s=None, fps=None, status="error", error="timeout"),
    )

    assert estimate.duration_s == 8.0
    assert estimate.duration_source == "local_json"
    assert estimate.frames_1fps == 8
    assert estimate.probe_status == "error"
    assert estimate.probe_error == "timeout"


@pytest.mark.parametrize(
    ("duration_s", "expected"),
    [(0.0, 0), (1.0, 1), (1.01, 2), (8.75, 9)],
)
def test_estimate_one_fps_frames_rounds_up_duration(
    duration_s: float,
    expected: int,
) -> None:
    """Count samples at timestamps 0, 1, 2, ... before video end."""

    assert estimate_one_fps_frames(duration_s) == expected


def test_build_yt_dlp_command_is_metadata_only() -> None:
    """Make the no-video-download boundary visible in the command."""

    command = build_yt_dlp_command(
        "https://youtube.com/watch?v=one",
        executable="yt-dlp",
    )

    assert "--skip-download" in command
    assert "--dump-single-json" in command
    assert "--no-playlist" in command
    assert "--no-cache-dir" in command
    assert "-o" not in command


def test_estimate_pipeline_calculates_stage_time_and_cost() -> None:
    """Convert measured stage rates into sequential wall time and VM cost."""

    records = [
        MediaRecord(
            video_id="L21_V001",
            source_path=Path("L21_V001.json"),
            watch_url="https://youtube.com/watch?v=one",
            local_length_s=8.75,
        ),
        MediaRecord(
            video_id="L21_V002",
            source_path=Path("L21_V002.json"),
            watch_url="https://youtube.com/watch?v=two",
            local_length_s=2.0,
        ),
    ]
    profile = CostProfile(
        stages={
            "caption": StageRate(units_per_second=5.0),
            "asr": StageRate(units_per_second=2.0),
        },
        hourly_rate_usd=0.38,
        billed_vm_count=2,
        overhead_fraction=0.10,
    )

    summary = estimate_pipeline(
        [resolve_video_estimate(record, ProbeResult(None, None, "disabled"))
         for record in records],
        profile,
    )

    assert summary["video_count"] == 2
    assert summary["total_duration_s"] == pytest.approx(10.75)
    assert summary["total_frames_1fps"] == 11
    assert summary["stage_estimates"]["caption"]["work_units"] == 11
    assert summary["stage_estimates"]["caption"]["seconds"] == pytest.approx(2.2)
    assert summary["stage_estimates"]["asr"]["work_units"] == pytest.approx(10.75)
    assert summary["stage_estimates"]["asr"]["seconds"] == pytest.approx(5.375)
    assert summary["estimated_wall_hours"] == pytest.approx(
        (2.2 + 5.375) * 1.10 / 3600
    )
    assert summary["estimated_cost_usd"] == pytest.approx(
        (2.2 + 5.375) * 1.10 / 3600 * 2 * 0.38
    )


def test_probe_result_parsing_accepts_duration_and_fps() -> None:
    """Parse the small metadata subset needed by the estimator."""

    from scripts.estimate_pipeline_cost import parse_probe_output

    result = parse_probe_output(
        json.dumps({"duration": 12.5, "fps": 29.97}),
    )

    assert result == ProbeResult(
        duration_s=12.5,
        fps=29.97,
        status="probed",
    )


def test_probe_runner_failure_is_returned_as_a_record() -> None:
    """Turn a failed metadata command into a resumable fallback result."""

    from scripts.estimate_pipeline_cost import probe_video_url

    def runner(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=1, stdout="", stderr="network down")

    result = probe_video_url(
        "https://youtube.com/watch?v=one",
        runner=runner,
    )

    assert result.status == "error"
    assert result.duration_s is None
    assert result.error == "network down"


def test_probe_cache_round_trips_success_and_failure(tmp_path: Path) -> None:
    """Persist both successful and failed probes for resumable runs."""

    cache_path = tmp_path / "probe_cache.json"
    expected = {
        "https://youtube.com/watch?v=one": ProbeResult(12.5, 29.97, "probed"),
        "https://youtube.com/watch?v=two": ProbeResult(
            None,
            None,
            "error",
            "network down",
        ),
    }

    save_probe_cache(cache_path, expected)

    assert load_probe_cache(cache_path) == expected


def test_load_cost_profile_reads_rates_and_billing(tmp_path: Path) -> None:
    """Keep measured throughput and billing assumptions reproducible."""

    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "hourly_rate_usd": 0.38,
                "billed_vm_count": 2,
                "overhead_fraction": 0.15,
                "stages": {
                    "caption": {"units_per_second": 4.0, "workers": 2},
                    "asr": {"units_per_second": 8.0},
                },
            }
        ),
        encoding="utf-8",
    )

    profile = load_cost_profile(profile_path)

    assert profile.hourly_rate_usd == 0.38
    assert profile.billed_vm_count == 2
    assert profile.overhead_fraction == 0.15
    assert profile.stages["caption"] == StageRate(4.0, 2)
    assert profile.stages["asr"] == StageRate(8.0, 1)


def test_run_estimator_writes_outputs_without_network_probe(tmp_path: Path) -> None:
    """Support a deterministic local run that never invokes yt-dlp."""

    media_info = tmp_path / "media-info"
    media_info.mkdir()
    (media_info / "L21_V001.json").write_text(
        json.dumps(
            {
                "length": 8.5,
                "watch_url": "https://youtube.com/watch?v=one",
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"

    summary = run_estimator(
        media_info_dir=media_info,
        output_dir=output_dir,
        profile=CostProfile(stages={}),
        probe=False,
    )

    assert summary["total_frames_1fps"] == 9
    assert (output_dir / "video_estimates.csv").is_file()
    assert (output_dir / "video_estimates.json").is_file()
    assert (output_dir / "summary.json").is_file()
    assert not (output_dir / "probe_cache.json").exists()
