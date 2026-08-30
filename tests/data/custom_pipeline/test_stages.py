"""Tests for isolated batch stage execution with CUDA OOM backoff.

Uses a tiny fake Python subprocess (not a real model) to simulate success,
OOM recovery, OOM exhaustion at batch size 1, non-OOM failure, and missing
declared output, without depending on any real CLI or GPU.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from offline.ingestion.custom_pipeline.config import SchedulingConfig
from offline.ingestion.custom_pipeline.stages import (
    StageCommand,
    StageExecutionError,
    run_batch_stages,
    run_stage,
)

_FAKE_STAGE_SCRIPT = textwrap.dedent(
    """
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--oom-above", type=int, default=None)
    parser.add_argument("--fail-message", default=None)
    args = parser.parse_args()

    if args.fail_message is not None:
        print(args.fail_message, file=sys.stderr)
        sys.exit(1)
    if args.oom_above is not None and args.batch_size > args.oom_above:
        print("CUDA out of memory: tried to allocate too much", file=sys.stderr)
        sys.exit(1)

    Path(args.output).write_text(f"batch_size={args.batch_size}")
    sys.exit(0)
    """
)


@pytest.fixture()
def fake_stage(tmp_path: Path) -> Path:
    script = tmp_path / "fake_stage.py"
    script.write_text(_FAKE_STAGE_SCRIPT)
    return script


def _argv(script: Path, output: Path, *, oom_above: int | None = None, fail_message: str | None = None) -> tuple[str, ...]:
    argv = [sys.executable, str(script), "--output", str(output)]
    if oom_above is not None:
        argv += ["--oom-above", str(oom_above)]
    if fail_message is not None:
        argv += ["--fail-message", fail_message]
    return tuple(argv)


# ---------------------------------------------------------------------------
# run_stage
# ---------------------------------------------------------------------------


def test_run_stage_succeeds_on_first_attempt(fake_stage: Path, tmp_path: Path) -> None:
    output = tmp_path / "out.txt"
    command = StageCommand(
        name="caption",
        argv=_argv(fake_stage, output),
        initial_batch_size=8,
        output_path=str(output),
    )
    result = run_stage(command)
    assert result.succeeded and result.effective_batch_size == 8
    assert len(result.attempts) == 1
    assert output.read_text() == "batch_size=8"


def test_run_stage_recovers_via_oom_backoff_8_4_2(fake_stage: Path, tmp_path: Path) -> None:
    output = tmp_path / "out.txt"
    command = StageCommand(
        name="caption",
        argv=_argv(fake_stage, output, oom_above=2),
        initial_batch_size=8,
        output_path=str(output),
    )
    result = run_stage(command)
    assert result.succeeded and result.effective_batch_size == 2
    assert [attempt.batch_size for attempt in result.attempts] == [8, 4, 2]
    assert [attempt.recognized_oom for attempt in result.attempts] == [True, True, False]


def test_run_stage_raises_when_oom_persists_at_batch_size_one(
    fake_stage: Path, tmp_path: Path
) -> None:
    output = tmp_path / "out.txt"
    command = StageCommand(
        name="caption",
        argv=_argv(fake_stage, output, oom_above=0),
        initial_batch_size=2,
        output_path=str(output),
    )
    with pytest.raises(StageExecutionError, match="minimum batch size 1"):
        run_stage(command)


def test_run_stage_raises_immediately_on_non_oom_failure(fake_stage: Path, tmp_path: Path) -> None:
    output = tmp_path / "out.txt"
    command = StageCommand(
        name="caption",
        argv=_argv(fake_stage, output, fail_message="disk full: cannot write output"),
        initial_batch_size=8,
        output_path=str(output),
    )
    with pytest.raises(StageExecutionError, match="disk full"):
        run_stage(command)


def test_run_stage_bounds_stderr_in_diagnostics(fake_stage: Path, tmp_path: Path) -> None:
    output = tmp_path / "out.txt"
    long_message = "x" * 10_000
    command = StageCommand(
        name="caption",
        argv=_argv(fake_stage, output, fail_message=long_message),
        initial_batch_size=8,
        output_path=str(output),
    )
    with pytest.raises(StageExecutionError) as excinfo:
        run_stage(command)
    assert len(str(excinfo.value)) < 1000


def test_run_stage_propagates_cpu_thread_environment(fake_stage: Path, tmp_path: Path) -> None:
    output = tmp_path / "env_check.txt"
    script = tmp_path / "env_stage.py"
    script.write_text(
        textwrap.dedent(
            """
            import argparse, os, sys
            parser = argparse.ArgumentParser()
            parser.add_argument("--batch-size", type=int, required=True)
            parser.add_argument("--output", required=True)
            args = parser.parse_args()
            with open(args.output, "w") as handle:
                handle.write(os.environ.get("OMP_NUM_THREADS", "unset"))
            sys.exit(0)
            """
        )
    )
    command = StageCommand(
        name="index_build",
        argv=(sys.executable, str(script), "--output", str(output)),
        initial_batch_size=1,
        output_path=str(output),
        cpu_threads=6,
    )
    run_stage(command)
    assert output.read_text() == "6"


# ---------------------------------------------------------------------------
# run_batch_stages
# ---------------------------------------------------------------------------


def test_run_batch_stages_executes_in_strict_order(fake_stage: Path, tmp_path: Path) -> None:
    order_log = tmp_path / "order.log"
    script = tmp_path / "ordered_stage.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import argparse, sys
            parser = argparse.ArgumentParser()
            parser.add_argument("--batch-size", type=int, required=True)
            parser.add_argument("--output", required=True)
            parser.add_argument("--name", required=True)
            args = parser.parse_args()
            with open({str(order_log)!r}, "a") as handle:
                handle.write(args.name + "\\n")
            with open(args.output, "w") as handle:
                handle.write("done")
            sys.exit(0)
            """
        )
    )
    stage_names = ["extraction", "caption", "ocr", "objects", "context", "visual_embedding"]
    commands = [
        StageCommand(
            name=stage_name,
            argv=(
                sys.executable,
                str(script),
                "--output",
                str(tmp_path / f"{stage_name}.out"),
                "--name",
                stage_name,
            ),
            initial_batch_size=1,
            output_path=str(tmp_path / f"{stage_name}.out"),
        )
        for stage_name in stage_names
    ]
    results = run_batch_stages(commands)
    assert [result.name for result in results] == stage_names
    assert order_log.read_text().splitlines() == stage_names


def test_run_batch_stages_raises_on_missing_declared_output(tmp_path: Path) -> None:
    script = tmp_path / "no_output_stage.py"
    script.write_text(
        textwrap.dedent(
            """
            import argparse, sys
            parser = argparse.ArgumentParser()
            parser.add_argument("--batch-size", type=int, required=True)
            parser.add_argument("--output", required=True)
            args = parser.parse_args()
            sys.exit(0)  # succeeds without writing the declared output
            """
        )
    )
    command = StageCommand(
        name="objects",
        argv=(sys.executable, str(script), "--output", str(tmp_path / "missing.out")),
        initial_batch_size=1,
        output_path=str(tmp_path / "missing.out"),
    )
    with pytest.raises(StageExecutionError, match="did not produce"):
        run_batch_stages([command])


def test_run_batch_stages_rejects_cpu_oversubscription(fake_stage: Path, tmp_path: Path) -> None:
    output = tmp_path / "out.txt"
    command = StageCommand(
        name="index_build",
        argv=_argv(fake_stage, output),
        initial_batch_size=1,
        output_path=str(output),
        cpu_threads=12,
    )
    with pytest.raises(ValueError, match="exceeds available_cpus"):
        run_batch_stages([command], scheduling=SchedulingConfig(available_cpus=6))
