"""Run isolated local batch stages with CUDA OOM backoff and CPU limits.

Every stage is an explicit external subprocess (argv, never shell-
interpolated) so exactly one GPU-heavy process is active at a time and its
VRAM is returned to the OS before the next stage starts. A recognized CUDA
OOM diagnostic halves only the model batch size down to a floor of one;
unrelated failures are never mislabeled or silently retried as OOM.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from hcmai.common.utils.logging import get_logger
from offline.ingestion.custom_pipeline.config import SchedulingConfig

logger = get_logger(__name__)

# Only well-known CUDA/cuBLAS allocation failures are treated as OOM; any
# other failure must propagate immediately instead of being retried.
_OOM_MARKERS = (
    "cuda out of memory",
    "cublas_status_alloc_failed",
    "cudaerrormemoryallocation",
    "out of memory",
)
_MAX_DIAGNOSTIC_CHARS = 2000


class StageExecutionError(RuntimeError):
    """Raised when a stage subprocess fails for a reason other than OOM."""


@dataclass(frozen=True)
class StageCommand:
    """One external batch-stage subprocess and its resource envelope."""

    name: str
    argv: tuple[str, ...]
    initial_batch_size: int
    output_path: str
    image_workers: int = 3
    cpu_threads: int | None = None
    batch_size_flag: str = "--batch-size"

    def __post_init__(self) -> None:
        if self.initial_batch_size < 1:
            raise ValueError("initial_batch_size must be positive")
        if self.image_workers < 1:
            raise ValueError("image_workers must be positive")


@dataclass(frozen=True)
class StageAttempt:
    """One executed attempt of a stage, whether it succeeded or backed off."""

    attempt: int
    batch_size: int
    elapsed_sec: float
    recognized_oom: bool
    succeeded: bool
    diagnostic: str | None = None


@dataclass(frozen=True)
class StageResult:
    """Complete outcome of one stage after success or exhausted backoff."""

    name: str
    succeeded: bool
    effective_batch_size: int
    attempts: tuple[StageAttempt, ...] = field(default_factory=tuple)


def _is_recognized_oom(diagnostic: str) -> bool:
    """Match only well-known CUDA/cuBLAS out-of-memory diagnostics."""

    lowered = diagnostic.lower()
    return any(marker in lowered for marker in _OOM_MARKERS)


def _build_argv_with_batch_size(command: StageCommand, batch_size: int) -> tuple[str, ...]:
    return (*command.argv, command.batch_size_flag, str(batch_size))


def _stage_environment(command: StageCommand) -> dict[str, str] | None:
    """Build a stage-scoped environment carrying only CPU thread limits.

    Only numeric thread-count variables are added; the parent environment
    (including any secrets already present there) is otherwise propagated
    unchanged and nothing new is logged from it.
    """

    if command.cpu_threads is None:
        return None
    env = dict(os.environ)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[variable] = str(command.cpu_threads)
    return env


def run_stage(command: StageCommand, *, max_attempts: int = 10) -> StageResult:
    """Run one stage, halving its model batch size on recognized CUDA OOM.

    The batch size never drops below 1.

    Raises:
        StageExecutionError: If the subprocess fails for a reason other than
            a recognized CUDA OOM diagnostic, if backoff reaches batch size 1
            and still fails, or if ``max_attempts`` is exhausted.
    """

    batch_size = command.initial_batch_size
    attempts: list[StageAttempt] = []
    env = _stage_environment(command)

    for attempt_number in range(1, max_attempts + 1):
        argv = _build_argv_with_batch_size(command, batch_size)
        logger.info(
            "running stage %s attempt=%d batch_size=%d",
            command.name,
            attempt_number,
            batch_size,
        )
        started = time.perf_counter()
        result = subprocess.run(list(argv), shell=False, capture_output=True, text=True, env=env)
        elapsed = time.perf_counter() - started

        if result.returncode == 0:
            attempts.append(StageAttempt(attempt_number, batch_size, elapsed, False, True))
            logger.info(
                "stage %s succeeded in %.2fs at batch_size=%d", command.name, elapsed, batch_size
            )
            return StageResult(command.name, True, batch_size, tuple(attempts))

        diagnostic = (result.stderr or result.stdout or "").strip()[:_MAX_DIAGNOSTIC_CHARS]
        recognized_oom = _is_recognized_oom(diagnostic)
        attempts.append(
            StageAttempt(attempt_number, batch_size, elapsed, recognized_oom, False, diagnostic)
        )

        if not recognized_oom:
            logger.error("stage %s failed (non-OOM): %s", command.name, diagnostic[:500])
            raise StageExecutionError(
                f"stage {command.name!r} failed (exit_code={result.returncode}): {diagnostic[:500]}"
            )
        if batch_size == 1:
            logger.error("stage %s exhausted OOM backoff at batch_size=1", command.name)
            raise StageExecutionError(
                f"stage {command.name!r} failed at the minimum batch size 1: {diagnostic[:500]}"
            )

        batch_size = max(1, batch_size // 2)
        logger.warning(
            "stage %s hit recognized CUDA OOM; halving batch_size to %d", command.name, batch_size
        )

    raise StageExecutionError(f"stage {command.name!r} exceeded max_attempts={max_attempts}")


def _require_stage_resource_limits(
    commands: Sequence[StageCommand], scheduling: SchedulingConfig
) -> None:
    """Reject any stage command whose worker/thread request oversubscribes CPUs."""

    for command in commands:
        if command.image_workers > scheduling.available_cpus:
            raise ValueError(
                f"stage {command.name!r} image_workers ({command.image_workers}) "
                f"exceeds available_cpus ({scheduling.available_cpus})"
            )
        if command.cpu_threads is not None and command.cpu_threads > scheduling.available_cpus:
            raise ValueError(
                f"stage {command.name!r} cpu_threads ({command.cpu_threads}) "
                f"exceeds available_cpus ({scheduling.available_cpus})"
            )


def run_batch_stages(
    commands: Sequence[StageCommand],
    *,
    scheduling: SchedulingConfig | None = None,
    max_attempts: int = 10,
) -> list[StageResult]:
    """Run every batch stage command in strict order, one process at a time.

    Stages run sequentially in the exact order given by ``commands`` (e.g.
    extraction, Caption, OCR, OCR-scratch cleanup, Objects, FrameContext,
    visual embedding, context embedding, then the three batch indexes), which
    guarantees no two stages ever overlap.

    Raises:
        ValueError: If ``scheduling`` is given and any command oversubscribes
            its worker/thread request beyond ``available_cpus``.
        StageExecutionError: If a command fails without a declared output, or
            :func:`run_stage` raises for that command.
    """

    if scheduling is not None:
        _require_stage_resource_limits(commands, scheduling)

    results: list[StageResult] = []
    for command in commands:
        result = run_stage(command, max_attempts=max_attempts)
        output_path = Path(command.output_path)
        if not output_path.exists():
            raise StageExecutionError(
                f"stage {command.name!r} succeeded but did not produce its "
                f"declared output: {output_path}"
            )
        results.append(result)
    return results


__all__ = [
    "StageAttempt",
    "StageCommand",
    "StageExecutionError",
    "StageResult",
    "run_batch_stages",
    "run_stage",
]
