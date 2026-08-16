"""Bounded execution policy for cached Frame and ASR preparation."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import Any

from hcmai.data.s3 import S3VideoObject

logger = logging.getLogger(__name__)


def prepare_cached_sources(
    sources: Sequence[S3VideoObject],
    *,
    resolve: Callable[[S3VideoObject], Path],
    prepare_frame: Callable[[Path, S3VideoObject], Any],
    prepare_transcript: Callable[[Path], Path],
    frame_pending: bool,
    asr_pending: bool,
    overlap: bool,
) -> list[Any]:
    """Prepare cached sources while each model-owning lane stays serial."""

    prepared: list[Any] = []
    if not sources:
        return prepared
    if not overlap or not (frame_pending and asr_pending) or len(sources) == 1:
        for index, source in enumerate(sources, start=1):
            logger.info(
                "Processing video %d/%d: %s",
                index,
                len(sources),
                source.video_id,
            )
            video = resolve(source)
            if frame_pending:
                prepared.append(prepare_frame(video, source))
            if asr_pending:
                prepare_transcript(video)
        return prepared

    first, remaining = sources[0], sources[1:]
    logger.info("Serial model warm-up: %s", first.video_id)
    first_video = resolve(first)
    prepared.append(prepare_frame(first_video, first))
    prepare_transcript(first_video)
    stop = Event()

    def frame_lane() -> list[Any]:
        started = perf_counter()
        tables: list[Any] = []
        for index, source in enumerate(remaining, start=2):
            if stop.is_set():
                break
            logger.info(
                "Frame lane video %d/%d: %s",
                index,
                len(sources),
                source.video_id,
            )
            tables.append(prepare_frame(resolve(source), source))
        logger.info(
            "Frame lane finished: videos=%d seconds=%.1f",
            len(tables),
            perf_counter() - started,
        )
        return tables

    def asr_lane() -> None:
        started = perf_counter()
        count = 0
        for index, source in enumerate(remaining, start=2):
            if stop.is_set():
                break
            logger.info(
                "ASR lane video %d/%d: %s",
                index,
                len(sources),
                source.video_id,
            )
            prepare_transcript(resolve(source))
            count += 1
        logger.info(
            "ASR lane finished: videos=%d seconds=%.1f",
            count,
            perf_counter() - started,
        )

    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="hcmai-preparation",
    ) as executor:
        frame_future = executor.submit(frame_lane)
        asr_future = executor.submit(asr_lane)
        done, _ = wait(
            (frame_future, asr_future),
            return_when=FIRST_EXCEPTION,
        )
        if any(future.exception() is not None for future in done):
            stop.set()
        for future in (frame_future, asr_future):
            future.result()
    prepared.extend(frame_future.result())
    return prepared
