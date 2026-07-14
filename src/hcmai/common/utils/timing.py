"""Helpers for measuring elapsed time in retrieval and processing stages."""

from __future__ import annotations

from time import perf_counter


def elapsed_ms(started_at: float, ended_at: float | None = None) -> float:
    """Return elapsed time in milliseconds from two monotonic timestamps."""

    end = perf_counter() if ended_at is None else ended_at
    return max(0.0, (end - started_at) * 1_000)


class Timer:
    """Reusable context manager for measuring one operation."""

    def __init__(self) -> None:
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.start()

    def start(self) -> None:
        """Start or restart the timer."""

        self.started_at = perf_counter()
        self.ended_at = None

    def stop(self) -> float:
        """Stop the timer and return the elapsed duration in milliseconds."""

        if self.started_at is None:
            raise RuntimeError("timer has not been started")

        self.ended_at = perf_counter()
        return self.elapsed_ms

    @property
    def elapsed_ms(self) -> float:
        """Return the current or final elapsed duration in milliseconds."""

        if self.started_at is None:
            raise RuntimeError("timer has not been started")

        return elapsed_ms(self.started_at, self.ended_at)

    def __enter__(self) -> "Timer":
        """Start timing when entering a ``with`` block."""

        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        """Stop timing when leaving a ``with`` block."""

        self.stop()


__all__ = ["Timer", "elapsed_ms"]
