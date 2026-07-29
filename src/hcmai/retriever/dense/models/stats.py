"""Runtime statistics for dense encoding.

Stats are produced while running models, kept separate from configuration
inputs (``config.py``) and artifact provenance metadata (``metadata.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EncodingStats:
    """Statistics from an encoding operation."""

    num_encoded: int = 0
    num_failed: int = 0
    total_time_ms: float = 0.0
    batch_times_ms: list[float] = field(default_factory=list)
    embedding_dim: int = 0

    @property
    def throughput_samples_per_sec(self) -> float:
        """Calculate throughput in samples per second."""
        if self.total_time_ms <= 0:
            return 0.0
        return (self.num_encoded / self.total_time_ms) * 1000

    @property
    def avg_batch_time_ms(self) -> float:
        """Calculate average batch processing time."""
        if not self.batch_times_ms:
            return 0.0
        return sum(self.batch_times_ms) / len(self.batch_times_ms)

    @property
    def p95_batch_time_ms(self) -> float:
        """Calculate 95th percentile batch processing time."""
        if not self.batch_times_ms:
            return 0.0
        sorted_times = sorted(self.batch_times_ms)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    def report(self) -> str:
        """Generate a human-readable report of encoding stats."""
        return (
            f"Encoded {self.num_encoded} samples in {self.total_time_ms:.1f}ms "
            f"({self.throughput_samples_per_sec:.1f} samples/sec), "
            f"embedding_dim={self.embedding_dim}, "
            f"failed={self.num_failed}, "
            f"avg_batch={self.avg_batch_time_ms:.1f}ms, "
            f"p95_batch={self.p95_batch_time_ms:.1f}ms"
        )
