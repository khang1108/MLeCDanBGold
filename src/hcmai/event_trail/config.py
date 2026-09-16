"""Configuration settings for EventTrail snapshots and sessions."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class EventTrailSettings:
    """Bounded capacity and TTL settings for EventTrail backend."""

    snapshot_ttl_seconds: int = 900
    session_ttl_seconds: int = 1800
    max_snapshots: int = 64
    max_sessions: int = 128

    @classmethod
    def from_env(cls) -> EventTrailSettings:
        """Load and validate EventTrail settings from environment variables."""

        def parse_pos_int(name: str, default: int) -> int:
            val = os.getenv(name)
            if val is None:
                return default
            val = val.strip()
            if not val:
                raise ValueError(f"{name} must not be blank")
            try:
                parsed = int(val)
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer") from exc
            if parsed <= 0:
                raise ValueError(f"{name} must be a positive integer")
            return parsed

        return cls(
            snapshot_ttl_seconds=parse_pos_int(
                "HCMAI_EVENT_TRAIL_SNAPSHOT_TTL_SECONDS", 900
            ),
            session_ttl_seconds=parse_pos_int(
                "HCMAI_EVENT_TRAIL_SESSION_TTL_SECONDS", 1800
            ),
            max_snapshots=parse_pos_int("HCMAI_EVENT_TRAIL_MAX_SNAPSHOTS", 64),
            max_sessions=parse_pos_int("HCMAI_EVENT_TRAIL_MAX_SESSIONS", 128),
        )
