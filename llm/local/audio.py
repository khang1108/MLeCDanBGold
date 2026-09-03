"""Bounded audio-reference download for local transcript inference."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import httpx


def download_audio(payload: Any, target: Path) -> None:
    """Download one audio reference while enforcing size and SHA-256 identity."""

    maximum = int(os.getenv("HCMAI_MAX_AUDIO_BYTES", str(1024 * 1024 * 1024)))
    digest = hashlib.sha256()
    total = 0
    with httpx.stream(
        "GET",
        payload.audio_url,
        timeout=300,
        follow_redirects=False,
    ) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_bytes(1024 * 1024):
                total += len(chunk)
                if total > maximum:
                    raise ValueError("remote audio exceeds configured byte limit")
                digest.update(chunk)
                handle.write(chunk)

    if digest.hexdigest() != payload.audio_sha256:
        raise ValueError("remote audio checksum mismatch")
