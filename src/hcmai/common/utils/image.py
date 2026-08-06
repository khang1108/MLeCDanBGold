"""Small helpers for loading images without leaking open file handles."""

from __future__ import annotations

import io
from os import PathLike
from pathlib import Path
from typing import Any, Hashable
from PIL import Image


PathValue = str | PathLike[str]


def load_image(path: PathValue, *, mode: str | None = None) -> Any:
    """Load an image and return a detached Pillow image object.
    """

    with Image.open(Path(path)) as image:
        loaded_image = image.convert(mode) if mode else image.copy()
        loaded_image.load()
        return loaded_image


def thumbnail_jpeg_bytes(
    path: PathValue,
    *,
    key: Hashable,
    cache: Any,
    maximum_size: tuple[int, int] = (384, 384),
    quality: int = 85,
) -> bytes:
    """Return cached compressed thumbnail bytes without retaining PIL objects."""

    cached = cache.get(key)
    if cached is not None:
        return cached
    with Image.open(Path(path)) as source:
        image = source.convert("RGB")
        image.thumbnail(maximum_size)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=quality)
        image.close()
    value = output.getvalue()
    cache.set(key, value)
    return value


__all__ = ["load_image", "thumbnail_jpeg_bytes"]
