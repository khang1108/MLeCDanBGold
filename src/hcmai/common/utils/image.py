"""Small helpers for loading images without leaking open file handles."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any
from PIL import Image


PathValue = str | PathLike[str]


def load_image(path: PathValue, *, mode: str | None = None) -> Any:
    """Load an image and return a detached Pillow image object.
    """

    with Image.open(Path(path)) as image:
        loaded_image = image.convert(mode) if mode else image.copy()
        loaded_image.load()
        return loaded_image


__all__ = ["load_image"]
