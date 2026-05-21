from __future__ import annotations

from typing import Any, Optional
from dataclasses import dataclass
from pathlib import Path
from pydantic import Field

ROOT_DIR = Path(__file__).parent.parent.parent

@dataclass(frozen=True)
class DataConfig:
    data_root: Path = ROOT_DIR / "data"
    output_root: Path = ROOT_DIR / "outputs"

    image_size: int = 224