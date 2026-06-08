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

    image_size: int = Field(default=224, description="Size of image")
    batch_size: int = Field(default=64, description="Batch size")
    num_workers: int = Field(default=2, description="Number of workers")
    shuffle: bool = False

    def __post_init__(self):
        """
        Run extra initialization logic after the dataclass-generated __init__
        has assigned all fields.
        """

        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "output_root", Path(self.output_root))


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = Field(
        default=20, description="Top k candidates will be retrieved from Vector DB"
    )
    embedding_dim: int = Field(
        default=512,
        description="Dimension of model embedding. Most of case it is 512 (default)",
    )
    normalize_embeddings: bool = False


@dataclass(frozen=True)
class Config:
    data: DataConfig
    retrieval: RetrievalConfig
