"""Canonical local filesystem layout for VBS preparation.

Everything durable lives below one data root.  The only network boundary in
this pipeline is model inference; corpus files and retrieval artifacts remain
local.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VBSDataPaths:
    """Resolved paths owned by the local VBS dataset."""

    root: Path

    @classmethod
    def from_root(cls, root: str | Path = "data") -> "VBSDataPaths":
        return cls(Path(root).expanduser().resolve())

    @property
    def videos(self) -> Path:
        return self.root / "videos"

    @property
    def frames(self) -> Path:
        return self.root / "frames"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def frame_table(self) -> Path:
        return self.artifacts / "frames.parquet"

    @property
    def frame_manifest(self) -> Path:
        return self.artifacts / "manifest.json"

    @property
    def captions(self) -> Path:
        return self.artifacts / "captions"

    @property
    def ocr(self) -> Path:
        return self.artifacts / "ocr"

    @property
    def objects(self) -> Path:
        return self.artifacts / "objects"

    @property
    def asr(self) -> Path:
        return self.artifacts / "asr"

    @property
    def context(self) -> Path:
        return self.artifacts / "context"

    @property
    def indexes(self) -> Path:
        return self.root / "indexes"

    @property
    def visual_index(self) -> Path:
        return self.indexes / "visual"

    @property
    def context_index(self) -> Path:
        return self.indexes / "context"

    @property
    def asr_index(self) -> Path:
        return self.indexes / "asr"

    @property
    def state(self) -> Path:
        return self.root / "state"

    @property
    def frame_state(self) -> Path:
        return self.state / "frames"

    def ensure(self) -> None:
        """Create the stable top-level directories without creating junk layers."""

        for path in (
            self.videos,
            self.frames,
            self.artifacts,
            self.indexes,
            self.state,
            self.frame_state,
        ):
            path.mkdir(parents=True, exist_ok=True)
