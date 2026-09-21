"""Canonical local filesystem layout for V3C/VBS preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VBSDataPaths:
    """Read-only source video root plus writable local work root."""

    root: Path
    videos_root: Path

    @classmethod
    def from_roots(
        cls,
        work_root: str | Path = "data",
        videos_root: str | Path | None = None,
    ) -> "VBSDataPaths":
        root = Path(work_root).expanduser().resolve()
        videos = (
            Path(videos_root).expanduser().resolve()
            if videos_root is not None
            else root / "videos"
        )
        return cls(root=root, videos_root=videos)

    @classmethod
    def from_root(cls, root: str | Path = "data") -> "VBSDataPaths":
        """Backward-compatible layout where videos live under the work root."""
        return cls.from_roots(root, None)

    @property
    def videos(self) -> Path:
        return self.videos_root

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
        """Create writable outputs; never create or mutate the source video root."""
        if not self.videos.is_dir():
            raise FileNotFoundError(f"Video directory does not exist: {self.videos}")
        for path in (self.frames, self.artifacts, self.indexes, self.state, self.frame_state):
            path.mkdir(parents=True, exist_ok=True)
