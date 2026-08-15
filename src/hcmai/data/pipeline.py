"""Entry point cho Data Pipeline chính.

Cung cấp `DataService`, là giao diện trung tâm điều phối toàn bộ quá trình xử lý dữ liệu từ video thô đến khi ra corpus.

Các tính năng chính:
1. Điều phối Preprocessing: Kích hoạt quá trình trích xuất và lọc frame từ video thô.
2. Điều phối Enrichment: Chạy các luồng OCR, Captioning, và Transcript để làm giàu metadata.
3. Quản lý trạng thái: Theo dõi tiến độ của pipeline, hỗ trợ resume an toàn khi có tác vụ bị lỗi."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from itertools import islice
from pathlib import Path

from hcmai.common.schemas import FrameRecord, RetrievalSource
from hcmai.common.schemas.search import SearchFilters
from hcmai.data.assets import FrameAssetResolver, FrameAssetStatus
from hcmai.data.stores import ASRStore, CaptionStore, FrameStore, OCRStore

EvidenceStore = CaptionStore | OCRStore | ASRStore
_EVIDENCE_STORES = {
    RetrievalSource.CAPTION: CaptionStore,
    RetrievalSource.OCR: OCRStore,
    RetrievalSource.ASR: ASRStore,
}


class DataService:
    """Expose canonical frame preparation and lookup through one facade."""

    def __init__(
        self,
        frame_store: FrameStore | None = None,
        evidence_stores: Mapping[RetrievalSource, EvidenceStore] | None = None,
        asset_resolver: FrameAssetResolver | None = None,
    ) -> None:
        self.frame_store = frame_store
        self.evidence_stores = dict(evidence_stores or {})
        self.asset_resolver = asset_resolver

    @classmethod
    def load(
        cls,
        frames_path: str | Path,
        evidence_paths: Mapping[RetrievalSource, str | Path] | None = None,
        *,
        dataset_root: str | Path | None = None,
    ) -> "DataService":
        """Load canonical frames and any explicitly configured evidence."""

        evidence = {
            source: _EVIDENCE_STORES[source](path)
            for source, path in (evidence_paths or {}).items()
        }
        resolver = FrameAssetResolver(dataset_root) if dataset_root is not None else None
        return cls(FrameStore(frames_path), evidence, resolver)

    def load_evidence(
        self, source: RetrievalSource, artifact_path: str | Path
    ) -> EvidenceStore:
        """Load one evidence channel without rereading canonical frames."""

        try:
            store_type = _EVIDENCE_STORES[source]
        except KeyError:
            raise ValueError(
                f"{source.value!r} is not a text evidence source"
            ) from None
        store = store_type(artifact_path)
        self.evidence_stores[source] = store
        return store

    @staticmethod
    def prepare(
        config_path: str | Path,
        *,
        resume: bool = True,
        limit: int | None = None,
    ) -> Path:
        """Run adaptive offline video preparation behind the data facade."""

        from hcmai.data.preprocessing import (
            PreprocessingConfig,
            prepare_frame_store,
            prepare_frame_store_from_s3,
        )

        config = PreprocessingConfig.from_yaml(config_path)
        if config.s3 is not None:
            return prepare_frame_store_from_s3(
                config, resume=resume, limit=limit
            )
        return prepare_frame_store(config, resume=resume, limit=limit)

    def get_frame(self, frame_id: str) -> FrameRecord:
        return self._frames().get(frame_id)

    def get_frames(self, frame_ids: Sequence[str]) -> list[FrameRecord]:
        return self._frames().get_many(frame_ids)

    def neighbors(
        self,
        frame_id: str,
        *,
        window_ms: int,
        include_self: bool = False,
    ) -> list[FrameRecord]:
        return self._frames().get_neighbors(
            frame_id, window_ms=window_ms, include_self=include_self
        )

    def filter_frame_ids(self, filters: SearchFilters | None = None) -> list[str]:
        return self._frames().filter_frame_ids(filters)

    def iter_frames(self) -> Iterator[FrameRecord]:
        return self._frames().iter_frames()

    def contains_submission(self, video_id: str, frame_idx: int) -> bool:
        return self._frames().contains_submission(video_id, frame_idx)

    def resolve_frame_asset(
        self,
        frame: FrameRecord | str,
        *,
        thumbnail: bool = False,
        require_file: bool = True,
    ) -> Path:
        if self.asset_resolver is None:
            raise RuntimeError("Frame asset resolver is not configured")
        record = self.get_frame(frame) if isinstance(frame, str) else frame
        return self.asset_resolver.resolve_frame(
            record,
            thumbnail=thumbnail,
            require_file=require_file,
        )

    def frame_asset_status(self, *, sample_size: int = 100) -> FrameAssetStatus:
        if self.asset_resolver is None:
            return FrameAssetStatus(False, 0, 0, 0)
        records = tuple(islice(self.iter_frames(), sample_size))
        return self.asset_resolver.sample_status(records, sample_size=sample_size)

    @property
    def record_count(self) -> int:
        return len(self._frames())

    def __len__(self) -> int:
        return self.record_count

    def get_evidence(
        self, frame_id: str, source: RetrievalSource
    ) -> str | None:
        store = self.evidence_stores.get(source)
        if store is None:
            return None
        try:
            return store.get_text(frame_id)
        except KeyError:
            return None

    def has_evidence(self, source: RetrievalSource) -> bool:
        """Report whether one text-evidence artifact is loaded."""

        return source in self.evidence_stores

    def iter_evidence(self, source: RetrievalSource) -> Iterator[object]:
        store = self.evidence_stores.get(source)
        if store is None:
            return iter(())
        return store.iter_records()

    def _frames(self) -> FrameStore:
        if self.frame_store is None:
            raise RuntimeError("Canonical frame metadata is not loaded")
        return self.frame_store
