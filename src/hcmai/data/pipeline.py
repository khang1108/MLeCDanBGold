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

from hcmai.common.schemas import (
    FrameContext,
    FrameRecord,
    ObjectEvidence,
    RetrievalSource,
    TranscriptSegment,
)
from hcmai.common.schemas.search import SearchFilters
from hcmai.common.utils.io import read_json, read_yaml
from hcmai.data.assets import FrameAssetResolver, FrameAssetStatus
from hcmai.data.enrichment.transcripts.store import TranscriptStore
from hcmai.data.ingestion import BTCIngestionConfig, import_btc_frame_store
from hcmai.data.stores import (
    ASRStore,
    CaptionStore,
    FrameContextStore,
    FrameStore,
    ObjectStore,
    OCRStore,
)

EvidenceStore = CaptionStore | OCRStore | ASRStore
_EVIDENCE_STORES = {
    RetrievalSource.CAPTION: CaptionStore,
    RetrievalSource.OCR: OCRStore,
    RetrievalSource.ASR: ASRStore,
}
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _project_path(value: str | Path) -> Path:
    """Resolve active-config paths from the repository, preserving absolutes."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else _PROJECT_ROOT / path


class DataService:
    """Expose canonical frame preparation and lookup through one facade."""

    def __init__(
        self,
        frame_store: FrameStore | None = None,
        evidence_stores: Mapping[RetrievalSource, EvidenceStore] | None = None,
        asset_resolver: FrameAssetResolver | None = None,
        object_store: ObjectStore | None = None,
        context_store: FrameContextStore | None = None,
        transcript_store: TranscriptStore | None = None,
    ) -> None:
        """Initialize the facade from already loaded specialist stores."""

        self.frame_store = frame_store
        self.evidence_stores = dict(evidence_stores or {})
        self.asset_resolver = asset_resolver
        self.object_store = object_store
        self.context_store = context_store
        self.transcript_store = transcript_store

    @classmethod
    def load(
        cls,
        frames_path: str | Path,
        evidence_paths: Mapping[RetrievalSource, str | Path] | None = None,
        *,
        dataset_root: str | Path | None = None,
        object_path: str | Path | None = None,
        context_path: str | Path | None = None,
        transcript_path: str | Path | None = None,
    ) -> "DataService":
        """Load canonical frames and any explicitly configured evidence."""

        frames = FrameStore(frames_path)
        evidence = {
            source: _EVIDENCE_STORES[source](path)
            for source, path in (evidence_paths or {}).items()
        }
        objects = ObjectStore(object_path) if object_path is not None else None
        contexts = (
            FrameContextStore(context_path) if context_path is not None else None
        )
        transcripts = None
        if transcript_path is not None:
            transcript_artifact = Path(transcript_path)
            if not transcript_artifact.exists():
                raise FileNotFoundError(
                    "Transcript artifact does not exist: "
                    f"{transcript_artifact}"
                )
            transcripts = TranscriptStore(transcript_artifact)
        for store in (*evidence.values(), objects, contexts):
            if store is not None and not isinstance(store, ASRStore):
                _validate_canonical_identity(frames, store)
                _validate_canonical_lineage(frames, store)
        resolver = (
            FrameAssetResolver(dataset_root)
            if dataset_root is not None
            else None
        )
        return cls(
            frames,
            evidence,
            resolver,
            objects,
            contexts,
            transcripts,
        )

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
        if not isinstance(store, ASRStore):
            _validate_canonical_identity(self._frames(), store)
            _validate_canonical_lineage(self._frames(), store)
        self.evidence_stores[source] = store
        return store

    @staticmethod
    def prepare(
        config_path: str | Path,
        *,
        resume: bool = True,
        limit: int | None = None,
    ) -> Path:
        """Import the BTC-native frame store configured for HCMAI 2026.

        ``resume`` and ``limit`` remain in the public signature for caller
        compatibility. BTC V1 ingestion is deterministic and always imports
        the complete metadata table.
        """

        resolved_config = Path(config_path).expanduser().resolve()
        raw_config = read_yaml(resolved_config)
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"Expected a YAML mapping in {config_path}")
        dataset = raw_config.get("dataset")
        if not isinstance(dataset, Mapping):
            raise ValueError("Enrichment YAML requires a dataset mapping")

        required = {
            "version",
            "source",
            "btc_root",
            "mapping_root",
            "data_root",
            "frame_store_id",
            "frames_path",
            "frame_store_output",
        }
        missing = sorted(required.difference(dataset))
        if missing:
            raise ValueError(
                "Missing dataset configuration: " + ", ".join(missing)
            )

        source = str(dataset["source"])
        if source != "btc_keyframes":
            raise ValueError(
                f"Unsupported dataset.source {source!r}; expected 'btc_keyframes'"
            )

        output_root = _project_path(str(dataset["frame_store_output"]))
        frames_path = _project_path(str(dataset["frames_path"]))
        expected_frames_path = output_root / "frames.parquet"
        if frames_path.resolve() != expected_frames_path.resolve():
            raise ValueError(
                "dataset.frames_path must equal "
                "dataset.frame_store_output/frames.parquet"
            )

        return import_btc_frame_store(
            BTCIngestionConfig(
                btc_root=_project_path(str(dataset["btc_root"])),
                mapping_root=_project_path(str(dataset["mapping_root"])),
                data_root=_project_path(str(dataset["data_root"])),
                output_root=output_root,
                frame_store_id=str(dataset["frame_store_id"]),
            )
        )

    @staticmethod
    def prepare_adaptive(
        config_path: str | Path,
        *,
        resume: bool = True,
        limit: int | None = None,
    ) -> Path:
        """Run the explicit legacy/custom-video preprocessing workflow."""

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

    def get_object_evidence(self, frame_id: str) -> ObjectEvidence | None:
        """Return structured object evidence without fusing other modalities."""

        if self.object_store is None:
            return None
        try:
            return self.object_store.get(frame_id)
        except KeyError:
            return None

    def get_frame_context(self, frame_id: str) -> FrameContext | None:
        """Return deterministic derived context for one frame, when loaded."""

        if self.context_store is None:
            return None
        try:
            return self.context_store.get(frame_id)
        except KeyError:
            return None

    def iter_frame_contexts(self) -> Iterator[FrameContext]:
        """Iterate loaded deterministic frame-native contexts in artifact order."""

        if self.context_store is None:
            return iter(())
        return self.context_store.iter_records()

    def get_frame_context_text(self, frame_id: str) -> str | None:
        """Return usable deterministic context text without fabricating evidence."""

        if self.context_store is None:
            return None
        try:
            return self.context_store.get_text(frame_id)
        except KeyError:
            return None

    def get_transcript_segments(
        self,
        video_id: str,
        start_ms: int,
        end_ms: int,
    ) -> list[TranscriptSegment]:
        """Return segments overlapping a half-open range chronologically."""

        if start_ms < 0:
            raise ValueError("start_ms must be non-negative")
        if end_ms < start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        if end_ms == start_ms:
            return []
        if self.transcript_store is None:
            return []
        records = self.transcript_store.get_in_range(video_id, start_ms, end_ms)
        return sorted(
            records,
            key=lambda row: (
                row.start_ms,
                row.end_ms,
                row.segment_index,
                row.segment_id,
            ),
        )

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


def _validate_canonical_identity(
    frames: FrameStore,
    store: CaptionStore | OCRStore | ObjectStore | FrameContextStore,
) -> None:
    """Reject typed evidence that invents or rewrites canonical frame identity."""

    for evidence in store.iter_records():
        try:
            frame = frames.get(evidence.frame_id)
        except KeyError:
            raise ValueError(
                f"Evidence contains unknown canonical frame_id: {evidence.frame_id}"
            ) from None
        if (
            evidence.video_id != frame.video_id
            or evidence.frame_idx != frame.frame_idx
            or evidence.timestamp_ms != frame.timestamp_ms
        ):
            raise ValueError(
                "Evidence does not match canonical identity for frame_id "
                f"{evidence.frame_id!r}"
            )


def _validate_canonical_lineage(
    frames: FrameStore,
    store: CaptionStore | OCRStore | ObjectStore | FrameContextStore,
) -> None:
    """Compare specialist lineage with the canonical frame manifest, if any."""

    manifest_path = frames.metadata_path.with_name("manifest.json")
    if not manifest_path.exists():
        return
    try:
        manifest = read_json(manifest_path)
    except Exception as error:
        raise ValueError(
            f"Malformed canonical frame manifest: {manifest_path}"
        ) from error
    if not isinstance(manifest, Mapping):
        raise ValueError("Canonical frame manifest must contain an object")
    canonical = manifest.get("frame_store_id")
    if canonical is None:
        return
    if (
        not isinstance(canonical, str)
        or not canonical
        or canonical.strip() != canonical
    ):
        raise ValueError(
            "Canonical frame manifest has invalid frame_store_id"
        )
    if store.frame_store_id != canonical:
        raise ValueError(
            "Evidence frame_store_id does not match canonical frame_store_id: "
            f"{store.artifact_path}"
        )
