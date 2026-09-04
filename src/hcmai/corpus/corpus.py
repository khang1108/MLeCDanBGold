"""Expose read-only runtime access to existing HCMAI corpus artifacts.

``Corpus`` composes the private specialist stores into the small runtime API
used by search callers.  It opens published artifacts only; it does not own
ingestion, enrichment, context construction, index building, or mutation.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from itertools import islice
from pathlib import Path

from hcmai.common.utils.io import read_json
from hcmai.corpus.assets import FrameAssetResolver, FrameAssetStatus
from hcmai.corpus.models import Frame, TranscriptSegment
from hcmai.corpus.stores import (
    CaptionStore,
    FrameStore,
    OCRStore,
    ObjectCountsStore,
    TranscriptStore,
    VideoMetadataStore,
)
from hcmai.retrieval.models import RetrievalSource


_TEXT_STORES = {
    RetrievalSource.CAPTION: CaptionStore,
    RetrievalSource.OCR: OCRStore,
}


class _CorpusFrameLoadError(FileNotFoundError):
    """Report a failure while opening required canonical frame metadata.

    Optional evidence may be unavailable during degraded startup, but this
    error marks an unreadable or invalid frame authority that must stop it.
    """


class Corpus:
    """Read canonical frames and explicitly configured runtime evidence.

    Instances are created with :meth:`open`, which loads existing artifacts at
    startup.  The facade keeps specialist evidence stores private so public
    callers cannot mutate them or depend on artifact-specific representations.
    """

    def __init__(
        self,
        frames: FrameStore,
        evidence: Mapping[RetrievalSource, CaptionStore | OCRStore],
        *,
        asset_resolver: FrameAssetResolver | None,
        object_counts: ObjectCountsStore | None,
        transcripts: TranscriptStore | None,
        video_metadata: VideoMetadataStore | None,
    ) -> None:
        """Initialize an already-open corpus from validated private stores."""

        self._frames = frames
        self._evidence = dict(evidence)
        self._asset_resolver = asset_resolver
        self._object_counts = object_counts
        self._transcripts = transcripts
        self._video_metadata = video_metadata

    @classmethod
    def open(
        cls,
        frames_path: str | Path,
        evidence_paths: Mapping[RetrievalSource, str | Path] | None = None,
        *,
        dataset_root: str | Path | None = None,
        object_counts_path: str | Path | None = None,
        transcript_path: str | Path | None = None,
        video_metadata_path: str | Path | None = None,
    ) -> "Corpus":
        """Open existing frame metadata and explicitly supplied artifacts.

        ``frames_path`` is required and loaded first, so missing canonical
        metadata fails before optional artifact processing.  Only caption and
        OCR artifact paths are accepted in ``evidence_paths``; frame-aligned
        ASR, raw detections, and derived context are outside this facade.
        """

        frame_artifact = Path(frames_path)
        try:
            if not frame_artifact.is_file():
                raise FileNotFoundError(
                    f"Frame artifact is not a file: {frame_artifact}"
                )
            frames = FrameStore(frame_artifact)
        except Exception as error:
            raise _CorpusFrameLoadError(
                f"Could not load canonical frame artifact: {frame_artifact}"
            ) from error
        evidence = cls._open_text_evidence(evidence_paths)
        object_counts = (
            ObjectCountsStore(object_counts_path)
            if object_counts_path is not None
            else None
        )
        if transcript_path is not None and not Path(transcript_path).exists():
            raise FileNotFoundError(
                f"Transcript artifact does not exist: {transcript_path}"
            )
        transcripts = TranscriptStore(transcript_path) if transcript_path else None
        video_metadata = (
            VideoMetadataStore(video_metadata_path)
            if video_metadata_path is not None
            else None
        )

        for store in evidence.values():
            cls._validate_evidence_identity(frames, store)
            cls._validate_evidence_lineage(frames, store)
        if object_counts is not None:
            cls._validate_object_counts_identity(frames, object_counts)
            cls._validate_object_counts_lineage(frames, object_counts)

        return cls(
            frames,
            evidence,
            asset_resolver=(
                FrameAssetResolver(dataset_root)
                if dataset_root is not None
                else None
            ),
            object_counts=object_counts,
            transcripts=transcripts,
            video_metadata=video_metadata,
        )

    def frame(self, frame_id: str) -> Frame:
        """Return one canonical frame, preserving its organizer coordinates."""

        return self._frames.get(frame_id)

    def frame_at_timestamp(self, video_id: str, timestamp_ms: int) -> Frame:
        """Resolve a manual viewer timestamp to one canonical keyframe.

        The returned frame retains its stored identity and may have a different
        timestamp from the requested viewer position when no exact keyframe
        exists. Callers must keep the requested playback timestamp separate.
        """

        return self._frames.get_nearest_by_video(
            video_id,
            timestamp_ms=timestamp_ms,
        )

    def __len__(self) -> int:
        """Return the number of canonical frames loaded for runtime search."""

        return len(self._frames)

    def frame_asset_status(self, *, sample_size: int = 100) -> FrameAssetStatus:
        """Sample canonical frame assets without exposing the asset resolver."""

        resolver = self._asset_resolver_or_error()
        return resolver.sample_status(
            tuple(islice(self._frames.iter_frames(), sample_size)),
            sample_size=sample_size,
        )

    def has_evidence(self, source: RetrievalSource) -> bool:
        """Report whether one configured runtime evidence view is available."""

        if source is RetrievalSource.ASR:
            return self._transcripts is not None
        return source in self._evidence

    def frames(self, frame_ids: Sequence[str]) -> list[Frame]:
        """Return canonical frames in the requested order, including duplicates."""

        return self._frames.get_many(frame_ids)

    def iter_frames(self) -> Iterator[Frame]:
        """Iterate canonical frames in their deterministic artifact order."""

        return self._frames.iter_frames()

    def caption(self, frame_id: str) -> str | None:
        """Return usable completed caption text, or ``None`` when unavailable."""

        return self._text(frame_id, RetrievalSource.CAPTION)

    def ocr(self, frame_id: str) -> str | None:
        """Return usable completed normalized OCR text, or ``None`` when absent."""

        return self._text(frame_id, RetrievalSource.OCR)

    def objects(self, frame_id: str) -> tuple[str, ...]:
        """Return stable sorted object labels from completed count evidence.

        This label-only projection supports existing retrieval callers. Use
        :meth:`object_counts` when display or literal search needs multiplicity.
        """

        if self._object_counts is None:
            return ()
        counts = self._object_counts.get_counts(frame_id)
        return tuple(sorted(counts)) if counts is not None else ()

    def object_counts(self, frame_id: str) -> dict[str, int]:
        """Return completed object counts without discarding multiplicity."""

        if self._object_counts is None:
            return {}
        return self._object_counts.get_counts(frame_id) or {}

    def has_object_counts(self) -> bool:
        """Report whether frame-aligned object evidence is configured."""

        return self._object_counts is not None

    def transcript(
        self,
        video_id: str,
        start_ms: int,
        end_ms: int,
    ) -> str | None:
        """Join non-empty transcript text overlapping a half-open time range."""

        texts = [
            segment.text.strip()
            for segment in self.transcript_segments(video_id, start_ms, end_ms)
            if segment.text.strip()
        ]
        return " ".join(texts) if texts else None

    def transcript_segments(
        self,
        video_id: str,
        start_ms: int,
        end_ms: int,
    ) -> tuple[TranscriptSegment, ...]:
        """Return exact transcript segments overlapping a half-open time range."""

        self._validate_time_range(start_ms, end_ms)
        if start_ms == end_ms or self._transcripts is None:
            return ()
        return tuple(
            sorted(
                self._transcripts.get_in_range(video_id, start_ms, end_ms),
                key=lambda segment: (
                    segment.start_ms,
                    segment.end_ms,
                    segment.segment_index,
                    segment.segment_id,
                ),
            )
        )

    def transcript_segments_for_video(
        self,
        video_id: str,
    ) -> tuple[TranscriptSegment, ...]:
        """Return all transcript segments for one video in timeline order."""

        if self._transcripts is None:
            return ()
        return tuple(sorted(
            self._transcripts.get_by_video(video_id),
            key=lambda segment: (
                segment.start_ms,
                segment.end_ms,
                segment.segment_index,
                segment.segment_id,
            ),
        ))

    def title(self, video_id: str) -> str | None:
        """Return organizer video title, or ``None`` when metadata is unavailable."""

        if self._video_metadata is None:
            return None
        metadata = self._video_metadata.get(video_id)
        return metadata.title if metadata is not None else None

    def has_titles(self) -> bool:
        """Report whether organizer video metadata is configured."""

        return self._video_metadata is not None

    def image_path(self, frame_id: str) -> Path:
        """Resolve and verify one frame image under the configured dataset root."""

        return self._asset_resolver_or_error().resolve_frame(self.frame(frame_id))

    def thumbnail_path(self, frame_id: str) -> Path:
        """Resolve and verify one frame thumbnail under the dataset root."""

        return self._asset_resolver_or_error().resolve_frame(
            self.frame(frame_id), thumbnail=True
        )

    @staticmethod
    def _open_text_evidence(
        evidence_paths: Mapping[RetrievalSource, str | Path] | None,
    ) -> dict[RetrievalSource, CaptionStore | OCRStore]:
        """Open only supported explicitly configured text evidence stores."""

        evidence: dict[RetrievalSource, CaptionStore | OCRStore] = {}
        for source, path in (evidence_paths or {}).items():
            try:
                store_type = _TEXT_STORES[source]
            except KeyError:
                source_name = getattr(source, "value", source)
                raise ValueError(
                    f"{source_name!r} is not supported by the Corpus text API"
                ) from None
            evidence[source] = store_type(path)
        return evidence

    def _text(self, frame_id: str, source: RetrievalSource) -> str | None:
        """Read one optional text projection without exposing its store."""

        store = self._evidence.get(source)
        if store is None:
            return None
        try:
            return store.get_text(frame_id)
        except KeyError:
            return None

    def _asset_resolver_or_error(self) -> FrameAssetResolver:
        """Require an explicit dataset root before resolving frame assets."""

        if self._asset_resolver is None:
            raise RuntimeError(
                "Corpus asset paths require dataset_root when opening the corpus"
            )
        return self._asset_resolver

    @staticmethod
    def _validate_time_range(start_ms: int, end_ms: int) -> None:
        """Reject invalid transcript windows without converting them to no-match."""

        if start_ms < 0:
            raise ValueError("start_ms must be non-negative")
        if end_ms < start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")

    @staticmethod
    def _validate_evidence_identity(
        frames: FrameStore,
        store: CaptionStore | OCRStore,
    ) -> None:
        """Reject text evidence that rewrites canonical frame identity."""

        for evidence in store.iter_records():
            try:
                frame = frames.get(evidence.frame_id)
            except KeyError:
                raise ValueError(
                    "Evidence contains unknown canonical frame_id: "
                    f"{evidence.frame_id}"
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

    @staticmethod
    def _validate_object_counts_identity(
        frames: FrameStore,
        store: ObjectCountsStore,
    ) -> None:
        """Reject count artifacts that rewrite canonical frame identity."""

        for record in store.iter_records():
            try:
                frame = frames.get(record.frame_id)
            except KeyError:
                raise ValueError(
                    "Object counts contain unknown canonical frame_id: "
                    f"{record.frame_id}"
                ) from None
            if (
                record.video_id != frame.video_id
                or record.frame_idx != frame.frame_idx
                or record.timestamp_ms != frame.timestamp_ms
            ):
                raise ValueError(
                    "Object counts do not match canonical identity for frame_id "
                    f"{record.frame_id!r}"
                )

    @staticmethod
    def _validate_evidence_lineage(
        frames: FrameStore,
        store: CaptionStore | OCRStore,
    ) -> None:
        """Require text evidence to match the canonical manifest lineage."""

        canonical = Corpus._canonical_frame_store_id(frames)
        if canonical is not None and store.frame_store_id != canonical:
            raise ValueError(
                "Evidence frame_store_id does not match canonical frame_store_id: "
                f"{store.artifact_path}"
            )

    @staticmethod
    def _validate_object_counts_lineage(
        frames: FrameStore,
        store: ObjectCountsStore,
    ) -> None:
        """Require optional object counts to preserve canonical lineage."""

        canonical = Corpus._canonical_frame_store_id(frames)
        if canonical is not None and store.frame_store_id not in (None, canonical):
            raise ValueError(
                "Object counts frame_store_id does not match canonical "
                f"frame_store_id: {store.artifact_path}"
            )

    @staticmethod
    def _canonical_frame_store_id(frames: FrameStore) -> str | None:
        """Read and validate optional published lineage beside frame metadata."""

        manifest_path = frames.metadata_path.with_name("manifest.json")
        if not manifest_path.exists():
            return None
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
            return None
        if (
            not isinstance(canonical, str)
            or not canonical
            or canonical.strip() != canonical
        ):
            raise ValueError("Canonical frame manifest has invalid frame_store_id")
        return canonical


__all__ = ["Corpus"]
