"""Expose read-only runtime access to existing HCMAI corpus artifacts.

``Corpus`` composes the private specialist stores into the small runtime API
used by search callers.  It opens published artifacts only; it does not own
ingestion, enrichment, context construction, index building, or mutation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from hcmai.common.schemas import RetrievalSource
from hcmai.corpus.assets import FrameAssetResolver
from hcmai.corpus.models import Frame, TranscriptSegment
from hcmai.corpus.stores import (
    CaptionStore,
    FrameStore,
    OCRStore,
    ObjectCountsStore,
    TranscriptStore,
    VideoMetadataStore,
)


_TEXT_STORES = {
    RetrievalSource.CAPTION: CaptionStore,
    RetrievalSource.OCR: OCRStore,
}


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
        if not frame_artifact.is_file():
            raise FileNotFoundError(
                f"Frame artifact is not a file: {frame_artifact}"
            )
        frames = FrameStore(frame_artifact)
        evidence = cls._open_text_evidence(evidence_paths)
        object_counts = (
            ObjectCountsStore(object_counts_path)
            if object_counts_path is not None
            else None
        )
        transcripts = (
            TranscriptStore(transcript_path)
            if transcript_path is not None
            else None
        )
        video_metadata = (
            VideoMetadataStore(video_metadata_path)
            if video_metadata_path is not None
            else None
        )

        for store in evidence.values():
            cls._validate_evidence_identity(frames, store)
        if object_counts is not None:
            cls._validate_object_counts_identity(frames, object_counts)

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

    def frames(self, frame_ids: Sequence[str]) -> list[Frame]:
        """Return canonical frames in the requested order, including duplicates."""

        return self._frames.get_many(frame_ids)

    def caption(self, frame_id: str) -> str | None:
        """Return usable completed caption text, or ``None`` when unavailable."""

        return self._text(frame_id, RetrievalSource.CAPTION)

    def ocr(self, frame_id: str) -> str | None:
        """Return usable completed normalized OCR text, or ``None`` when absent."""

        return self._text(frame_id, RetrievalSource.OCR)

    def objects(self, frame_id: str) -> tuple[str, ...]:
        """Return stable sorted object labels from completed count evidence.

        Counts remain an offline-only representation; this runtime projection
        intentionally exposes each current label key once, including keys with
        a count of zero.
        """

        if self._object_counts is None:
            return ()
        counts = self._object_counts.get_counts(frame_id)
        return tuple(sorted(counts)) if counts is not None else ()

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

    def title(self, video_id: str) -> str | None:
        """Return organizer video title, or ``None`` when metadata is unavailable."""

        if self._video_metadata is None:
            return None
        metadata = self._video_metadata.get(video_id)
        return metadata.title if metadata is not None else None

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


__all__ = ["Corpus"]
