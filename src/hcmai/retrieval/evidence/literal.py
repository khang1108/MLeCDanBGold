"""Build and query a normalized literal-text projection of the runtime corpus.

The projection keeps canonical frame order and original evidence for display.
It does not call embedding models, indexes, rerankers, or write artifacts.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Mapping

import pyarrow as pa
import pyarrow.compute as pc

from hcmai.corpus import Corpus
from hcmai.corpus.models import Frame
from hcmai.retrieval.models import RetrievalSource


_SOURCES = ("title", "caption", "ocr", "asr", "objects")


def normalize_literal_text(values: pa.Array) -> pa.Array:
    """Normalize text for accent-insensitive, case-insensitive matching."""

    text = pc.fill_null(values, "")
    text = pc.replace_substring(text, pattern="đ", replacement="d")
    text = pc.replace_substring(text, pattern="Đ", replacement="D")
    text = pc.utf8_normalize(text, form="NFD")
    text = pc.replace_substring_regex(text, pattern=r"\p{M}+", replacement="")
    text = pc.utf8_lower(text)
    return pc.utf8_trim_whitespace(
        pc.replace_substring_regex(text, pattern=r"\s+", replacement=" ")
    )


class LiteralTextIndex:
    """Search loaded Title, Caption, OCR, ASR, and Object text by substring."""

    def __init__(self, corpus: Corpus) -> None:
        """Build the in-memory projection once in canonical frame order."""

        self.frames = tuple(corpus.iter_frames())
        self._folder_ids = pa.array(
            [frame.video_id.partition("_")[0] for frame in self.frames]
        )
        self._video_ids = pa.array([frame.video_id for frame in self.frames])
        values = self._source_values(corpus)
        self._raw = {
            source: pa.array(items, type=pa.string())
            for source, items in values.items()
        }
        self._normalized = {
            source: normalize_literal_text(items)
            for source, items in self._raw.items()
        }
        self._normalized_object_counts = self._object_count_projection(corpus)
        self.available_sources = tuple(
            source for source in _SOURCES if source in self._raw
        )

    def search(
        self,
        *,
        text_filters: Mapping[str, str],
        object_filters: Mapping[str, int],
        folder_id: str | None,
        video_id: str | None,
        page_id: int,
        page_size: int,
    ) -> tuple[int, list[tuple[Frame, dict[str, str | None], dict[str, str]]]]:
        """Return one canonical-order page satisfying every active predicate.

        A predicate for a globally unavailable evidence source is omitted. A
        predicate for an available source is strict: missing evidence on a
        frame cannot satisfy it. Object labels use normalized exact equality
        and their counts must meet the requested minimum.
        """

        if not self.available_sources:
            raise RuntimeError("Literal text sources are unavailable")

        mask = pa.array([True] * len(self.frames))
        source_masks: dict[str, pa.Array] = {}
        for source, query in text_filters.items():
            values = self._normalized.get(source)
            if values is None:
                continue
            needle = normalize_literal_text(pa.array([query]))[0].as_py()
            if not needle:
                continue
            source_mask = pc.match_substring(values, pattern=needle)
            source_masks[source] = source_mask
            mask = pc.and_(mask, source_mask)

        if object_filters and "objects" in self._raw:
            normalized_filters = self._normalize_object_filters(object_filters)
            object_mask = pa.array([
                all(
                    label in counts and counts[label] >= minimum_count
                    for label, minimum_count in normalized_filters.items()
                )
                for counts in self._normalized_object_counts
            ])
            source_masks["objects"] = object_mask
            mask = pc.and_(mask, object_mask)

        if folder_id:
            mask = pc.and_(mask, pc.equal(self._folder_ids, folder_id))
        if video_id:
            mask = pc.and_(mask, pc.equal(self._video_ids, video_id))

        indices = pc.indices_nonzero(mask).to_pylist()
        start = (page_id - 1) * page_size
        hits = []
        for index in indices[start : start + page_size]:
            metadata = {
                source: self._raw[source][index].as_py() or None
                for source in self.available_sources
            }
            matches = {
                source: metadata[source]
                for source in self.available_sources
                if source in source_masks
                and source_masks[source][index].as_py()
                and metadata[source]
            }
            hits.append((self.frames[index], metadata, matches))
        return len(indices), hits

    def _object_count_projection(self, corpus: Corpus) -> tuple[dict[str, int], ...]:
        """Materialize normalized exact-label object counts beside frame order."""

        if not corpus.has_object_counts():
            return tuple({} for _ in self.frames)

        return tuple(
            self._normalize_object_counts(corpus.object_counts(frame.frame_id))
            for frame in self.frames
        )

    @staticmethod
    def _normalize_object_counts(
        object_filters: Mapping[str, int],
    ) -> dict[str, int]:
        """Normalize stored labels and preserve total multiplicity after collisions."""

        if not object_filters:
            return {}

        raw_labels = [str(label) for label in object_filters]
        normalized_labels = normalize_literal_text(pa.array(raw_labels)).to_pylist()
        normalized_counts: dict[str, int] = {}
        for raw_label, normalized_label in zip(raw_labels, normalized_labels, strict=True):
            if not normalized_label:
                continue
            normalized_counts[normalized_label] = (
                normalized_counts.get(normalized_label, 0)
                + int(object_filters[raw_label])
            )
        return normalized_counts

    @staticmethod
    def _normalize_object_filters(
        object_filters: Mapping[str, int],
    ) -> dict[str, int]:
        """Normalize request labels and retain their strictest minimum count."""

        if not object_filters:
            return {}

        raw_labels = [str(label) for label in object_filters]
        normalized_labels = normalize_literal_text(pa.array(raw_labels)).to_pylist()
        normalized_filters: dict[str, int] = {}
        for raw_label, normalized_label in zip(raw_labels, normalized_labels, strict=True):
            if not normalized_label:
                continue
            normalized_filters[normalized_label] = max(
                normalized_filters.get(normalized_label, 0),
                int(object_filters[raw_label]),
            )
        return normalized_filters

    def _source_values(self, corpus: Corpus) -> dict[str, list[str | None]]:
        """Read configured evidence while retaining raw text for result display."""

        values: dict[str, list[str | None]] = {}
        if corpus.has_titles():
            values["title"] = [corpus.title(frame.video_id) for frame in self.frames]
        if corpus.has_evidence(RetrievalSource.CAPTION):
            values["caption"] = [
                corpus.caption(frame.frame_id) for frame in self.frames
            ]
        if corpus.has_evidence(RetrievalSource.OCR):
            values["ocr"] = [corpus.ocr(frame.frame_id) for frame in self.frames]
        if corpus.has_evidence(RetrievalSource.ASR):
            values["asr"] = self._project_asr(corpus)
        if corpus.has_object_counts():
            values["objects"] = [
                ", ".join(
                    f"{label}: {count}"
                    for label, count in sorted(
                        corpus.object_counts(frame.frame_id).items()
                    )
                )
                or None
                for frame in self.frames
            ]
        return values

    def _project_asr(self, corpus: Corpus) -> list[str | None]:
        """Attach speech only to frames contained by each transcript segment."""

        by_video: defaultdict[str, list[tuple[int, Frame]]] = defaultdict(list)
        for index, frame in enumerate(self.frames):
            by_video[frame.video_id].append((index, frame))

        texts: list[list[str]] = [[] for _ in self.frames]
        for video_id, indexed_frames in by_video.items():
            indexed_frames.sort(key=lambda item: item[1].timestamp_ms)
            timestamps = [frame.timestamp_ms for _, frame in indexed_frames]
            for segment in corpus.transcript_segments_for_video(video_id):
                start = bisect_left(timestamps, segment.start_ms)
                end = bisect_left(timestamps, segment.end_ms)
                for frame_position in range(start, end):
                    frame_index = indexed_frames[frame_position][0]
                    texts[frame_index].append(segment.text.strip())
        return [" ".join(items) or None for items in texts]


__all__ = ["LiteralTextIndex", "normalize_literal_text"]
