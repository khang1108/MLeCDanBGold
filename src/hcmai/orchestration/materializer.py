"""Materialize KIS HTTP results from canonical aligned paths.

This module resolves representative-frame metadata and backend-owned asset
URLs from loaded data stores. It does not retrieve, rerank, or alter temporal
alignment paths.
"""

from __future__ import annotations

from urllib.parse import quote

from hcmai.api.contracts import SearchResult, SearchResultMetadata
from hcmai.corpus import Corpus
from hcmai.temporal import AlignedPath


class SearchMaterializer:
    """Resolve aligned KIS paths into public representative-frame results."""

    def __init__(self, corpus: Corpus) -> None:
        """Retain the read-only Corpus facade used for materialization."""

        self.corpus = corpus

    def build_kis_result(self, path: AlignedPath) -> SearchResult:
        """Project one aligned path to its upper-middle canonical frame.

        Metadata comes only from the selected representative frame and its
        transcript segments at the representative timestamp. The complete
        aligned frame and timestamp arrays remain untouched in the result.
        """

        if not path.frame_ids:
            raise ValueError("aligned path must contain at least one frame")
        if not (
            len(path.frame_ids)
            == len(path.frame_idxs)
            == len(path.timestamps_ms)
        ):
            raise ValueError("aligned path arrays must have equal lengths")

        representative = len(path.frame_ids) // 2
        frame_id = path.frame_ids[representative]
        frame = self.corpus.frame(frame_id)

        # The organizer-provided coordinate is submission-critical. A path
        # may not silently replace it with keyframe order or any local index.
        if frame.frame_idx != path.frame_idxs[representative]:
            raise ValueError("aligned frame_idx disagrees with canonical frame")
        if frame.video_id != path.video_id:
            raise ValueError("aligned video_id disagrees with canonical frame")
        if frame.timestamp_ms != path.timestamps_ms[representative]:
            raise ValueError("aligned timestamp disagrees with canonical frame")

        thumbnail_urls = [
            self._thumbnail_url(aligned_id) for aligned_id in path.frame_ids
        ]
        # A one-millisecond half-open range retains Phase A's point-containment
        # semantics without treating nearby timeline speech as frame evidence.
        title = self.corpus.title(frame.video_id)
        caption = self.corpus.caption(frame.frame_id)
        ocr = self.corpus.ocr(frame.frame_id)
        objects = list(self.corpus.objects(frame.frame_id))
        asr = self.corpus.transcript(
            frame.video_id,
            frame.timestamp_ms,
            frame.timestamp_ms + 1,
        )

        return SearchResult(
            frame_id=frame.frame_id,
            video_id=frame.video_id,
            frame_idx=frame.frame_idx,
            timestamp_ms=frame.timestamp_ms,
            score=path.score,
            frame_ids=list(path.frame_ids),
            timestamps_ms=list(path.timestamps_ms),
            thumbnail_urls=thumbnail_urls,
            frame_url=self._frame_url(frame.frame_id),
            thumbnail_url=self._thumbnail_url(frame.frame_id),
            metadata=SearchResultMetadata(
                title=title,
                caption=caption,
                ocr=ocr,
                objects=objects,
                asr=asr,
            ),
        )

    @staticmethod
    def _thumbnail_url(frame_id: str) -> str:
        """Return the backend-owned thumbnail route for one canonical ID."""

        encoded = quote(frame_id, safe="")
        return f"/api/v1/frames/{encoded}/thumbnail"

    @staticmethod
    def _frame_url(frame_id: str) -> str:
        """Return the backend-owned full-image route for one canonical ID."""

        encoded = quote(frame_id, safe="")
        return f"/api/v1/frames/{encoded}/image"
