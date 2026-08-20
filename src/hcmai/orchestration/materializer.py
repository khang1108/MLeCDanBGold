"""Canonical retrieval-candidate response materialization."""

from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import quote

from hcmai.common.schemas import (
    RetrievalCandidate,
    RetrievalSource,
    SearchLatency,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchScores,
)
from hcmai.common.utils.video import derive_fps, format_video_id
from hcmai.data.pipeline import DataService


class SearchMaterializer:
    """Resolve every public identity through the canonical data authority."""

    def __init__(self, data: DataService) -> None:
        self.data = data

    def build_response(
        self,
        request: SearchRequest,
        candidates: Sequence[RetrievalCandidate],
        request_id: str,
    ) -> SearchResponse:
        results = [
            self.build_result(candidate, rank)
            for rank, candidate in enumerate(candidates, start=1)
        ]
        return SearchResponse(
            request_id=request_id,
            search_id=request.search_id,
            query=request.query,
            query_type=request.query_type,
            top_k=request.top_k,
            total_results=len(results),
            latency_ms=SearchLatency(total=0),
            results=results,
        )

    def build_result(
        self, candidate: RetrievalCandidate, rank: int
    ) -> SearchResult:
        frame = self.data.get_frame(candidate.frame_id)
        encoded_id = quote(candidate.frame_id, safe="")
        fields = {
            RetrievalSource.CAPTION: "caption",
            RetrievalSource.OCR: "ocr_text",
            RetrievalSource.ASR: "asr_text",
        }
        text = {
            field: self.data.get_evidence(candidate.frame_id, source)
            for source, field in fields.items()
        }

        # Calculate frame index from timestamp
        fps = derive_fps(frame)
        frame_idx = (
            frame.frame_idx
            if frame.frame_idx is not None
            else round(frame.timestamp_ms * fps / 1000.0)
        )

        # Get frame_ids from metadata
        metadata = candidate.metadata or {}
        scene_frame_ids = metadata.get("frame_ids")
        if scene_frame_ids and isinstance(scene_frame_ids, list):
            frame_ids = (
                scene_frame_ids
                if candidate.frame_id in scene_frame_ids
                else [candidate.frame_id, *scene_frame_ids]
            )
        else:
            frame_ids = [candidate.frame_id]

        return SearchResult(
            rank=rank,
            frame_ids=frame_ids,
            video_id=format_video_id(
                frame.video_id, fallback_path=getattr(frame, "image_path", None)
            ),
            frame_idx=frame_idx,
            fps=fps,
            timestamp_ms=frame.timestamp_ms,
            thumbnail_url=f"/api/v1/frames/{encoded_id}/thumbnail",
            frame_url=f"/api/v1/frames/{encoded_id}/image",
            caption=text["caption"],
            ocr_text=text["ocr_text"],
            asr_text=text["asr_text"],
            scores=_build_scores(candidate),
        )

def _build_scores(candidate: RetrievalCandidate) -> SearchScores:
    values = {
        getattr(key, "value", key): value
        for key, value in candidate.source_scores.items()
    }
    final = candidate.final_score
    if final is None:
        final = candidate.reranker_score
    if final is None:
        final = candidate.fusion_score
    if final is None:
        final = values.get("visual", 0.0)
    return SearchScores(
        visual=values.get("visual"),
        context=values.get("context"),
        caption=values.get("caption"),
        ocr=values.get("ocr"),
        asr=values.get("asr"),
        fusion=candidate.fusion_score,
        reranker=candidate.reranker_score,
        final=final,
    )
