"""Public orchestration service for online competition search."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Any, TYPE_CHECKING

from hcmai.api.contracts import (
    FilterRequest,
    FilterResponse,
    FilterResult,
    FrameInspectionResponse,
    ImageSearchResponse,
    TRAKERequest,
    TRAKEResponse,
)

from hcmai.api.contracts.kis import (
    KISExplorationEventSeed,
    KISExplorationSeed,
    KISRevisionSearchRequest,
    KISRevisionSearchResponse,
)
from hcmai.common.config import ApiConfig, SearchConfig
from hcmai.corpus import Corpus
from hcmai.corpus.models import Frame
from hcmai.orchestration.errors import RevisionConflictError
from hcmai.orchestration.health import build_health_report
from hcmai.orchestration.workflows.image_search import ImageSearchService
from hcmai.orchestration.materializer import SearchMaterializer
from hcmai.orchestration.workflows.temporal_search import TemporalSearchService
from hcmai.orchestration.workflows.kis import KISPipeline
from hcmai.orchestration.workflows.trake import TRAKEPipeline
from hcmai.kis.models import KISIntent
from hcmai.retrieval.plan import KISRetrievalEvent, KISRetrievalPlan

if TYPE_CHECKING:
    from hcmai.kis.resolver import KISIntentResolver
    from hcmai.retrieval.translation.service import EventTranslator
    from hcmai.retrieval.evidence.hybrid import TemporalEvidenceScorer
    from hcmai.retrieval.evidence.literal import LiteralTextIndex
    from hcmai.retrieval.embedding.models.contracts import ImageEmbeddingAdapter
    from hcmai.retrieval.retriever.models.contracts import VectorRetriever
    from hcmai.retrieval.retriever.pipeline import RetrievalService
    from llm.pipeline import LLMService


class SearchServiceUnavailableError(RuntimeError):
    """A required configured search dependency is unavailable."""


class SearchService:
    """Expose image, KIS, TRAKE, and literal search over shared runtime data."""

    def __init__(
        self,
        corpus: Corpus | None,
        retrieval: RetrievalService | None,
        config: SearchConfig | None = None,
        llm: LLMService | None = None,
        event_translator: EventTranslator | None = None,
        temporal_evidence: TemporalEvidenceScorer | None = None,
        image_encoder: ImageEmbeddingAdapter | None = None,
        api_config: ApiConfig | None = None,
        literal_text: LiteralTextIndex | None = None,
        visual_retriever: VectorRetriever | None = None,
        intent_resolver: KISIntentResolver | None = None,
    ) -> None:
        """Initialize explicit task workflows over one temporal service."""

        self.corpus = corpus
        self.retrieval = retrieval
        self.config = config or SearchConfig()
        self.llm = llm
        self.event_translator = event_translator
        self.literal_text = literal_text
        self.temporal_evidence = temporal_evidence
        self.api_config = api_config or ApiConfig()
        self.intent_resolver = intent_resolver

        self.image_search = (
            ImageSearchService(
                corpus,
                visual_retriever,
                image_encoder,
                max_upload_bytes=self.api_config.image_max_upload_bytes,
                max_pixels=self.api_config.image_max_pixels,
            )
            if (
                corpus is not None
                and visual_retriever is not None
                and image_encoder is not None
            )
            else None
        )

        temporal = (
            TemporalSearchService(
                self.corpus,
                self.temporal_evidence,
                self.config.alignment,
                self.config.max_temporal_event_count,
            )
            if self.corpus is not None and self.temporal_evidence is not None
            else None
        )
        self.kis = KISPipeline(
            self.corpus,
            temporal,
            self.config.max_temporal_event_count,
        )
        self.trake = TRAKEPipeline(
            temporal,
            self.config.max_temporal_event_count,
        )

    @staticmethod
    def load(messages: list[str]) -> SearchService:
        """Load the configured search service and append startup diagnostics."""

        from hcmai.orchestration.setup import load_search_service

        return load_search_service(messages)

    def get_frame(self, frame_id: str) -> Frame:
        """Resolve one frame through the public canonical Corpus authority."""

        if self.corpus is None:
            raise SearchServiceUnavailableError("Frame store not loaded")
        return self.corpus.frame(frame_id)

    def inspect_frame_at_timestamp(
        self,
        video_id: str,
        timestamp_ms: int,
    ) -> FrameInspectionResponse:
        """Resolve one viewer moment and materialize its canonical evidence.

        The requested timestamp is retained separately from the resolved frame's
        timestamp so the frontend can seek precisely without corrupting
        canonical identity or pretending an arbitrary video time has a frame ID.
        """

        if self.corpus is None:
            raise SearchServiceUnavailableError("Frame store not loaded")

        frame = self.corpus.frame_at_timestamp(video_id, timestamp_ms)
        metadata = SearchMaterializer(self.corpus).build_frame_metadata(frame)
        return FrameInspectionResponse(
            requested_timestamp_ms=timestamp_ms,
            frame_id=frame.frame_id,
            video_id=frame.video_id,
            frame_idx=frame.frame_idx,
            timestamp_ms=frame.timestamp_ms,
            fps=frame.fps,
            metadata=metadata,
        )

    def health(self, startup_messages: Sequence[str] = ()) -> dict[str, Any]:
        """Delegate read-only readiness projection to the health module."""
        return build_health_report(self, startup_messages=startup_messages)

    def close(self) -> None:
        """Close optional inference resources owned by the service."""

        if self.llm is not None:
            self.llm.close()

    def search_kis_revision(
        self, request: KISRevisionSearchRequest
    ) -> KISRevisionSearchResponse:
        """Execute a revisioned KIS search using semantic intent resolution."""
        if request.has_revision_conflict:
            raise RevisionConflictError(
                f"Expected revision {request.expected_revision} does not match base {request.previous_revision}"
            )

        self._ensure_search_ready()

        if self.intent_resolver is None:
            raise SearchServiceUnavailableError("KIS intent resolver is unavailable")

        clue_texts = [item.text for item in request.inputs]
        intent_started = perf_counter()
        intent = self.intent_resolver.resolve(
            clue_texts, revision=request.expected_revision + 1
        )
        intent_ms = (perf_counter() - intent_started) * 1_000

        canonical_events = tuple(event.text for event in intent.events)
        translation_started = perf_counter()
        if request.use_dense and intent.language != "en":
            if self.event_translator is None:
                raise SearchServiceUnavailableError(
                    "Event translation capability is unavailable"
                )
            dense = self.event_translator.translate(
                canonical_events, language=intent.language,
            )
        else:
            dense = canonical_events if request.use_dense else None
        translation_ms = (perf_counter() - translation_started) * 1_000

        plan = KISRetrievalPlan(
            events=tuple(
                KISRetrievalEvent(
                    event_id=event.id,
                    canonical_text=event.text,
                    dense_text=None if dense is None else dense[index],
                    bm25_text=event.text if request.use_bm25 else None,
                )
                for index, event in enumerate(intent.events)
            )
        )
        if plan.event_ids != tuple(event.id for event in intent.events):
            raise ValueError("retrieval plan event IDs must match intent event order")

        execution = self.kis.execute(
            intent=intent,
            retrieval_plan=plan,
            use_dense=request.use_dense,
            use_bm25=request.use_bm25,
            top_k=request.top_k,
            intent_ms=intent_ms,
            translation_ms=translation_ms,
        )

        return KISRevisionSearchResponse(
            intent=intent,
            exploration_seed=KISExplorationSeed(
                semantic_revision=intent.revision,
                events=[
                    KISExplorationEventSeed(
                        event_id=event.event_id,
                        canonical_text=event.canonical_text,
                        dense_text=event.dense_text,
                        bm25_text=event.bm25_text,
                    )
                    for event in plan.events
                ],
                use_dense=request.use_dense,
                use_bm25=request.use_bm25,
            ),
            use_dense=request.use_dense,
            use_bm25=request.use_bm25,
            results=execution.results,
            latency=execution.latency,
        )

    def search_image(
        self,
        payload: bytes,
        *,
        content_type: str | None,
        top_k: int,
    ) -> ImageSearchResponse:
        """Search canonical frames using one uploaded image as visual evidence."""

        if self.image_search is None:
            raise SearchServiceUnavailableError(
                "Image search dependencies not loaded: visual image encoder"
            )
        return self.image_search.search(
            payload,
            content_type=content_type,
            top_k=top_k,
        )

    def filter_frames(self, request: FilterRequest) -> FilterResponse:
        """Filter raw evidence without semantic retrieval or reranking."""

        if self.corpus is None or self.literal_text is None:
            raise SearchServiceUnavailableError("Literal filter is unavailable")
        try:
            total, hits = self.literal_text.search(
                text_filters=request.metadata_filters.populated_text(),
                object_filters=request.metadata_filters.objects,
                folder_id=request.folder_id,
                video_id=request.video_id,
                page_id=request.page_id,
                page_size=request.frames_per_pages,
            )
        except RuntimeError as error:
            raise SearchServiceUnavailableError(str(error)) from error

        results = [
            FilterResult(
                frame_id=frame.frame_id,
                video_id=frame.video_id,
                frame_idx=frame.frame_idx,
                timestamp_ms=frame.timestamp_ms,
                fps=frame.fps,
                folder_id=frame.video_id.partition("_")[0],
                title=metadata.get("title"),
                caption=metadata.get("caption"),
                ocr=metadata.get("ocr"),
                objects=self.corpus.object_counts(frame.frame_id),
                asr=metadata.get("asr"),
                matches=matches,
            )
            for frame, metadata, matches in hits
        ]
        page_size = request.frames_per_pages
        return FilterResponse(
            page_id=request.page_id,
            frames_per_pages=page_size,
            total_pages=(total + page_size - 1) // page_size,
            total_results=total,
            available_sources=list(self.literal_text.available_sources),
            results=results,
        )

    def search_trake(self, request: TRAKERequest) -> TRAKEResponse:
        """Execute a validated TRAKE request through the explicit TRAKE workflow."""

        self._ensure_search_ready()
        return self.trake.execute(request)

    def _ensure_search_ready(self) -> None:
        """Reject online search when canonical data or retrieval is unavailable."""

        missing: list[str] = []
        if self.corpus is None:
            missing.append("canonical frame data")
        if self.temporal_evidence is None:
            missing.append("temporal evidence service")
        if missing:
            raise SearchServiceUnavailableError(
                f"Search dependencies not loaded: {', '.join(missing)}"
            )
