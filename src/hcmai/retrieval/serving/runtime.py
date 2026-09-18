"""Standalone retrieval runtime composition and execution facade.

This module owns the heavy retrieval components loaded once in the retrieval
process: canonical Corpus, FAISS/vector indexes, BM25, query encoders, temporal
alignment, and direct image search.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import uuid

from hcmai.api.contracts import ImageSearchResponse
from hcmai.common.environment import load_repository_environment
from hcmai.common.utils.logging import get_logger
from hcmai.corpus import Corpus
from hcmai.orchestration.corpus_setup import load_configured_corpus
from hcmai.orchestration.retrieval_setup import (
    load_image_encoder,
    load_retrieval,
    load_temporal_evidence,
    select_visual_retriever,
)

from hcmai.orchestration.workflows.image_search import ImageSearchService
from hcmai.orchestration.workflows.temporal_search import (
    SelectedVideoScoreResult,
    TemporalSearchArtifact,
    TemporalSearchResult,
    TemporalSearchService,
)
from hcmai.retrieval.evidence.image_query import ImageQueryTemporalScorer
from hcmai.retrieval.models import RetrievalResult
from hcmai.retrieval.plan import KISRetrievalPlan
from hcmai.retrieval.retriever.pipeline import RetrievalService

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalCapabilities:
    """Diagnostic capabilities and limits advertised by a ready runtime."""

    ready: bool
    scoring_revision: str
    active_modalities: tuple[str, ...]
    startup_messages: tuple[str, ...]
    max_temporal_event_count: int
    image_max_upload_bytes: int
    image_max_pixels: int


@dataclass(frozen=True, slots=True)
class RetrievalRuntime:
    """Heavy retrieval composition facade serving requests."""

    corpus: Corpus
    temporal: TemporalSearchService
    image_scorer: ImageQueryTemporalScorer | None
    image_search: ImageSearchService | None
    active_modalities: tuple[str, ...]
    startup_messages: tuple[str, ...]
    max_temporal_event_count: int
    image_max_upload_bytes: int
    image_max_pixels: int
    scoring_revision: str
    retrieval: RetrievalService | None = None

    @classmethod
    def load(cls, messages: list[str]) -> RetrievalRuntime | None:
        """Compose all configured heavy retrieval artifacts once.

        Returns None if required visual or temporal capabilities cannot be
        loaded, while appending diagnostics to messages.
        """
        from hcmai.orchestration.setup import (
            load_app_config,
            load_kis_image_assets,
            load_model_config,
            load_remote_inference,
        )

        load_repository_environment()
        settings = load_app_config()
        models = load_model_config()

        corpus = load_configured_corpus(settings, messages)
        if corpus is None:
            messages.append("Retrieval runtime unavailable: Corpus metadata could not be loaded")
            return None

        llm = load_remote_inference(settings, messages)
        retrieval = load_retrieval(settings, models, llm, messages, corpus=corpus)
        if retrieval is None:
            messages.append("Retrieval runtime unavailable: Visual index could not be loaded")
            return None

        visual_retriever = select_visual_retriever(retrieval)
        if visual_retriever is None:
            messages.append("Retrieval runtime unavailable: Visual retriever could not be selected")
            return None

        image_encoder = load_image_encoder(models, visual_retriever, llm, messages)
        temporal_evidence = load_temporal_evidence(settings, retrieval, visual_retriever, messages)
        if temporal_evidence is None:
            messages.append("Retrieval runtime unavailable: Temporal evidence scorer could not be loaded")
            return None

        kis_image_assets = load_kis_image_assets(settings, messages)

        image_search = (
            ImageSearchService(
                corpus=corpus,
                visual_retriever=visual_retriever,
                encoder=image_encoder,
                max_upload_bytes=settings.api.image_max_upload_bytes,
                max_pixels=settings.api.image_max_pixels,
            )
            if image_encoder is not None
            else None
        )

        image_scorer = (
            ImageQueryTemporalScorer(
                visual_index=visual_retriever.index,
                image_encoder=image_encoder,
                asset_store=kis_image_assets,
                chunk_size=settings.search.alignment.chunk_size,
            )
            if (image_encoder is not None and kis_image_assets is not None)
            else None
        )

        max_temporal_event_count = settings.search.max_temporal_event_count
        temporal = TemporalSearchService(
            corpus=corpus,
            evidence=temporal_evidence,
            config=settings.search.alignment,
            max_temporal_event_count=max_temporal_event_count,
        )

        active = ["visual"]
        if temporal_evidence.context_dense_ready:
            active.append("context")
        if temporal_evidence.asr_dense_ready:
            active.append("asr")
        if temporal_evidence.bm25 is not None:
            active.append("bm25")
        if image_scorer is not None:
            active.append("image_query")
        if image_search is not None:
            active.append("image_search")

        scoring_revision = f"rev-{uuid.uuid4().hex[:12]}"

        logger.info(
            "RetrievalRuntime loaded revision=%s modalities=%s frames=%d",
            scoring_revision,
            active,
            len(corpus),
        )

        return cls(
            corpus=corpus,
            retrieval=retrieval,
            temporal=temporal,
            image_scorer=image_scorer,
            image_search=image_search,
            active_modalities=tuple(active),
            startup_messages=tuple(messages),
            max_temporal_event_count=max_temporal_event_count,
            image_max_upload_bytes=settings.api.image_max_upload_bytes,
            image_max_pixels=settings.api.image_max_pixels,
            scoring_revision=scoring_revision,
        )

    def search_text(self, query: str, *, top_k: int) -> RetrievalResult:
        if self.retrieval is None:
            raise RuntimeError("text retrieval capability is not loaded")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not query.strip():
            raise ValueError("query must not be blank")
        return self.retrieval.search(query.strip(), top_k=top_k)

    def capabilities(self) -> RetrievalCapabilities:
        """Return declared capabilities and limits without executing retrieval."""
        return RetrievalCapabilities(
            ready=True,
            scoring_revision=self.scoring_revision,
            active_modalities=self.active_modalities,
            startup_messages=self.startup_messages,
            max_temporal_event_count=self.max_temporal_event_count,
            image_max_upload_bytes=self.image_max_upload_bytes,
            image_max_pixels=self.image_max_pixels,
        )

    def search_plan(
        self,
        plan: KISRetrievalPlan,
        *,
        use_dense: bool = True,
        use_bm25: bool = False,
        top_k: int = 20,
    ) -> TemporalSearchArtifact:
        """Execute multimodal KIS retrieval plan and return bounded score artifact."""
        image_component = None
        has_any_images = any(len(ev.image_refs) > 0 for ev in plan.events)
        if has_any_images and self.image_scorer is not None:
            image_component = self.image_scorer.score_events(plan.image_ref_rows)

        artifact = self.temporal.search_plan_artifact(
            plan,
            image_component=image_component,
            use_dense=use_dense,
            use_bm25=use_bm25,
            top_k=top_k,
        )

        returned_video_ids = {path.video_id for path in artifact.result.paths}
        trimmed_scores = tuple(
            v for v in artifact.video_scores if v.video_id in returned_video_ids
        )

        return TemporalSearchArtifact(
            result=artifact.result,
            video_scores=trimmed_scores,
            decoder_config=artifact.decoder_config,
            scoring_revision=self.scoring_revision,
        )

    def search_events(
        self,
        original_events: Sequence[str],
        *,
        top_k: int,
        retrieval_events: Sequence[str] | None = None,
        caption_events: Sequence[str] | None = None,
        use_dense: bool = True,
        use_bm25: bool = False,
    ) -> TemporalSearchResult:
        """Execute ordered event temporal search (TRAKE)."""
        return self.temporal.search(
            original_events,
            top_k=top_k,
            retrieval_events=retrieval_events,
            caption_events=caption_events,
            use_dense=use_dense,
            use_bm25=use_bm25,
        )

    def score_video(
        self,
        plan: KISRetrievalPlan,
        video_id: str,
        *,
        use_dense: bool = True,
        use_bm25: bool = False,
    ) -> SelectedVideoScoreResult:
        """Score one selected video under the plan and return its snapshot."""
        image_component = None
        has_any_images = any(len(ev.image_refs) > 0 for ev in plan.events)
        if has_any_images and self.image_scorer is not None:
            image_component = self.image_scorer.score_events(plan.image_ref_rows)

        selected = self.temporal.score_video(
            plan,
            video_id=video_id,
            image_component=image_component,
            use_dense=use_dense,
            use_bm25=use_bm25,
        )
        return SelectedVideoScoreResult(
            video=selected.video,
            retrieval_ms=selected.retrieval_ms,
            decoder_config=selected.decoder_config,
            scoring_revision=self.scoring_revision,
        )

    def search_image(
        self,
        payload: bytes,
        *,
        content_type: str | None,
        top_k: int,
    ) -> ImageSearchResponse:
        """Execute direct image query search."""
        if self.image_search is None:
            raise RuntimeError("image search capability is not loaded")
        return self.image_search.search(
            payload,
            content_type=content_type,
            top_k=top_k,
        )
