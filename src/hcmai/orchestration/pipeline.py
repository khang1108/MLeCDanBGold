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
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from hcmai.api.contracts.kis import (
    EventPatch,
    KISExplorationEventSeed,
    KISExplorationSeed,
    KISOperationSummary,
    KISSearchRequest,
    KISSearchResponse,
    KISSearchResult,
)
from hcmai.common.config import ApiConfig, SearchConfig
from hcmai.corpus import Corpus
from hcmai.corpus.models import Frame
from hcmai.event_trail.config import EventTrailSettings
from hcmai.event_trail.decoder import TemporalConstraintDecoder
from hcmai.event_trail.errors import EventTrailError
from hcmai.event_trail.models import EvidenceSnapshot, SnapshotResult, freeze_video_scores
from hcmai.event_trail.service import EventTrailService
from hcmai.event_trail.store import EventTrailSessionStore, EvidenceSnapshotStore
from hcmai.orchestration.utils.errors import (
    InvalidQueryInputError,
    RevisionConflictError,
    SearchServiceGatewayError,
)
from hcmai.orchestration.utils.health import build_health_report
from hcmai.orchestration.utils.materializer import SearchMaterializer
from hcmai.orchestration.workflows.image_search import (
    ImageQueryTooLargeError,
    ImageSearchService,
)
from hcmai.orchestration.workflows.temporal_search import (
    TemporalSearchGateway,
    TemporalSearchService,
)
from hcmai.orchestration.workflows.kis import KISPipeline
from hcmai.orchestration.workflows.trake import TRAKEPipeline
from hcmai.retrieval.evidence.image_query import ImageQueryTemporalScorer
from hcmai.retrieval_service.errors import (
    RetrievalClientError,
    RetrievalInvalidRequestError,
    RetrievalNotFoundError,
    RetrievalProtocolError,
    RetrievalTooLargeError,
    RetrievalUnavailableError,
)
from hcmai.kis.models import (
    DEFAULT_MAX_TEMPORAL_EVENT_COUNT,
    KISEvent,
    KISImageRef,
    KISIntent,
    KISTemporalEdge,
)
from hcmai.kis.parser import EventPatchInstruction
from hcmai.kis.scoped_resolver import (
    apply_scoped_resolutions,
    canonical_query_text,
)
from hcmai.retrieval.plan import KISRetrievalEvent, KISRetrievalPlan

if TYPE_CHECKING:
    from hcmai.kis.assets import KISImageAssetStore
    from hcmai.kis.resolver import KISIntentResolver
    from hcmai.kis.rewriter import KISGlobalRewriter
    from hcmai.kis.scoped_resolver import KISScopedResolver
    from hcmai.retrieval.evidence.hybrid import TemporalEvidenceScorer
    from hcmai.retrieval.evidence.literal import LiteralTextIndex
    from hcmai.retrieval.embedding.models.contracts import ImageEmbeddingAdapter
    from hcmai.retrieval.retriever.models.contracts import VectorRetriever
    from hcmai.retrieval.retriever.pipeline import RetrievalService
    from hcmai.retrieval_service.client import RetrievalGrpcClient
    from hcmai.retrieval_service.remote import (
        RemoteImageSearchService,
        RemoteTemporalSearchService,
    )
    from llm.pipeline import LLMService


class SearchServiceUnavailableError(RuntimeError):
    """A required configured search dependency is unavailable."""


class SearchService:
    """Expose image, KIS, TRAKE, and literal search over shared runtime data."""

    def __init__(
        self,
        corpus: Corpus | None,
        config: SearchConfig | None = None,
        temporal: TemporalSearchGateway | None = None,
        image_search: RemoteImageSearchService | ImageSearchService | None = None,
        remote_retrieval: RetrievalGrpcClient | None = None,
        api_config: ApiConfig | None = None,
        literal_text: LiteralTextIndex | None = None,
        intent_resolver: KISIntentResolver | None = None,
        scoped_resolver: KISScopedResolver | None = None,
        global_rewriter: KISGlobalRewriter | None = None,
        kis_image_assets: KISImageAssetStore | None = None,
        event_trail_settings: EventTrailSettings | None = None,
        *,
        retrieval: RetrievalService | None = None,
        temporal_evidence: TemporalEvidenceScorer | None = None,
        image_encoder: ImageEmbeddingAdapter | None = None,
        visual_retriever: VectorRetriever | None = None,
        llm: LLMService | None = None,
    ) -> None:
        """Initialize explicit task workflows over one temporal service."""

        self.corpus = corpus
        self.retrieval = retrieval
        self.config = config or SearchConfig()
        self.llm = llm
        self.literal_text = literal_text
        self.temporal_evidence = temporal_evidence
        self.api_config = api_config or ApiConfig()
        self.intent_resolver = intent_resolver
        self.scoped_resolver = scoped_resolver
        self.global_rewriter = global_rewriter
        self.kis_image_assets = kis_image_assets
        self.remote_retrieval = remote_retrieval
        self.event_trail_settings = event_trail_settings or EventTrailSettings.from_env()
        self.event_trail_snapshots = EvidenceSnapshotStore(
            ttl_seconds=self.event_trail_settings.snapshot_ttl_seconds,
            max_entries=self.event_trail_settings.max_snapshots,
        )

        if image_search is not None:
            self.image_search = image_search
        elif (
            corpus is not None
            and visual_retriever is not None
            and image_encoder is not None
        ):
            self.image_search = ImageSearchService(
                corpus,
                visual_retriever,
                image_encoder,
                max_upload_bytes=self.api_config.image_max_upload_bytes,
                max_pixels=self.api_config.image_max_pixels,
            )
        else:
            self.image_search = None

        if (
            visual_retriever is not None
            and image_encoder is not None
            and kis_image_assets is not None
        ):
            self.image_query_scorer = ImageQueryTemporalScorer(
                visual_index=visual_retriever.index,
                image_encoder=image_encoder,
                asset_store=kis_image_assets,
                chunk_size=self.config.alignment.chunk_size,
            )
        else:
            self.image_query_scorer = None

        if temporal is not None:
            self.temporal = temporal
        elif self.corpus is not None and self.temporal_evidence is not None:
            self.temporal = TemporalSearchService(
                self.corpus,
                self.temporal_evidence,
                self.config.alignment,
                self.config.max_temporal_event_count,
            )
        else:
            self.temporal = None

        self.kis = KISPipeline(
            self.corpus,
            self.temporal,
            self.config.max_temporal_event_count,
            image_scorer=self.image_query_scorer,
        )
        self.trake = TRAKEPipeline(
            self.temporal,
            self.config.max_temporal_event_count,
        )

        self.event_trail_sessions = EventTrailSessionStore(
            ttl_seconds=self.event_trail_settings.session_ttl_seconds,
            max_entries=self.event_trail_settings.max_sessions,
        )
        if self.kis.temporal is not None:
            self.event_trail_decoder = TemporalConstraintDecoder(self.kis.temporal)
            self.event_trail = EventTrailService(
                snapshot_store=self.event_trail_snapshots,
                session_store=self.event_trail_sessions,
                decoder=self.event_trail_decoder,
            )
        else:
            self.event_trail_decoder = None
            self.event_trail = None

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
        """Close optional inference resources and remote gRPC client."""

        if self.llm is not None:
            self.llm.close()
        if self.remote_retrieval is not None:
            self.remote_retrieval.close()

    def _canonical_image_refs(
        self, refs: Sequence[KISImageRef | str]
    ) -> list[KISImageRef]:
        """Resolve incoming image references through the canonical asset store.

        This ensures user-uploaded client metadata (such as client-provided
        content_type) is replaced with the store's authoritative metadata.
        """
        if not refs:
            return []
        if self.kis_image_assets is None:
            raise SearchServiceUnavailableError("KIS image asset store is unavailable")
        canonical = []
        for ref in refs:
            asset_id = ref if isinstance(ref, str) else ref.asset_id
            try:
                canonical.append(self.kis_image_assets.ref(asset_id))
            except (KeyError, ValueError) as exc:
                raise InvalidQueryInputError(f"Unknown image asset: {asset_id}") from exc
        return canonical

    def _resolve_operation(
        self, request: KISSearchRequest
    ) -> tuple[KISIntent, KISOperationSummary, float]:
        """Resolve the requested semantic operation into the canonical KISIntent."""
        # 1. Base intent and revision boundary validation before inference
        if request.base_intent is None:
            if request.expected_revision != 0:
                raise RevisionConflictError(
                    f"Expected revision {request.expected_revision} must be 0 when base_intent is None"
                )
            if request.operation.kind != "initial_resolve":
                raise RevisionConflictError(
                    f"Operation '{request.operation.kind}' requires an existing base_intent"
                )
        else:
            if request.expected_revision != request.base_intent.revision:
                raise RevisionConflictError(
                    f"Expected revision {request.expected_revision} does not match base {request.base_intent.revision}"
                )
            if request.operation.kind == "initial_resolve":
                raise RevisionConflictError(
                    "initial_resolve cannot be performed with an existing base_intent"
                )

        # 2. initial_resolve
        if request.operation.kind == "initial_resolve":
            op = request.operation
            # Natural route
            if not op.patches:
                if op.text is not None and not op.image_refs:
                    # Natural text only -> full initial resolver
                    if self.intent_resolver is None:
                        raise SearchServiceUnavailableError("KIS intent resolver is unavailable")
                    t0 = perf_counter()
                    intent = self.intent_resolver.resolve_initial(op.text, revision=1)
                    intent_ms = (perf_counter() - t0) * 1_000.0
                    summary = KISOperationSummary(
                        kind="initial_resolve",
                        affected_event_ids=[event.id for event in intent.events],
                    )
                    return intent, summary, intent_ms

                if op.text is None and op.image_refs:
                    # Image only -> deterministic E1, no LLM call
                    event = KISEvent(
                        id="E1",
                        text=None,
                        images=self._canonical_image_refs(op.image_refs),
                        bindings=[],
                    )
                    intent = KISIntent(
                        revision=1,
                        query_text=None,
                        entities=[],
                        events=[event],
                        temporal_edges=[],
                    )
                    summary = KISOperationSummary(kind="initial_resolve", affected_event_ids=["E1"])
                    return intent, summary, 0.0

                if op.text is not None and op.image_refs:
                    # Text + image -> scoped resolver for E1, attach image deterministically
                    if self.scoped_resolver is None:
                        raise SearchServiceUnavailableError("KIS scoped resolver is unavailable")
                    t0 = perf_counter()
                    batch = self.scoped_resolver.resolve(
                        base=None,
                        instructions=[EventPatchInstruction(event_id="E1", instruction=op.text)],
                    )
                    intent_ms = (perf_counter() - t0) * 1_000.0
                    base_resolved = apply_scoped_resolutions(base=None, resolved=batch, revision=1)
                    canonical_images = self._canonical_image_refs(op.image_refs)
                    e1 = base_resolved.events[0].model_copy(update={"images": canonical_images})
                    intent = base_resolved.model_copy(update={"events": [e1]})
                    summary = KISOperationSummary(kind="initial_resolve", affected_event_ids=["E1"])
                    return intent, summary, intent_ms

            # Explicit patches route E1..Ek
            expected_patch_ids = [f"E{i + 1}" for i in range(len(op.patches))]
            actual_patch_ids = [p.event_id for p in op.patches]
            if actual_patch_ids != expected_patch_ids:
                raise InvalidQueryInputError(
                    f"Explicit initial patches must be contiguous starting at E1, got {actual_patch_ids}"
                )
            for p in op.patches:
                if p.remove_image_ids:
                    raise InvalidQueryInputError("Cannot remove images during initial resolve")
                if not p.instruction and not p.add_image_ids:
                    raise InvalidQueryInputError(f"Patch {p.event_id} requires text instruction or image")

            # Resolve image assets
            patch_images: dict[str, list[KISImageRef]] = {
                p.event_id: self._canonical_image_refs(p.add_image_ids)
                for p in op.patches
            }

            text_patches = [p for p in op.patches if p.instruction]
            if text_patches:
                if self.scoped_resolver is None:
                    raise SearchServiceUnavailableError("KIS scoped resolver is unavailable")
                t0 = perf_counter()
                batch = self.scoped_resolver.resolve(
                    base=None,
                    instructions=[EventPatchInstruction(event_id=p.event_id, instruction=p.instruction) for p in text_patches],
                )
                intent_ms = (perf_counter() - t0) * 1_000.0
                resolved_by_id = {e.event_id: e for e in batch.events}
                events = []
                for p in op.patches:
                    txt = resolved_by_id[p.event_id].text if p.event_id in resolved_by_id else None
                    events.append(KISEvent(id=p.event_id, text=txt, images=patch_images[p.event_id], bindings=[]))
                q_text = canonical_query_text(events)
                edges = [KISTemporalEdge(source=f"E{i}", target=f"E{i + 1}") for i in range(1, len(events))]
                intent = KISIntent(
                    revision=1,
                    query_text=q_text,
                    entities=[],
                    events=events,
                    temporal_edges=edges,
                )
            else:
                intent_ms = 0.0
                events = [
                    KISEvent(id=p.event_id, text=None, images=patch_images[p.event_id], bindings=[])
                    for p in op.patches
                ]
                edges = [KISTemporalEdge(source=f"E{i}", target=f"E{i + 1}") for i in range(1, len(events))]
                intent = KISIntent(
                    revision=1,
                    query_text=None,
                    entities=[],
                    events=events,
                    temporal_edges=edges,
                )
            summary = KISOperationSummary(kind="initial_resolve", affected_event_ids=actual_patch_ids)
            return intent, summary, intent_ms

        # 3. patch_events
        if request.operation.kind == "patch_events":
            op = request.operation
            base = request.base_intent
            assert base is not None

            patch_ids = [p.event_id for p in op.patches]
            if len(patch_ids) != len(set(patch_ids)):
                raise InvalidQueryInputError("Duplicate patch event IDs")

            base_count = len(base.events)
            patch_numbers = [int(pid[1:]) for pid in patch_ids]
            if any(num < 1 for num in patch_numbers):
                raise InvalidQueryInputError("Event IDs must be >= E1")
            new_numbers = sorted(num for num in patch_numbers if num > base_count)
            if new_numbers and new_numbers != list(range(base_count + 1, new_numbers[-1] + 1)):
                raise InvalidQueryInputError("New event IDs must extend the timeline contiguously without gaps")

            existing_images_by_id = {e.id: {img.asset_id for img in e.images} for e in base.events}
            for p in op.patches:
                for aid in p.remove_image_ids:
                    if p.event_id not in existing_images_by_id or aid not in existing_images_by_id[p.event_id]:
                        raise InvalidQueryInputError(f"Cannot remove asset {aid} from event {p.event_id}: not present")
                self._canonical_image_refs(p.add_image_ids)

            text_patches = [p for p in op.patches if p.instruction]
            if text_patches:
                if self.scoped_resolver is None:
                    raise SearchServiceUnavailableError("KIS scoped resolver is unavailable")
                t0 = perf_counter()
                batch = self.scoped_resolver.resolve(
                    base=base,
                    instructions=[EventPatchInstruction(event_id=p.event_id, instruction=p.instruction) for p in text_patches],
                )
                intent_ms = (perf_counter() - t0) * 1_000.0
                intermediate_intent = apply_scoped_resolutions(base, batch, revision=base.revision + 1)
            else:
                intent_ms = 0.0
                intermediate_intent = base.model_copy(update={"revision": base.revision + 1})

            # Assemble patch events atomically and preserve appended bindings
            assembled = {event.id: event for event in intermediate_intent.events}
            for patch in op.patches:
                previous = assembled.get(patch.event_id)
                images = list(previous.images) if previous is not None else []
                images = [image for image in images if image.asset_id not in set(patch.remove_image_ids)]
                image_by_id = {image.asset_id: image for image in images}
                for image in self._canonical_image_refs(patch.add_image_ids):
                    image_by_id.setdefault(image.asset_id, image)
                text = previous.text if previous is not None else None
                bindings = list(previous.bindings) if previous is not None else []
                if text is None and not image_by_id:
                    raise InvalidQueryInputError(f"Event {patch.event_id} requires text or image evidence")
                assembled[patch.event_id] = KISEvent(
                    id=patch.event_id,
                    text=text,
                    images=list(image_by_id.values()),
                    bindings=bindings,
                )

            final_events = [assembled[f"E{i}"] for i in range(1, len(assembled) + 1)]
            edges = [KISTemporalEdge(source=f"E{i}", target=f"E{i + 1}") for i in range(1, len(final_events))]
            q_text = canonical_query_text(final_events)
            intent = intermediate_intent.model_copy(
                update={"events": final_events, "temporal_edges": edges, "query_text": q_text}
            )
            summary = KISOperationSummary(kind="patch_events", affected_event_ids=patch_ids)
            return intent, summary, intent_ms

        # 4. global_rewrite
        if request.operation.kind == "global_rewrite":
            op = request.operation
            base = request.base_intent
            assert base is not None
            if self.global_rewriter is None:
                raise SearchServiceUnavailableError("KIS global rewriter is unavailable")
            t0 = perf_counter()
            intent = self.global_rewriter.rewrite(base=base, instruction=op.instruction)
            intent_ms = (perf_counter() - t0) * 1_000.0
            summary = KISOperationSummary(
                kind="global_rewrite",
                affected_event_ids=[e.id for e in intent.events],
            )
            return intent, summary, intent_ms

        # 5. search_only
        if request.operation.kind == "search_only":
            base = request.base_intent
            assert base is not None
            summary = KISOperationSummary(kind="search_only", affected_event_ids=[])
            return base, summary, 0.0

        raise InvalidQueryInputError(f"Unsupported operation kind: {request.operation.kind}")

    def search_kis(self, request: KISSearchRequest) -> KISSearchResponse:
        """Execute a stateless semantic KIS search."""
        intent, summary, intent_ms = self._resolve_operation(request)

        self._ensure_search_ready()

        dense_map = {
            index: event.text
            for index, event in enumerate(intent.events)
            if request.use_dense and event.text is not None
        }

        plan = KISRetrievalPlan(
            events=tuple(
                KISRetrievalEvent(
                    event_id=event.id,
                    canonical_text=event.text,
                    dense_text=dense_map.get(index) if request.use_dense else None,
                    bm25_text=event.text if request.use_bm25 else None,
                    image_refs=tuple(event.images),
                )
                for index, event in enumerate(intent.events)
            )
        )
        if plan.event_ids != tuple(event.id for event in intent.events):
            raise ValueError("retrieval plan event IDs must match intent event order")

        try:
            execution = self.kis.execute(
                intent=intent,
                retrieval_plan=plan,
                use_dense=request.use_dense,
                use_bm25=request.use_bm25,
                top_k=request.top_k,
                intent_ms=intent_ms,
            )
        except RetrievalUnavailableError as error:
            raise SearchServiceUnavailableError(str(error)) from error
        except RetrievalInvalidRequestError as error:
            raise InvalidQueryInputError(str(error)) from error
        except RetrievalNotFoundError as error:
            raise KeyError(str(error)) from error
        except RetrievalTooLargeError as error:
            raise ImageQueryTooLargeError(str(error)) from error
        except (RetrievalProtocolError, RetrievalClientError) as error:
            raise SearchServiceGatewayError(str(error)) from error

        result_ids = [f"r_{uuid4().hex}" for _ in execution.results]
        kis_results = [
            KISSearchResult(result_id=rid, **result.model_dump())
            for rid, result in zip(result_ids, execution.results, strict=True)
        ]

        warnings: list[str] = []
        snapshot_id: str | None = None
        snapshot_ms: float = 0.0

        if execution.temporal_artifact is not None:
            artifact = execution.temporal_artifact
            if len(execution.results) != len(artifact.result.paths):
                raise ValueError(
                    f"Mismatch between execution results count ({len(execution.results)}) "
                    f"and artifact paths count ({len(artifact.result.paths)})"
                )

            scoring_revision = artifact.scoring_revision or (
                getattr(artifact.result, "scoring_revision", None)
                if artifact.result
                else None
            )
            if not scoring_revision or not scoring_revision.strip():
                raise SearchServiceGatewayError(
                    "Temporal search artifact missing nonblank scoring revision."
                )

            snapshot_started = perf_counter()
            try:
                try:
                    score_map = {v.video_id: v for v in artifact.video_scores}
                    returned_video_ids = {r.video_id for r in execution.results}
                    video_evidence = {
                        vid: freeze_video_scores(score_map[vid])
                        for vid in returned_video_ids
                    }
                except MemoryError as exc:
                    raise EventTrailError(
                        "SNAPSHOT_UNAVAILABLE",
                        "Insufficient memory to snapshot temporal evidence",
                    ) from exc

                created_at = datetime.now(timezone.utc)
                expires_at = created_at + timedelta(
                    seconds=self.event_trail_settings.snapshot_ttl_seconds
                )
                sid = f"snap_{uuid4().hex}"

                snapshot_results = {
                    rid: SnapshotResult(
                        result_id=rid,
                        video_id=result.video_id,
                        initial_path=tuple(path.frame_ids),
                        path_score=path.score,
                    )
                    for rid, result, path in zip(
                        result_ids, execution.results, artifact.result.paths, strict=True
                    )
                }

                snapshot = EvidenceSnapshot(
                    snapshot_id=sid,
                    kis_revision=intent.revision,
                    scoring_revision=scoring_revision,
                    event_ids=tuple(event.id for event in intent.events),
                    decoder_config=artifact.decoder_config,
                    results=snapshot_results,
                    video_evidence=video_evidence,
                    created_at=created_at,
                    expires_at=expires_at,
                )
                self.event_trail_snapshots.put(snapshot)
                snapshot_id = sid
            except EventTrailError as exc:
                if exc.code == "SNAPSHOT_UNAVAILABLE":
                    snapshot_id = None
                    warnings.append("EVENT_TRAIL_UNAVAILABLE")
                else:
                    raise
            finally:
                snapshot_ms = (perf_counter() - snapshot_started) * 1_000.0

        latency = execution.latency.model_copy(
            update={
                "snapshot_ms": snapshot_ms,
                "total_ms": execution.latency.total_ms + snapshot_ms,
            }
        )

        return KISSearchResponse(
            intent=intent,
            operation_summary=summary,
            exploration_seed=KISExplorationSeed(
                semantic_revision=intent.revision,
                events=[
                    KISExplorationEventSeed(
                        event_id=event.event_id,
                        canonical_text=event.canonical_text,
                        dense_text=event.dense_text,
                        bm25_text=event.bm25_text,
                        image_refs=list(event.image_refs),
                    )
                    for event in plan.events
                ],
                use_dense=request.use_dense,
                use_bm25=request.use_bm25,
            ),
            use_dense=request.use_dense,
            use_bm25=request.use_bm25,
            results=kis_results,
            latency=latency,
            evidence_snapshot_id=snapshot_id,
            warnings=warnings,
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
        try:
            return self.image_search.search(
                payload,
                content_type=content_type,
                top_k=top_k,
            )
        except RetrievalUnavailableError as error:
            raise SearchServiceUnavailableError(str(error)) from error
        except RetrievalInvalidRequestError as error:
            raise InvalidQueryInputError(str(error)) from error
        except RetrievalNotFoundError as error:
            raise KeyError(str(error)) from error
        except RetrievalTooLargeError as error:
            raise ImageQueryTooLargeError(str(error)) from error
        except (RetrievalProtocolError, RetrievalClientError) as error:
            raise SearchServiceGatewayError(str(error)) from error

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
        try:
            return self.trake.execute(request)
        except RetrievalUnavailableError as error:
            raise SearchServiceUnavailableError(str(error)) from error
        except RetrievalInvalidRequestError as error:
            raise InvalidQueryInputError(str(error)) from error
        except RetrievalNotFoundError as error:
            raise KeyError(str(error)) from error
        except RetrievalTooLargeError as error:
            raise ImageQueryTooLargeError(str(error)) from error
        except (RetrievalProtocolError, RetrievalClientError) as error:
            raise SearchServiceGatewayError(str(error)) from error

    def _ensure_search_ready(self) -> None:
        """Reject online search when canonical data or retrieval is unavailable."""

        missing: list[str] = []
        if self.corpus is None:
            missing.append("canonical frame data")
        if self.temporal is None:
            missing.append("temporal search gateway")
        if missing:
            raise SearchServiceUnavailableError(
                f"Search dependencies not loaded: {', '.join(missing)}"
            )
