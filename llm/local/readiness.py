"""Readiness projection for the process-local inference adapter.

This module observes already-constructed model owners. It does not load models
or decide which capabilities should be enabled.
"""

from __future__ import annotations

from typing import Any

from offline.enrichment.inference_contracts import InferenceReadiness


def _model_status(**values: object) -> Any:
    """Return values validated by the shared readiness response model."""

    return values


def _capabilities(**values: bool) -> Any:
    """Return capability flags validated by the shared readiness model."""

    return values


def build_readiness(adapter: Any) -> InferenceReadiness:
    """Project loaded model state without mutating the local adapter."""

    generator_loaded = (
        adapter.captioner is not None and adapter.captioner.model is not None
    )
    visual_loaded = (
        adapter.visual_encoder is not None and adapter.visual_encoder.model is not None
    )
    caption_loaded = (
        adapter.caption_encoder is not None
        and adapter.caption_encoder.model is not None
    )
    reranker_loaded = (
        adapter.reranker is not None and adapter.reranker._base_model is not None
    )
    ocr_loaded = (
        adapter.ocr_adapter is not None and adapter.ocr_adapter.model is not None
    )
    asr_loaded = adapter.asr is not None
    diarization_loaded = adapter.diarization is not None
    query_preparation_loaded = (
        adapter.query_preparer is not None and adapter.query_preparer.model is not None
    )

    transcript_config = adapter.transcript_config
    return InferenceReadiness(
        ready=(not adapter.enable_caption or generator_loaded)
        and (not adapter.enable_visual_embedding or visual_loaded)
        and (not adapter.enable_caption_embedding or caption_loaded)
        and (not adapter.enable_reranker or reranker_loaded)
        and (not adapter.enable_ocr or ocr_loaded)
        and (not adapter.enable_asr or asr_loaded)
        and (not adapter.enable_diarization or diarization_loaded)
        and (not adapter.enable_query_preparation or query_preparation_loaded),
        models={
            "caption_generation": _model_status(
                enabled=adapter.enable_caption,
                loaded=generator_loaded,
                checkpoint=adapter.config.caption_generation.model_checkpoint,
                revision=(
                    adapter.captioner.resolved_revision
                    if adapter.captioner is not None
                    else None
                ),
            ),
            "visual_embedding": _model_status(
                enabled=adapter.enable_visual_embedding,
                loaded=visual_loaded,
                checkpoint=adapter.config.visual_embedding.model_name,
                revision=adapter.config.visual_embedding.revision,
            ),
            "caption_embedding": _model_status(
                enabled=adapter.enable_caption_embedding,
                loaded=caption_loaded,
                checkpoint=adapter.config.caption_embedding.model_name,
                revision=adapter.config.caption_embedding.revision,
            ),
            "reranker": _model_status(
                enabled=adapter.enable_reranker,
                loaded=reranker_loaded,
                checkpoint=adapter.config.reranker.checkpoint,
                revision=(
                    adapter.reranker.resolved_revision
                    if adapter.reranker is not None
                    else None
                ),
            ),
            "ocr": _model_status(
                enabled=adapter.enable_ocr,
                loaded=ocr_loaded,
                checkpoint=(
                    adapter.ocr_adapter.config.checkpoint
                    if adapter.ocr_adapter is not None
                    else None
                ),
                revision=(
                    adapter.ocr_adapter.config.revision
                    if adapter.ocr_adapter is not None
                    else None
                ),
            ),
            "asr": _model_status(
                enabled=adapter.enable_asr,
                loaded=asr_loaded,
                checkpoint=(
                    transcript_config.asr.model_name if transcript_config else None
                ),
                revision=(
                    transcript_config.asr.revision if transcript_config else None
                ),
            ),
            "diarization": _model_status(
                enabled=adapter.enable_diarization,
                loaded=diarization_loaded,
                checkpoint=(
                    transcript_config.diarization.model_name
                    if transcript_config
                    else None
                ),
                revision=(
                    transcript_config.diarization.revision
                    if transcript_config
                    else None
                ),
            ),
            "query_preparation": _model_status(
                enabled=adapter.enable_query_preparation,
                loaded=query_preparation_loaded,
                checkpoint=adapter.config.query_preparation.model_checkpoint,
                revision=adapter.config.query_preparation.revision,
            ),
        },
        capabilities=_capabilities(
            embedding=visual_loaded or caption_loaded,
            reranking=reranker_loaded,
            structured_parsing=False,
            image_embedding=visual_loaded,
            caption=generator_loaded,
            ocr=ocr_loaded,
            asr=asr_loaded,
            diarization=diarization_loaded,
            query_preparation=query_preparation_loaded,
        ),
    )
