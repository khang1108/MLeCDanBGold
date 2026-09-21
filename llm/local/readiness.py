"""Readiness projection for the process-local inference adapter.

This module observes already-constructed model owners. It does not load models
or decide which capabilities should be enabled.
"""

from __future__ import annotations

from typing import Any

from llm.contracts import InferenceReadiness


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
    text_loaded = (
        getattr(adapter, "text_encoder", None) is not None
        and getattr(adapter.text_encoder, "model", None) is not None
    )
    ocr_loaded = (
        adapter.ocr_adapter is not None and adapter.ocr_adapter.model is not None
    )
    objects_loaded = getattr(adapter, "object_detector", None) is not None
    asr_loaded = adapter.asr is not None
    diarization_loaded = adapter.diarization is not None
    transcript_config = getattr(adapter, "transcript_config", None)
    text_generator = getattr(adapter, "text_generator", None)
    text_generation_loaded = (
        text_generator is not None and getattr(text_generator, "model", None) is not None
    )
    enable_text_gen = getattr(adapter, "enable_text_generation", False)
    enable_objects = getattr(adapter, "enable_objects", False)
    return InferenceReadiness(
        ready=(not adapter.enable_caption or generator_loaded)
        and (not adapter.enable_visual_embedding or visual_loaded)
        and (not getattr(adapter, "enable_text_embedding", False) or text_loaded)
        and (not adapter.enable_ocr or ocr_loaded)
        and (not enable_objects or objects_loaded)
        and (not adapter.enable_asr or asr_loaded)
        and (not adapter.enable_diarization or diarization_loaded)
        and (not enable_text_gen or text_generation_loaded),
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
                enabled=getattr(adapter, "enable_text_embedding", False),
                loaded=text_loaded,
                checkpoint=adapter.config.caption_embedding.model_name,
                revision=adapter.config.caption_embedding.revision,
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
            "objects": _model_status(
                enabled=enable_objects,
                loaded=objects_loaded,
                checkpoint=(
                    adapter.config.object_detection.model
                    if hasattr(adapter.config, "object_detection")
                    else None
                ),
                revision=None,
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
            "text_generation": _model_status(
                enabled=enable_text_gen,
                loaded=text_generation_loaded,
                checkpoint=(
                    adapter.config.text_generation.checkpoint
                    if hasattr(adapter.config, "text_generation")
                    else None
                ),
                revision=(
                    adapter.config.text_generation.revision
                    if hasattr(adapter.config, "text_generation")
                    else None
                ),
            ),
        },
        capabilities=_capabilities(
            embedding=text_loaded,
            structured_parsing=text_generation_loaded,
            image_embedding=visual_loaded,
            caption=generator_loaded,
            ocr=ocr_loaded,
            objects=objects_loaded,
            asr=asr_loaded,
            diarization=diarization_loaded,
        ),
    )
