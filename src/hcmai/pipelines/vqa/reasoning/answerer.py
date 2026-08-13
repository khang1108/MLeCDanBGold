"""Budgeted multi-frame VQA orchestration with strict identity validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, cast

from PIL import Image

from hcmai.common.schemas import VQAInferenceEvidence, VQAInferenceEvidenceItem
from ..domain.models import (
    EvidenceBundle,
    GroundedAnswerCandidate,
    ParsedVQAQuery,
    QuestionType,
)
from ..domain.ports import AnswerData
from ..legacy_localization.windows import expand_neighbor_window
from ..query.normalization import normalize_answer
from .evidence import build_evidence_bundle


class VQAService(Protocol):
    """Minimal inference capability consumed by the VQA answer stage."""

    def answer_vqa(self, *args: Any, **kwargs: Any) -> Any:
        """Answer a question from one selected evidence frame."""

        ...


ImageLoader = Callable[[str], Any]


def answer_windows(
    localized: list[EvidenceBundle],
    parsed: ParsedVQAQuery,
    llm: VQAService,
    *,
    max_calls: int = 8,
    image_loader: ImageLoader | None = None,
    data: AnswerData | None = None,
    retry_threshold: float = 0.4,
) -> tuple[list[GroundedAnswerCandidate], list[str]]:
    """Answer bounded localized scenes and collect deterministic warnings."""

    if max_calls < 1:
        raise ValueError("max_calls must be positive")
    load = image_loader or _load_image
    answers: list[GroundedAnswerCandidate] = []
    warnings: list[str] = []
    calls = 0
    for localized_window in localized:
        if calls >= max_calls:
            break
        candidate, error = _answer_one(
            localized_window, parsed, llm, load, data
        )
        calls += 1
        if error:
            warnings.append(error)
        if candidate is not None:
            answers.append(candidate)
        needs_retry = (
            parsed.question_type == QuestionType.TEMPORAL
            and data is not None
            and (candidate is None or candidate.answer_confidence < retry_threshold)
            and calls < max_calls
        )
        if needs_retry:
            assert data is not None
            expanded = expand_neighbor_window(localized_window, data)
            if expanded is not None:
                retry_bundle = build_evidence_bundle(expanded, data)
                retry_localized = replace(
                    retry_bundle,
                    scene=retry_bundle.scene.model_copy(update={
                        "final_score": localized_window.scene.final_score,
                        "reason_labels": ("temporal_neighbor_fallback",),
                    }),
                )
                retry, retry_error = _answer_one(
                    retry_localized, parsed, llm, load, data
                )
                calls += 1
                warnings.append("temporal_neighbor_fallback")
                if retry_error:
                    warnings.append(retry_error)
                if retry is not None:
                    answers.append(retry)
    return answers, list(dict.fromkeys(warnings))


def _answer_one(
    bundle: EvidenceBundle,
    parsed: ParsedVQAQuery,
    llm: VQAService,
    load: ImageLoader,
    data: AnswerData | None,
):
    """Answer one scene while enforcing supplied canonical frame identity."""

    if not bundle.image_frames:
        return None, "empty_evidence_window"
    frame = max(
        bundle.image_frames,
        key=lambda item: (
            item.frame_id
            in {
                source.frame.frame_id
                for source in bundle.scene.evidence
            },
            -abs(
                item.timestamp_ms
                - (bundle.scene.start_ms + bundle.scene.end_ms) // 2
            ),
            -item.frame_idx,
        ),
    )
    evidence = _provider_evidence(bundle.items)
    images = []
    try:
        resolve = getattr(data, "resolve_frame_asset", None)
        for image_frame in bundle.image_frames:
            image_path = (
                resolve(image_frame)
                if callable(resolve)
                else image_frame.image_path
            )
            images.append(load(str(image_path)))
        multi = getattr(llm, "answer_vqa_multi", None)
        response = None
        # Prefer chronological multi-frame inference. Providers without this
        # capability fall back deterministically to the representative frame.
        if callable(multi) and len(images) > 1:
            try:
                response = multi(
                    request_id=f"vqa:{bundle.scene.scene_id}:multi",
                    video_id=bundle.scene.video_id,
                    frame_ids=list(bundle.image_frame_ids),
                    scene_context=parsed.retrieval_query,
                    question=parsed.question,
                    images=images,
                    evidence=evidence,
                )
            except TypeError:
                response = multi(
                    parsed.question,
                    images,
                    list(bundle.image_frame_ids),
                    evidence,
                )
            except RuntimeError:
                response = None
        if response is None:
            try:
                response = llm.answer_vqa(
                    request_id=f"vqa:{bundle.scene.scene_id}:{frame.frame_id}",
                    frame_id=frame.frame_id,
                    video_id=frame.video_id,
                    scene_context=parsed.retrieval_query,
                    question=parsed.question,
                    image=images[bundle.image_frames.index(frame)],
                    evidence=evidence,
                )
            except TypeError:
                # Local and legacy adapters retain the three-argument method.
                response = llm.answer_vqa(
                    parsed.question,
                    images[bundle.image_frames.index(frame)],
                    evidence,
                )
    except FileNotFoundError:
        return None, f"frame_asset_missing(frame_id={frame.frame_id})"
    except (TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
        return None, f"vqa_provider_{type(exc).__name__.lower()}"
    finally:
        for image in images:
            close = getattr(image, "close", None)
            if callable(close):
                close()
    values = _response_values(response)
    returned_video = values.get("video_id")
    if returned_video is not None and returned_video != bundle.scene.video_id:
        return None, "provider_returned_unknown_video_id"
    returned_frame = (
        values.get("selected_frame_id")
        or values.get("selected_evidence_frame_id")
        or values.get("frame_id")
        or frame.frame_id
    )
    if returned_frame not in bundle.image_frame_ids:
        return None, "provider_returned_unknown_frame_id"
    answer = str(values.get("answer", "")).strip()
    answerable = bool(values.get(
        "answerable",
        values.get("answerability", values.get("grounded", True)),
    ))
    if not answer or not answerable:
        return None, "provider_returned_unanswerable"
    confidence = _confidence(
        values.get("confidence", values.get("answer_confidence", 0.5))
    )
    source = next(
        (
            item
            for item in bundle.scene.evidence
            if item.frame.frame_id == returned_frame
        ),
        None,
    )
    retrieval_scene_score = (
        max(item.score for item in bundle.scene.evidence)
        + bundle.scene.coverage_score
    )
    frame_score = source.score if source else retrieval_scene_score
    coverage = len({item.source for item in bundle.items}) / 4.0
    return GroundedAnswerCandidate(
        scene=bundle.scene,
        evidence_frame_id=str(returned_frame),
        answer=answer,
        normalized_answer=normalize_answer(answer, parsed.question_type),
        video_score=max(item.score for item in bundle.scene.evidence),
        frame_score=frame_score,
        localization_score=bundle.scene.final_score,
        evidence_coverage_score=min(1.0, coverage),
        answer_confidence=confidence,
        warnings=bundle.warnings,
    ), None


def _provider_evidence(items) -> VQAInferenceEvidence:
    """Group timestamped scene items into the provider evidence contract."""

    grouped: dict[str, list[str]] = {"caption": [], "ocr": [], "asr": []}
    for item in items:
        key = "ocr" if item.source == "ocr" else item.source
        if key in grouped:
            grouped[key].append(item.value)
    return VQAInferenceEvidence(
        items=[
            VQAInferenceEvidenceItem(
                source=item.source,
                value=item.value,
                frame_id=item.frame_id,
                start_ms=item.start_ms,
                end_ms=item.end_ms,
                confidence=item.confidence,
                provenance=item.provenance,
            )
            for item in items
        ],
        caption=" | ".join(grouped["caption"]) or None,
        ocr_text=" | ".join(grouped["ocr"]) or None,
        asr_text=" | ".join(grouped["asr"]) or None,
    )


def _response_values(response: Any) -> dict[str, Any]:
    """Normalize dictionary and Pydantic provider responses for validation."""

    if isinstance(response, dict):
        return response
    dump = getattr(response, "model_dump", None)
    if callable(dump):
        return cast(dict[str, Any], dump())
    fields = (
        "answer",
        "answerable",
        "video_id",
        "frame_ids",
        "frame_id",
        "grounded",
        "confidence",
        "answer_confidence",
        "answerability",
        "selected_evidence_frame_id",
        "selected_frame_id",
    )
    return {
        name: getattr(response, name)
        for name in fields
        if hasattr(response, name)
    }


def _confidence(value: Any) -> float:
    """Coerce provider confidence into the public [0, 1] range."""

    if isinstance(value, str):
        value = {"low": 0.25, "medium": 0.6, "high": 0.9}.get(
            value.casefold(),
            0.5,
        )
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _load_image(path: str) -> Image.Image:
    """Load one image as detached RGB data so the file closes immediately."""

    with Image.open(Path(path)) as image:
        return image.convert("RGB")
