"""Budgeted one-frame VQA orchestration with strict identity validation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

from PIL import Image

from hcmai.common.schemas import VQAInferenceEvidence
from .contracts import AnswerData
from .evidence import build_evidence_bundle
from .models import GroundedAnswerCandidate, LocalizedWindow, ParsedVQAQuery, QuestionType
from .normalization import normalize_answer
from .windows import expand_neighbor_window


class VQAService(Protocol):
    def answer_vqa(self, *args: Any, **kwargs: Any) -> Any: ...


ImageLoader = Callable[[str], Any]


def answer_windows(
    localized: list[LocalizedWindow],
    parsed: ParsedVQAQuery,
    llm: VQAService,
    *,
    max_calls: int = 8,
    image_loader: ImageLoader | None = None,
    data: AnswerData | None = None,
    retry_threshold: float = 0.4,
) -> tuple[list[GroundedAnswerCandidate], list[str]]:
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
            expanded = expand_neighbor_window(localized_window.bundle.window, data)
            if expanded is not None:
                retry_bundle = build_evidence_bundle(expanded, data)
                retry_localized = LocalizedWindow(retry_bundle, localized_window.score, ("temporal_neighbor_fallback",))
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
    localized: LocalizedWindow,
    parsed: ParsedVQAQuery,
    llm: VQAService,
    load: ImageLoader,
    data: AnswerData | None,
):
    bundle = localized.bundle
    if not bundle.window.sampled_frames:
        return None, "empty_evidence_window"
    frame = max(
        bundle.window.sampled_frames,
        key=lambda item: (
            item.frame_id in {source.frame.frame_id for source in bundle.window.source_frames},
            -abs(item.timestamp_ms - (bundle.window.start_ms + bundle.window.end_ms) // 2),
            -item.frame_idx,
        ),
    )
    evidence = _provider_evidence(bundle.items)
    image = None
    try:
        resolve = getattr(data, "resolve_frame_asset", None)
        image_path = resolve(frame) if callable(resolve) else frame.image_path
        image = load(str(image_path))
        try:
            response = llm.answer_vqa(
                request_id=f"vqa:{bundle.window.window_id}:{frame.frame_id}",
                frame_id=frame.frame_id,
                question=parsed.question,
                image=image,
                evidence=evidence,
            )
        except TypeError:
            # Local and legacy fake adapters retain the three-argument method.
            response = llm.answer_vqa(parsed.question, image, evidence)
    except FileNotFoundError:
        return None, f"frame_asset_missing(frame_id={frame.frame_id})"
    except (TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
        return None, f"vqa_provider_{type(exc).__name__.lower()}"
    finally:
        close = getattr(image, "close", None)
        if callable(close):
            close()
    values = _response_values(response)
    returned_frame = values.get("selected_evidence_frame_id") or values.get("frame_id") or frame.frame_id
    if returned_frame not in bundle.image_frame_ids:
        return None, "provider_returned_unknown_frame_id"
    answer = str(values.get("answer", "")).strip()
    answerable = bool(values.get("answerability", values.get("grounded", True)))
    if not answer or not answerable:
        return None, "provider_returned_unanswerable"
    confidence = _confidence(values.get("confidence", values.get("answer_confidence", 0.5)))
    source = next((item for item in bundle.window.source_frames if item.frame.frame_id == returned_frame), None)
    frame_score = source.score if source else bundle.window.score
    coverage = len({item.source for item in bundle.items}) / 4.0
    return GroundedAnswerCandidate(
        window=bundle.window, evidence_frame_id=str(returned_frame), answer=answer,
        normalized_answer=normalize_answer(answer, parsed.question_type),
        video_score=max(item.score for item in bundle.window.source_frames),
        frame_score=frame_score, localization_score=localized.score,
        evidence_coverage_score=min(1.0, coverage), answer_confidence=confidence,
        warnings=bundle.warnings,
    ), None


def _provider_evidence(items) -> VQAInferenceEvidence:
    grouped: dict[str, list[str]] = {"caption": [], "ocr": [], "asr": []}
    for item in items:
        key = "ocr" if item.source == "ocr" else item.source
        if key in grouped:
            grouped[key].append(item.value)
    return VQAInferenceEvidence(
        caption=" | ".join(grouped["caption"]) or None,
        ocr_text=" | ".join(grouped["ocr"]) or None,
        asr_text=" | ".join(grouped["asr"]) or None,
    )


def _response_values(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    dump = getattr(response, "model_dump", None)
    if callable(dump):
        return cast(dict[str, Any], dump())
    return {name: getattr(response, name) for name in (
        "answer", "frame_id", "grounded", "confidence", "answer_confidence",
        "answerability", "selected_evidence_frame_id",
    ) if hasattr(response, name)}


def _confidence(value: Any) -> float:
    if isinstance(value, str):
        value = {"low": 0.25, "medium": 0.6, "high": 0.9}.get(value.casefold(), 0.5)
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _load_image(path: str) -> Image.Image:
    with Image.open(Path(path)) as image:
        return image.convert("RGB")
