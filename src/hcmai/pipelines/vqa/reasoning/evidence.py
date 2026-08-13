"""Timestamped, budgeted multimodal evidence bundles."""

from __future__ import annotations

from hcmai.common.schemas import RetrievalSource
from ..domain.models import EvidenceBundle, EvidenceItem, ParsedVQAQuery
from ..domain.ports import EvidenceLookup


def build_evidence_bundle(
    bundle: EvidenceBundle,
    data: EvidenceLookup,
    *,
    max_items: int = 24,
    max_chars: int = 4_000,
) -> EvidenceBundle:
    """Attach bounded, timestamped caption/OCR/ASR text to a scene bundle."""

    if max_items < 1 or max_chars < 1:
        raise ValueError("evidence budgets must be positive")
    items: list[EvidenceItem] = []
    seen: set[tuple[str, str]] = set()
    used_chars = 0
    for frame in bundle.image_frames:
        sources = (
            RetrievalSource.CAPTION,
            RetrievalSource.OCR,
            RetrievalSource.ASR,
        )
        for source in sources:
            value = data.get_evidence(frame.frame_id, source)
            normalized = " ".join(value.split()) if value else ""
            key = (source.value, normalized.casefold())
            if not normalized or key in seen:
                continue
            if len(items) >= max_items or used_chars + len(normalized) > max_chars:
                continue
            seen.add(key)
            used_chars += len(normalized)
            items.append(EvidenceItem(
                source=source.value, value=normalized, frame_id=frame.frame_id,
                start_ms=frame.timestamp_ms, end_ms=frame.timestamp_ms,
            ))
    warnings = () if items else ("text_evidence_unavailable",)
    return EvidenceBundle(
        scene=bundle.scene,
        image_frames=bundle.image_frames,
        items=tuple(items),
        warnings=warnings,
    )


def serialize_evidence(bundle: EvidenceBundle) -> dict[str, object]:
    """Serialize in chronological order for local or remote adapters."""

    return {
        "video_id": bundle.scene.video_id,
        "window": {"start_ms": bundle.scene.start_ms, "end_ms": bundle.scene.end_ms},
        "image_frame_ids": list(bundle.image_frame_ids),
        "items": [
            {
                "source": item.source, "value": item.value, "frame_id": item.frame_id,
                "start_ms": item.start_ms, "end_ms": item.end_ms,
                "confidence": item.confidence, "provenance": item.provenance,
            }
            for item in bundle.items
        ],
    }


def select_question_evidence(
    bundle: EvidenceBundle,
    parsed: ParsedVQAQuery,
    *,
    max_items: int,
) -> EvidenceBundle:
    """Use question intent only after localization to prioritize scene evidence."""

    preferred = {source.value for source in parsed.required_modalities}
    ordered = sorted(
        bundle.items,
        key=lambda item: (
            item.source not in preferred,
            item.start_ms,
            item.end_ms,
            item.frame_id,
        ),
    )[:max_items]
    return EvidenceBundle(
        scene=bundle.scene,
        image_frames=tuple(sorted(
            bundle.image_frames,
            key=lambda frame: (frame.timestamp_ms, frame.frame_idx, frame.frame_id),
        )),
        items=tuple(ordered),
        warnings=bundle.warnings,
    )
