"""Timestamped, budgeted multimodal evidence bundles."""

from __future__ import annotations

from hcmai.common.schemas import RetrievalSource
from .contracts import EvidenceLookup
from .models import EvidenceBundle, EvidenceItem, TemporalWindow


def build_evidence_bundle(
    window: TemporalWindow,
    data: EvidenceLookup,
    *,
    max_items: int = 24,
    max_chars: int = 4_000,
) -> EvidenceBundle:
    if max_items < 1 or max_chars < 1:
        raise ValueError("evidence budgets must be positive")
    items: list[EvidenceItem] = []
    seen: set[tuple[str, str]] = set()
    used_chars = 0
    for frame in window.sampled_frames:
        for source in (RetrievalSource.CAPTION, RetrievalSource.OCR, RetrievalSource.ASR):
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
        window=window,
        items=tuple(items),
        image_frame_ids=window.frame_ids,
        warnings=warnings,
    )


def serialize_evidence(bundle: EvidenceBundle) -> dict[str, object]:
    """Serialize in chronological order for local or remote adapters."""

    return {
        "video_id": bundle.window.video_id,
        "window": {"start_ms": bundle.window.start_ms, "end_ms": bundle.window.end_ms},
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
