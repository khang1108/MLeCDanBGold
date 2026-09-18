"""Domain package for KIS semantic intent resolution and models."""

from .models import (
    InitialEventText,
    KISEntity,
    KISEntityBinding,
    KISEvent,
    KISInitialResolution,
    KISInitialResolutionEvent,
    KISIntent,
    KISTemporalEdge,
)
from hcmai.kis.resolution import (
    GlobalRewriteEvent,
    GlobalRewriteResolution,
    KISGlobalRewriter,
    KISIntentResolver,
    KISResolutionError,
    KISScopedResolver,
    ScopedResolutionBatch,
    ScopedResolvedEvent,
    apply_scoped_resolutions,
)

__all__ = [
    "InitialEventText",
    "KISEntity",
    "KISEntityBinding",
    "KISEvent",
    "KISInitialResolution",
    "KISInitialResolutionEvent",
    "KISIntent",
    "KISIntentResolver",
    "KISResolutionError",
    "KISGlobalRewriter",
    "GlobalRewriteEvent",
    "GlobalRewriteResolution",
    "KISScopedResolver",
    "ScopedResolvedEvent",
    "ScopedResolutionBatch",
    "apply_scoped_resolutions",
    "KISTemporalEdge",
]

