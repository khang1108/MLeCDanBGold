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
from .resolver import KISIntentResolver, KISResolutionError
from .rewriter import GlobalRewriteEvent, GlobalRewriteResolution, KISGlobalRewriter
from .scoped_resolver import (
    KISScopedResolver,
    ScopedResolvedEvent,
    ScopedResolutionBatch,
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

