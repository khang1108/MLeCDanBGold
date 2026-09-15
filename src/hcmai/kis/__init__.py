"""Domain package for KIS semantic intent resolution and models."""

from .models import (
    KISEntity,
    KISEntityBinding,
    KISEvent,
    KISIntent,
    KISResolution,
    KISResolutionEntity,
    KISResolutionEvent,
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
    "KISEntity",
    "KISEntityBinding",
    "KISEvent",
    "KISIntent",
    "KISIntentResolver",
    "KISResolution",
    "KISResolutionEntity",
    "KISResolutionEvent",
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
