"""Semantic intent resolution, scoped event updating, and rewriting using LLMs."""

from __future__ import annotations

from hcmai.kis.resolution.initial import (
    DEFAULT_INITIAL_RESOLVER_MAX_TOKENS,
    KISIntentResolver,
    KISResolutionError,
)
from hcmai.kis.resolution.prompts import (
    KIS_INITIAL_RESOLVER_SYSTEM_PROMPT,
    KIS_SCOPED_RESOLVER_SYSTEM_PROMPT,
    build_kis_global_rewrite_messages,
    build_kis_initial_messages,
    build_kis_scoped_messages,
)
from hcmai.kis.resolution.rewrite import (
    GlobalRewriteEvent,
    GlobalRewriteResolution,
    KISGlobalRewriter,
)
from hcmai.kis.resolution.scoped import (
    KISScopedResolver,
    ScopedResolutionBatch,
    ScopedResolvedEvent,
    apply_scoped_resolutions,
    canonical_query_text,
)

__all__ = [
    "DEFAULT_INITIAL_RESOLVER_MAX_TOKENS",
    "GlobalRewriteEvent",
    "GlobalRewriteResolution",
    "KISGlobalRewriter",
    "KISIntentResolver",
    "KISResolutionError",
    "KISScopedResolver",
    "KIS_INITIAL_RESOLVER_SYSTEM_PROMPT",
    "KIS_SCOPED_RESOLVER_SYSTEM_PROMPT",
    "ScopedResolutionBatch",
    "ScopedResolvedEvent",
    "apply_scoped_resolutions",
    "build_kis_global_rewrite_messages",
    "build_kis_initial_messages",
    "build_kis_scoped_messages",
    "canonical_query_text",
]
