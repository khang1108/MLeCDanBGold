"""Structured Query Hypothesis package for KIS.

This package owns source grounding, inspectable event topology, deterministic
structural mutations (split, merge, reorder, edit, add, undo), bounded session
storage, and preview/commit lifecycle for KIS queries.
"""

from hcmai.kis.hypothesis.grounding import align_source_fragments, infer_query_language

__all__ = [
    "align_source_fragments",
    "infer_query_language",
]
