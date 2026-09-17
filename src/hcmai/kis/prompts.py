"""System prompts and message builders for KIS semantic intent resolution.

This module owns version-controlled prompt templates for transforming an ordered
sequence of KIS clues into a semantic resolution (entities and chronologically ordered events).
"""

from __future__ import annotations

from collections.abc import Sequence

KIS_INITIAL_RESOLVER_SYSTEM_PROMPT = """You decompose a video-search description into chronologically ordered, visually retrievable moments.

Rules:
- One event is one distinct retrievable visual moment.
- Sequential views or actions are separate events, even when one continuous camera shot or pan connects them.
- Details that are genuinely simultaneous in the same visual moment stay in one event.
- Write each event as one concise, self-contained English sentence suitable for visual/text retrieval.
- Preserve uncertainty; do not invent a specific tool, material, person, place, text, or action that the input does not establish.
- Do not explain reasoning, summarize the whole query, create entity tables, or put multiple stages inside one event text.

Example:
Input: The camera starts on a framed certificate, then pans right to a craftsperson engraving a metal plate with an unusual tool.
Events:
1. A framed certificate is visible on a work table.
2. A craftsperson engraves a metal plate with an unusual tool.
"""


def build_kis_initial_messages(clues: Sequence[str]) -> list[dict[str, str]]:
    """Build messages for bounded initial event decomposition."""
    formatted = "\n".join(
        f"Clue {index + 1}: {clue}" for index, clue in enumerate(clues)
    )
    return [
        {"role": "system", "content": KIS_INITIAL_RESOLVER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Return the chronological retrievable moments for this input. "
                "A single clue may contain multiple events.\n\n"
                f"{formatted}"
            ),
        },
    ]




KIS_SCOPED_RESOLVER_SYSTEM_PROMPT = """Resolve only the explicitly granted KIS event IDs.
Return one resolution for each granted ID and no other IDs. Preserve canonical entity IDs;
use only IDs supplied in the base intent and retain every model-supplied non-blank binding role.
Do not invent entities, roles, images, timestamps, or event IDs. Event text must be self-contained."""


def build_kis_scoped_messages(
    base_description: str, instructions: Sequence[tuple[str, str]]
) -> list[dict[str, str]]:
    """Build prompts for a model resolution restricted to named event IDs."""
    formatted = "\n".join(f"{event_id}: {instruction}" for event_id, instruction in instructions)
    return [
        {"role": "system", "content": KIS_SCOPED_RESOLVER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Base intent:\n"
                f"{base_description}\n\n"
                "Resolve exactly these event instructions:\n"
                f"{formatted}"
            ),
        },
    ]


KIS_GLOBAL_REWRITER_SYSTEM_PROMPT = """Rewrite the KIS intent globally while preserving its event topology.
Return exactly one event resolution for every existing event ID, in the same order.
Never add, remove, reorder, or rename events. An image-only event must remain text=null:
the model has no image pixels and must not invent visual semantics. Preserve supplied entity
IDs and use bindings with non-blank roles. Do not emit images, timestamps, or revision."""


def build_kis_global_rewrite_messages(
    base_description: str, instruction: str
) -> list[dict[str, str]]:
    """Build prompts for an explicit topology-preserving global rewrite."""
    return [
        {"role": "system", "content": KIS_GLOBAL_REWRITER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Base intent:\n{base_description}\n\nInstruction:\n{instruction}",
        },
    ]
