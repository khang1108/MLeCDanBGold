"""System prompts and message builders for KIS semantic intent resolution.

This module owns version-controlled prompt templates for transforming an ordered
sequence of KIS clues into a semantic resolution (entities and chronologically ordered events).
"""

from __future__ import annotations

from collections.abc import Sequence

KIS_RESOLVER_SYSTEM_PROMPT = """You are an expert video retrieval query intent resolver for multimodal search competitions.
Your task is to analyze an ordered history of clues revealed over time for a video segment and resolve them into a clean semantic resolution.

Guidelines:
1. ENTITY TRACKING: Identify key entities (persons, objects, places, texts) appearing across the clue history.
2. COREFERENCE & PRONOUNS: Resolve ambiguous pronouns ("he", "she", "they", "it", "that thing") to their canonical entity descriptions.
3. CONTRADICTIONS & CORRECTIONS: If a subsequent clue explicitly corrects an earlier clue (e.g., "actually orange, not red"), the canonical query_text, entities, and events MUST reflect the correction instead of concatenating contradictions.
4. TEMPORAL ORDERING: Output events in strictly chronological order even when clues reveal them out of order (e.g., if a clue says "Before taking the plate, they move to the left", the movement event MUST precede the taking plate event).
5. MERGING & SPLITTING:
   - Merge simultaneous descriptions of the same moment/scene into one event.
   - Split genuinely sequential actions into separate events.
6. EVENT TEXT: Write self-contained event text suitable for dense and lexical retrieval. Do not leave dangling pronouns.
7. ENTITY INDICES: For each event, entity_indices must be a list of zero-based integer indices referencing entities[] involved in this event.
8. LANGUAGE: Output in the dominant language of the input clues ("vi" or "en").
9. RESTRICTIONS:
   - Do not emit timestamps, candidate IDs, retrieval translations, event IDs, entity IDs, clue history copies, or edges.
"""


def build_kis_intent_messages(clues: Sequence[str]) -> list[dict[str, str]]:
    """Build OpenAI-style chat messages for KIS semantic intent resolution."""
    formatted_clues = "\n".join(f"Clue {idx + 1}: {clue}" for idx, clue in enumerate(clues))
    user_prompt = (
        f"Analyze the following ordered history of {len(clues)} clue(s) for a video segment "
        f"and resolve them into a semantic KISResolution:\n\n"
        f"{formatted_clues}"
    )

    return [
        {"role": "system", "content": KIS_RESOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


KIS_SCOPED_RESOLVER_SYSTEM_PROMPT = """Resolve only the explicitly granted KIS event IDs.
Return one resolution for each granted ID and no other IDs. Preserve canonical entity IDs;
use only IDs supplied in the base intent and retain every model-supplied non-blank binding role.
Do not invent entities, roles, images, timestamps, or event IDs. Event text must be self-contained.
Return language as vi or en and use the base language when one is already established."""


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
