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
