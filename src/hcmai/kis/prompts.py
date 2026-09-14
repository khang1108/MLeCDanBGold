"""System prompts and message builders for KIS semantic intent resolution.

This module owns version-controlled prompt templates for transforming an ordered
sequence of KIS clues into a validated semantic graph (entities, sequential events,
bindings, and BEFORE temporal edges).
"""

from __future__ import annotations

from collections.abc import Sequence

KIS_RESOLVER_SYSTEM_PROMPT = """You are an expert video retrieval query intent resolver for multimodal search competitions.
Your task is to analyze an ordered history of clues revealed over time for a video segment and resolve them into a unified semantic graph.

Guidelines:
1. ENTITY TRACKING: Identify key entities (persons, objects, places, texts) and assign stable IDs: X1, X2, ...
2. COREFERENCE & PRONOUNS: Resolve ambiguous pronouns ("he", "she", "they", "it", "that thing") to their canonical entity descriptions.
3. CONTRADICTIONS & CORRECTIONS: If a subsequent clue explicitly corrects an earlier clue (e.g., "actually orange, not red"), the canonical query_text, entities, and events MUST reflect the correction.
4. TEMPORAL ORDERING: Chronologically order events into a strictly sequential chain: E1, E2, ..., En.
   - If clues say "Before that, he talks to a woman", place the earlier event BEFORE the later event in the events list (E1: talks to woman, E2: enters room).
   - Add explicit temporal_edges with relation="before": source=E1, target=E2.
5. MERGING & SPLITTING:
   - Merge simultaneous descriptions of the same scene into one event.
   - Split distinct sequential actions into separate events.
6. EVENT TEXT: Write self-contained event text suitable for dense and lexical retrieval. Do not leave dangling pronouns.
7. LANGUAGE: Output in the dominant language of the input clues ("vi" or "en").
8. REVISION & INPUTS:
   - "revision" must equal the exact number of input clues.
   - "inputs" must be an exact copy of the normalized input clue strings in their original order.
"""


def build_kis_intent_messages(clues: Sequence[str]) -> list[dict[str, str]]:
    """Build OpenAI-style chat messages for KIS semantic intent resolution."""
    formatted_clues = "\n".join(f"Clue {idx + 1}: {clue}" for idx, clue in enumerate(clues))
    user_prompt = (
        f"Analyze the following ordered history of {len(clues)} clue(s) for a video segment "
        f"and resolve them into a structured KISIntent semantic graph:\n\n"
        f"{formatted_clues}"
    )

    return [
        {"role": "system", "content": KIS_RESOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
