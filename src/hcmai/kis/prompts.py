"""System prompts and message builders for KIS semantic intent resolution.

This module owns version-controlled prompt templates for transforming an ordered
sequence of KIS clues into a semantic resolution (entities and chronologically ordered events).
"""

from __future__ import annotations

from collections.abc import Sequence

KIS_RESOLVER_SYSTEM_PROMPT = """You are an expert video retrieval query intent resolver for multimodal search competitions.
Your task is to analyze either a single narrative containing multiple chronological moments or an ordered history of clues revealed over time for a video segment, then resolve the input into a clean semantic resolution.
A clue is not an event: one clue may resolve to one or many chronological events.

Required JSON Structure:
You MUST output a JSON object containing ALL of the following fields:
- "query_text": concise string summarizing the full video query (required)
- "entities": list of tracked entities [{"kind": "person"|"object"|"place"|"text"|"other", "description": "..."}, ...], or [] if none
- "events": list of chronological events [{"text": "...", "entity_indices": [...]}, ...] (at least 1 event required)

Guidelines:
1. ENTITY TRACKING: Identify key entities (persons, objects, places, texts) appearing across the clue history or narrative.
2. COREFERENCE & PRONOUNS: Resolve ambiguous pronouns ("he", "she", "they", "it", "that thing") to their canonical entity descriptions.
3. CONTRADICTIONS & CORRECTIONS: If a subsequent clue explicitly corrects an earlier clue (e.g., "actually orange, not red"), the canonical query_text, entities, and events MUST reflect the correction instead of concatenating contradictions.
4. TEMPORAL ORDERING: Output events in strictly chronological order even when clues reveal them out of order.
5. TEMPORAL DECOMPOSITION:
   - Identify distinct moments or actions that occur sequentially in the described video.
   - Each distinct sequential moment MUST be a separate object in the "events" array.
   - Do NOT combine several sequential moments into a single event text.
   - Merge actions only when they occur as part of the same temporal moment or simultaneous scene.
   - Temporal transition phrases such as "initially", "then", "afterward", "later", "bắt đầu khi", "sau đó", "ngay sau đó", and "tiếp theo" are examples of evidence for temporal change, not an exhaustive keyword list.
6. EVENT TEXT: Write a concise, self-contained English description of exactly one temporal moment suitable for dense and lexical visual retrieval. Resolve pronouns to explicit entities where useful. Never write an Event 1/Event 2/Event 3 list or a multi-stage narrative inside a single event text.
7. ENTITY INDICES: For each event, entity_indices must be a list of 0-based indices into the entities array. If entities is empty [], entity_indices MUST be empty [] for all events. The same entity index may appear in multiple events when that entity persists across time.
8. RESTRICTIONS: Do not emit timestamps, candidate IDs, retrieval translations, event IDs, entity IDs, clue history copies, or edges.

Example — one clue can contain multiple events:
Input:
A man first enters a room carrying a box. Then he puts the box on a table. Finally a woman opens the box.

Output:
{
  "query_text": "A man enters a room carrying a box, places it on a table, and a woman later opens the box.",
  "entities": [
    {"kind": "person", "description": "a man carrying and placing a box"},
    {"kind": "object", "description": "a box"},
    {"kind": "person", "description": "a woman who opens the box"}
  ],
  "events": [
    {"text": "A man enters a room carrying a box.", "entity_indices": [0, 1]},
    {"text": "The man places the box on a table.", "entity_indices": [0, 1]},
    {"text": "A woman opens the box on the table.", "entity_indices": [2, 1]}
  ]
}
"""


def build_kis_intent_messages(clues: Sequence[str]) -> list[dict[str, str]]:
    """Build OpenAI-style chat messages for KIS semantic intent resolution."""
    formatted_clues = "\n".join(f"Clue {idx + 1}: {clue}" for idx, clue in enumerate(clues))
    user_prompt = (
        "The input may be either a single narrative containing multiple chronological "
        "moments or an ordered history of progressively revealed clues. One clue may "
        "produce multiple events when it describes sequential moments.\n\n"
        f"Analyze the following {len(clues)} clue(s) for a video segment and resolve "
        f"them into a semantic KISResolution:\n\n{formatted_clues}"
    )

    return [
        {"role": "system", "content": KIS_RESOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
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
