"""Parse deterministic KIS composer commands without model inference.

This module recognizes initial natural-language input, ordered event-scoped
instructions, and the explicit global rewrite command. It only validates command
grammar and event numbering; semantic interpretation belongs to KIS resolvers.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, TypeAlias


_EVENT_HEADER = re.compile(r"(?m)^E([1-9]\d*):[ \t]*(.*)$")
_MALFORMED_EVENT_PREFIX = re.compile(
    r"(?m)^E(?=[0-9+\-: \t])[^\r\n:]*:"
)


class KISCommandError(ValueError):
    """Raised when composer text does not satisfy the deterministic grammar."""


@dataclass(frozen=True, slots=True)
class EventPatchInstruction:
    """One user-authored semantic instruction scoped to a canonical event ID."""

    event_id: str
    instruction: str


@dataclass(frozen=True, slots=True)
class ParsedInitialNatural:
    """An initial unscoped natural-language request."""

    kind: Literal["initial_natural"] = "initial_natural"
    text: str = ""


@dataclass(frozen=True, slots=True)
class ParsedInitialExplicit:
    """An initial contiguous batch whose event IDs are explicitly supplied."""

    kind: Literal["initial_explicit"] = "initial_explicit"
    patches: tuple[EventPatchInstruction, ...] = ()


@dataclass(frozen=True, slots=True)
class ParsedPatchEvents:
    """An ordered batch of semantic updates to named events."""

    kind: Literal["patch_events"] = "patch_events"
    patches: tuple[EventPatchInstruction, ...] = ()


@dataclass(frozen=True, slots=True)
class ParsedGlobalRewrite:
    """An explicit instruction authorizing global semantic rewriting."""

    kind: Literal["global_rewrite"] = "global_rewrite"
    instruction: str = ""


ParsedKISCommand: TypeAlias = (
    ParsedInitialNatural
    | ParsedInitialExplicit
    | ParsedPatchEvents
    | ParsedGlobalRewrite
)


def parse_kis_command(text: str, *, base_event_count: int) -> ParsedKISCommand:
    """Classify and validate one KIS composer submission deterministically.

    Existing intents require an explicit ``E#:`` target for local changes. New
    event IDs must extend the existing timeline contiguously, while all supplied
    headers must already be in strictly increasing user order.
    """
    if base_event_count < 0:
        raise ValueError("base_event_count must be non-negative")

    normalized = text.strip()
    if not normalized:
        raise KISCommandError("KIS command must contain non-empty text")

    if normalized.startswith("/llm-rewrite"):
        return _parse_global_rewrite(normalized)

    headers = tuple(_EVENT_HEADER.finditer(normalized))
    if not headers:
        if _MALFORMED_EVENT_PREFIX.search(normalized):
            raise KISCommandError("Malformed E#: event prefix")
        if base_event_count:
            raise KISCommandError(
                "Existing KIS intents require explicit E#: event instructions"
            )
        return ParsedInitialNatural(text=normalized)

    if headers[0].start() != 0:
        raise KISCommandError("Explicit E#: event instructions must begin the command")
    _reject_malformed_event_prefixes(normalized, headers)
    patches = _parse_event_patches(normalized, headers)
    _validate_event_ids(patches, base_event_count=base_event_count)

    if base_event_count == 0:
        return ParsedInitialExplicit(patches=patches)
    return ParsedPatchEvents(patches=patches)


def _parse_global_rewrite(text: str) -> ParsedGlobalRewrite:
    """Validate the dedicated two-line global rewrite command."""
    lines = text.splitlines()
    if lines[0] != "/llm-rewrite":
        raise KISCommandError("Malformed /llm-rewrite command")
    instruction = "\n".join(lines[1:]).strip()
    if not instruction:
        raise KISCommandError("/llm-rewrite requires a non-empty instruction")
    return ParsedGlobalRewrite(instruction=instruction)


def _reject_malformed_event_prefixes(
    text: str, headers: tuple[re.Match[str], ...]
) -> None:
    """Reject event-like lines that the strict header expression did not accept."""
    valid_starts = {header.start() for header in headers}
    for malformed in _MALFORMED_EVENT_PREFIX.finditer(text):
        if malformed.start() not in valid_starts:
            raise KISCommandError("Malformed E#: event prefix")


def _parse_event_patches(
    text: str, headers: tuple[re.Match[str], ...]
) -> tuple[EventPatchInstruction, ...]:
    """Collect each header's content through the start of its next header."""
    patches: list[EventPatchInstruction] = []
    for index, header in enumerate(headers):
        next_start = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        instruction = (header.group(2) + text[header.end() : next_start]).strip()
        if not instruction:
            raise KISCommandError(f"{header.group(0).split(':', 1)[0]} requires an instruction")
        patches.append(
            EventPatchInstruction(event_id=f"E{header.group(1)}", instruction=instruction)
        )
    return tuple(patches)


def _validate_event_ids(
    patches: tuple[EventPatchInstruction, ...], *, base_event_count: int
) -> None:
    """Enforce duplicate-free, ordered headers and contiguous newly added IDs."""
    numbers = [int(patch.event_id[1:]) for patch in patches]
    if len(numbers) != len(set(numbers)):
        raise KISCommandError("duplicate event IDs are not allowed")
    if numbers != sorted(numbers):
        raise KISCommandError("event headers must be in strictly increasing order")

    new_numbers = [number for number in numbers if number > base_event_count]
    if new_numbers:
        expected = list(range(base_event_count + 1, new_numbers[-1] + 1))
        if new_numbers != expected:
            missing = next(number for number in expected if number not in new_numbers)
            raise KISCommandError(f"missing E{missing} before a new event")


__all__ = [
    "EventPatchInstruction",
    "KISCommandError",
    "ParsedGlobalRewrite",
    "ParsedInitialExplicit",
    "ParsedInitialNatural",
    "ParsedKISCommand",
    "ParsedPatchEvents",
    "parse_kis_command",
]
