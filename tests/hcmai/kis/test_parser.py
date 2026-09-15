"""Tests for deterministic KIS composer command parsing."""

from __future__ import annotations

import pytest

from hcmai.kis.parser import KISCommandError, parse_kis_command


def test_parser_extracts_scoped_batch_in_explicit_header_order() -> None:
    """Existing intents accept ordered updates and one appended adjacent event."""
    command = parse_kis_command(
        "E2: chef wears black\nE4: then she takes a plate",
        base_event_count=3,
    )

    assert command.kind == "patch_events"
    assert [patch.event_id for patch in command.patches] == ["E2", "E4"]


def test_parser_collects_continuation_lines_until_next_header() -> None:
    """Continuation text belongs to its preceding explicit event instruction."""
    command = parse_kis_command(
        "E1: woman enters\nthrough the side door\nE2: she sits",
        base_event_count=2,
    )

    assert command.patches[0].instruction == "woman enters\nthrough the side door"


def test_parser_rejects_gap_for_new_event() -> None:
    """An appended event must be the next event after the base timeline."""
    with pytest.raises(KISCommandError, match="missing E4"):
        parse_kis_command("E5: new event", base_event_count=3)


def test_parser_rejects_duplicate_event_ids() -> None:
    """One command cannot provide ambiguous repeated instructions for an event."""
    with pytest.raises(KISCommandError, match="duplicate"):
        parse_kis_command("E1: first\nE1: second", base_event_count=1)


def test_parser_rejects_out_of_order_headers() -> None:
    """Parser preserves user order and rejects headers that would require sorting."""
    with pytest.raises(KISCommandError, match="increasing"):
        parse_kis_command("E2: second\nE1: first", base_event_count=2)


def test_parser_rejects_malformed_event_prefix() -> None:
    """Malformed event-like prefixes cannot silently become natural language."""
    with pytest.raises(KISCommandError, match="Malformed"):
        parse_kis_command("E: missing number", base_event_count=0)


def test_parser_rejects_unscoped_progressive_text() -> None:
    """A follow-up must explicitly name its event once an intent exists."""
    with pytest.raises(KISCommandError, match="explicit E#"):
        parse_kis_command("chef wears black", base_event_count=1)


def test_parser_routes_initial_explicit_batch() -> None:
    """A contiguous initial explicit batch selects the explicit-initial operation."""
    command = parse_kis_command("E1: woman enters\nE2: woman sits", base_event_count=0)

    assert command.kind == "initial_explicit"
    assert [patch.event_id for patch in command.patches] == ["E1", "E2"]


def test_parser_routes_initial_natural_text() -> None:
    """Unscoped text creates a natural initial resolution only for a new intent."""
    command = parse_kis_command("A woman enters a kitchen.", base_event_count=0)

    assert command.kind == "initial_natural"
    assert command.text == "A woman enters a kitchen."


def test_parser_routes_global_rewrite() -> None:
    """The dedicated command is the only route to global semantic rewriting."""
    command = parse_kis_command(
        "/llm-rewrite\nResolve all pronouns explicitly.",
        base_event_count=3,
    )

    assert command.kind == "global_rewrite"
    assert command.instruction == "Resolve all pronouns explicitly."
