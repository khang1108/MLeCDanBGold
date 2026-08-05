"""Smoke test for TRAKE query splitting with a fake structured provider."""

from __future__ import annotations

import pytest

from hcmai.agents.trake import TrakeParserError, TrakeQueryParser

_PACK = "1. xe máy đỏ dừng lại | 2. người lái tháo mũ bảo hiểm"
_PROSE = "Xe máy đỏ dừng lại, sau đó người lái tháo mũ bảo hiểm."
_EVENTS = ["a red motorbike stops", "the rider takes off the helmet"]


def test_model_events_are_used() -> None:
    parser = TrakeQueryParser(lambda _: {"events": _EVENTS})
    assert parser.parse(_PACK).events == _EVENTS


def test_wrong_model_count_falls_back_to_the_pack_split() -> None:
    parser = TrakeQueryParser(lambda _: {"events": ["only one"]})
    parsed = parser.parse(_PACK)
    assert parsed.events == ["xe máy đỏ dừng lại", "người lái tháo mũ bảo hiểm"]


def test_prose_query_with_wrong_count_and_no_fallback_raises() -> None:
    parser = TrakeQueryParser(lambda _: {"events": ["only one"]})
    with pytest.raises(TrakeParserError, match="expected 2"):
        parser.parse(_PROSE, 2)
