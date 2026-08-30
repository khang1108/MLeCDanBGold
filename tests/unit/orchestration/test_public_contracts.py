"""Characterize public KIS and TRAKE request construction before migration."""

from hcmai.common.schemas import SearchRequest, TRAKERequest


def test_current_public_requests_are_constructible() -> None:
    """Keep KIS optional events and TRAKE ordered events backward compatible."""

    kis = SearchRequest(query="chef coats food with flour")
    trake = TRAKERequest(
        query="ordered cooking events",
        events=["chef holds skewer", "chef coats skewer"],
    )

    assert kis.query == "chef coats food with flour"
    assert trake.events == ["chef holds skewer", "chef coats skewer"]


def test_kis_events_are_optional_and_whitespace_normalized() -> None:
    """Extend KIS without requiring existing clients to send an event list."""

    request = SearchRequest(
        query="chef coats food with flour",
        events=[" chef holds skewer ", "chef coats skewer"],
    )

    assert request.events == ["chef holds skewer", "chef coats skewer"]
