"""Tests for QueryHypothesisService lifecycle, preview, commit, undo, and revision guards."""

import pytest

from hcmai.kis.hypothesis.models import EditEvent, QueryHypothesisError
from hcmai.kis.hypothesis.service import QueryHypothesisService
from hcmai.kis.hypothesis.store import QueryHypothesisStore
from hcmai.kis.models import (
    KISEvent,
    KISIntent,
    SourceProvenance,
)


class FakeResolver:
    def resolve_initial(self, text: str, revision: int = 1) -> KISIntent:
        return KISIntent(
            revision=revision,
            query_text=text,
            language="vi",
            entities=[],
            events=[
                KISEvent(
                    id="E1",
                    text=text,
                    origin="source",
                    source_provenance=SourceProvenance(
                        source_text=text, start_char=0, end_char=len(text)
                    ),
                    bindings=[],
                )
            ],
            temporal_edges=[],
        )


@pytest.fixture
def service() -> QueryHypothesisService:
    store = QueryHypothesisStore(ttl_seconds=1800, max_entries=50)
    return QueryHypothesisService(resolver=FakeResolver(), store=store)


@pytest.fixture
def opened_session(service: QueryHypothesisService):
    return service.open("người đàn ông ngồi xuống")


def test_preview_does_not_bump_revision(
    service: QueryHypothesisService, opened_session
) -> None:
    before = service.get(opened_session.session_id)
    preview = service.preview(
        opened_session.session_id,
        expected_revision=before.intent.revision,
        action=EditEvent(event_id="E1", text="preview text"),
    )
    after = service.get(opened_session.session_id)
    assert preview.intent.revision == before.intent.revision + 1
    assert after.intent == before.intent


def test_commit_rejects_stale_revision(
    service: QueryHypothesisService, opened_session
) -> None:
    service.commit(
        opened_session.session_id, 1, EditEvent(event_id="E1", text="first")
    )
    try:
        service.commit(
            opened_session.session_id, 1, EditEvent(event_id="E1", text="stale")
        )
    except QueryHypothesisError as exc:
        assert exc.code == "QUERY_REVISION_CONFLICT"
    else:
        raise AssertionError("stale query commit must fail")


def test_undo_creates_new_monotonic_revision(
    service: QueryHypothesisService, opened_session
) -> None:
    committed = service.commit(
        opened_session.session_id, 1, EditEvent(event_id="E1", text="changed")
    )
    restored = service.undo(opened_session.session_id, committed.intent.revision)
    assert restored.intent.revision == committed.intent.revision + 1
    assert restored.intent.events[0].text == "người đàn ông ngồi xuống"
