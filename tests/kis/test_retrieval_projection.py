"""Tests for retrieval projection separating canonical text and translated dense text."""

from hcmai.kis.models import KISEvent, KISIntent
from hcmai.retrieval.plan import build_retrieval_plan


def make_intent() -> KISIntent:
    return KISIntent(
        revision=1,
        query_text="người đàn ông ngồi xuống",
        language="vi",
        entities=[],
        events=[KISEvent(id="E1", text="người đàn ông ngồi xuống")],
        temporal_edges=[],
    )


def test_dense_translation_does_not_replace_canonical_or_bm25_text():
    plan = build_retrieval_plan(
        make_intent(),
        dense_text_by_event={"E1": "a man sits down"},
        use_dense=True,
        use_bm25=True,
    )
    row = plan.events[0]
    assert row.canonical_text == "người đàn ông ngồi xuống"
    assert row.dense_text == "a man sits down"
    assert row.bm25_text == "người đàn ông ngồi xuống"
