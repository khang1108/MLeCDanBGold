"""Tests for source grounding and language classification in KIS query hypotheses."""

from hcmai.kis.hypothesis.grounding import align_source_fragments, infer_query_language


def test_align_source_fragments_is_left_to_right_for_repeated_text():
    query = "cốc rồi cốc rồi bàn"
    spans = align_source_fragments(query, ["cốc", "cốc", "bàn"])
    assert [(s.start_char, s.end_char) for s in spans] == [(0, 3), (8, 11), (16, 19)]


def test_align_source_fragments_rejects_paraphrase():
    try:
        align_source_fragments("người đàn ông vào phòng", ["a man enters a room"])
    except ValueError as exc:
        assert "not grounded" in str(exc)
    else:
        raise AssertionError("paraphrase must not be accepted as source grounding")


def test_infer_query_language_is_server_owned_and_bounded():
    assert infer_query_language("A person walks into a room") == "en"
    assert infer_query_language("Người đàn ông bước vào phòng") == "vi"
    assert infer_query_language("Người đàn ông picks up a cup") in {"vi", "mixed"}
