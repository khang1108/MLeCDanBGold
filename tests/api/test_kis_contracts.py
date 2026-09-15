import unittest

from pydantic import ValidationError

from hcmai.api.contracts.kis import KISRevisionSearchRequest


class KISRevisionSearchRequestTest(unittest.TestCase):
    def test_first_input_expects_revision_zero(self) -> None:
        request = KISRevisionSearchRequest(
            inputs=[{"text": "A woman is standing in a kitchen."}],
            expected_revision=0,
        )
        self.assertEqual(request.expected_revision, 0)
        self.assertEqual(len(request.inputs), 1)

    def test_second_input_expects_previous_revision_one(self) -> None:
        request = KISRevisionSearchRequest(
            inputs=[
                {"text": "A woman is standing in a kitchen."},
                {"text": "She is talking to a man."},
            ],
            expected_revision=1,
        )
        self.assertEqual(len(request.inputs), 2)

    def test_exposes_previous_revision_for_router_conflict_check(self) -> None:
        request = KISRevisionSearchRequest(
            inputs=[
                {"text": "A woman is standing in a kitchen."},
                {"text": "She is talking to a man."},
            ],
            expected_revision=0,
        )
        self.assertEqual(request.previous_revision, 1)
        self.assertTrue(request.has_revision_conflict)

    def test_requires_at_least_one_retrieval_source(self) -> None:
        with self.assertRaisesRegex(ValidationError, "at least one"):
            KISRevisionSearchRequest(
                inputs=[{"text": "A woman is standing in a kitchen."}],
                expected_revision=0,
                use_dense=False,
                use_bm25=False,
            )


if __name__ == "__main__":
    unittest.main()


def test_task2_step7_seed_response_removes_prepared_string_aliases():
    from hcmai.api.contracts.kis import KISExplorationSeed, KISRevisionSearchResponse
    from hcmai.api.contracts.latency import SearchLatency

    seed = KISExplorationSeed(semantic_revision=1, events=[dict(event_id="E1", canonical_text="boat", dense_text="boat")], use_dense=True, use_bm25=False)
    assert seed.model_dump()["events"][0]["dense_text"] == "boat"
    assert "dense_events" not in KISRevisionSearchResponse.model_fields
    assert "bm25_events" not in KISRevisionSearchResponse.model_fields
    assert "exploration_seed" in KISRevisionSearchResponse.model_fields
    assert SearchLatency(intent_ms=1, translation_ms=2).translation_ms == 2
