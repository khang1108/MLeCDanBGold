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
