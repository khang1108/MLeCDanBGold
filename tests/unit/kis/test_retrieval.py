from hcmai.common.schemas import RetrievalCandidate, RetrievalResult, TaskType
from hcmai.kis.retrieval import retrieve_query_variants
from hcmai.kis.variants import QueryVariant


class Retrieval:
    def search_batch(self, queries, top_k, filters, query_type):
        assert queries == ["original", "variant"]
        assert (top_k, filters, query_type) == (10, None, TaskType.KIS)
        return [
            RetrievalResult(candidates=[
                RetrievalCandidate(frame_id="original-top"),
                RetrievalCandidate(frame_id="shared"),
            ]),
            RetrievalResult(candidates=[
                RetrievalCandidate(frame_id="variant-top"),
                RetrievalCandidate(frame_id="shared"),
            ]),
        ]


def test_variant_fusion_preserves_original_top_and_is_deterministic():
    variants = (
        QueryVariant("original", "original", 1.0),
        QueryVariant("variant", "generated:literal", 0.35),
    )

    first = retrieve_query_variants(
        Retrieval(), variants, top_k=10, filters=None, query_type=TaskType.KIS
    )
    second = retrieve_query_variants(
        Retrieval(), variants, top_k=10, filters=None, query_type=TaskType.KIS
    )

    assert [item.frame_id for item in first] == [
        "original-top", "shared", "variant-top"
    ]
    assert [item.frame_id for item in first] == [item.frame_id for item in second]
    assert len(first[1].metadata["query_variant_provenance"]) == 2
