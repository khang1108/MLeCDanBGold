from hcmai.api.contracts.avs import AvsSearchRequest
from hcmai.common.config import AvsConfig
from hcmai.corpus.models import Frame
from hcmai.orchestration.workflows.avs_search import AvsSearchService
from hcmai.retrieval.serving.client import RemoteTextCandidate, RemoteTextSearchResult


class FakeCorpus:
    def __init__(self) -> None:
        self.frames = {
            "f1": Frame("f1", "V1", 10, 10_000, "/f1.jpg", fps=25.0),
            "f2": Frame("f2", "V1", 11, 11_000, "/f2.jpg", fps=25.0),
            "f3": Frame("f3", "V1", 30, 30_000, "/f3.jpg", fps=25.0),
            "f4": Frame("f4", "V2", 40, 40_000, "/f4.jpg", fps=25.0),
            "f5": Frame("f5", "V3", 50, 50_000, "/f5.jpg", fps=25.0),
        }

    def frame(self, frame_id: str) -> Frame:
        return self.frames[frame_id]

    def title(self, video_id: str):
        return f"title-{video_id}"

    def caption(self, frame_id: str):
        return None

    def ocr(self, frame_id: str):
        return None

    def objects(self, frame_id: str):
        return ()

    def transcript(self, video_id: str, start_ms: int, end_ms: int):
        return None


class FakeTextGateway:
    def __init__(self) -> None:
        self.calls = []

    def search_text(self, query: str, *, top_k: int) -> RemoteTextSearchResult:
        self.calls.append((query, top_k))
        return RemoteTextSearchResult(
            candidates=(
                RemoteTextCandidate("f1", 1, .99),
                RemoteTextCandidate("f2", 2, .98),
                RemoteTextCandidate("f3", 3, .97),
                RemoteTextCandidate("f4", 4, .96),
                RemoteTextCandidate("f5", 5, .95),
            ),
            retrieval_ms=8.0,
            warnings=(),
        )


def test_avs_search_uses_direct_retrieval_then_coverage():
    gateway = FakeTextGateway()
    service = AvsSearchService(
        corpus=FakeCorpus(),
        retrieval=gateway,
        config=AvsConfig(
            candidate_pool_size=500,
            default_page_size=3,
            maximum_page_size=100,
            temporal_dedup_window_ms=3000,
        ),
    )

    response = service.search(AvsSearchRequest(query="seafood", page_size=3))

    assert gateway.calls == [("seafood", 500)]
    assert [item.video_id for item in response.results] == ["V1", "V2", "V3"]
    assert [item.candidate_id for item in response.results] == ["f1", "f4", "f5"]
    assert response.candidate_pool_size == 5
    assert response.deduplicated_candidate_count == 4
    assert response.unique_videos == 3


def test_search_service_avs_fast_path_sentinels():
    from unittest.mock import Mock
    from hcmai.orchestration.pipeline import SearchService
    from hcmai.common.config import SearchConfig

    class StrictTextOnlyGateway:
        def search_text(self, query: str, *, top_k: int) -> RemoteTextSearchResult:
            return RemoteTextSearchResult(
                candidates=(RemoteTextCandidate("f1", 1, 0.99),),
                retrieval_ms=1.0,
                warnings=(),
            )

    corpus = FakeCorpus()
    gateway = StrictTextOnlyGateway()
    config = SearchConfig()
    service = SearchService(
        corpus=corpus,
        config=config,
        remote_retrieval=gateway,
    )
    # Exploding sentinels: any call to KIS or EventTrail or rewrites raises AssertionError
    service.intent_resolver = Mock(side_effect=AssertionError("KIS intent resolver called"))
    service.scoped_resolver = Mock(side_effect=AssertionError("KIS scoped resolver called"))
    service.global_rewriter = Mock(side_effect=AssertionError("KIS global rewriter called"))
    service.kis = Mock(execute=Mock(side_effect=AssertionError("KIS temporal pipeline called")))
    service.event_trail = Mock(side_effect=AssertionError("EventTrail called"))

    response = service.search_avs(AvsSearchRequest(query="test query", page_size=10))
    assert response is not None
    assert len(response.results) == 1
    assert response.results[0].candidate_id == "f1"


def test_avs_spec_fixture_video_first_ordering():
    corpus = FakeCorpus()
    # Add non-neighbor V1 candidates (timestamps 10s, 20s, 30s)
    corpus.frames["f2"] = Frame("f2", "V1", 20, 20_000, "/f2.jpg", fps=25.0)
    corpus.frames["f3"] = Frame("f3", "V1", 30, 30_000, "/f3.jpg", fps=25.0)

    class SpecFixtureGateway:
        def search_text(self, query: str, *, top_k: int) -> RemoteTextSearchResult:
            return RemoteTextSearchResult(
                candidates=(
                    RemoteTextCandidate("f1", 1, 0.99),
                    RemoteTextCandidate("f2", 2, 0.98),
                    RemoteTextCandidate("f3", 3, 0.97),
                    RemoteTextCandidate("f4", 4, 0.96),
                    RemoteTextCandidate("f5", 5, 0.95),
                ),
                retrieval_ms=2.0,
                warnings=(),
            )

    service = AvsSearchService(
        corpus=corpus,
        retrieval=SpecFixtureGateway(),
        config=AvsConfig(
            candidate_pool_size=500,
            default_page_size=5,
            maximum_page_size=100,
            temporal_dedup_window_ms=0,
        ),
    )

    response = service.search(AvsSearchRequest(query="spec fixture", page_size=5))
    assert [item.video_id for item in response.results] == ["V1", "V2", "V3", "V1", "V1"]
