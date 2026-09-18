import httpx
from fastapi.testclient import TestClient

from hcmai.common.observability.models import RetrievalTrace
from hcmai.retrieval.models import RetrievalCandidate, RetrievalResult, RetrievalSource
from hcmai.retrieval.serving.client import RetrievalHttpClient
from hcmai.retrieval.serving.server import create_app
from hcmai.retrieval.serving.utils.config import RetrievalClientSettings


class FakeRuntime:
    def __init__(self):
        self.search_text_calls = []

    def search_text(self, query: str, *, top_k: int) -> RetrievalResult:
        self.search_text_calls.append((query, top_k))
        return RetrievalResult(
            candidates=[
                RetrievalCandidate(
                    frame_id="f1",
                    source_scores={RetrievalSource.VISUAL: .8},
                    source_ranks={RetrievalSource.VISUAL: 1},
                    fusion_score=.02,
                ),
                RetrievalCandidate(
                    frame_id="f2",
                    source_scores={RetrievalSource.VISUAL: .7},
                    source_ranks={RetrievalSource.VISUAL: 2},
                    fusion_score=.01,
                ),
            ],
            trace=RetrievalTrace(),
            warnings=["context unavailable"],
        )


def test_search_text_endpoint_preserves_retrieval_order_and_rank():
    runtime = FakeRuntime()
    client = TestClient(create_app(runtime=runtime))

    response = client.post("/search_text", json={"query": "seafood", "top_k": 50})

    assert response.status_code == 200
    assert runtime.search_text_calls == [("seafood", 50)]
    assert response.json()["candidates"] == [
        {"frame_id": "f1", "rank": 1, "score": .02},
        {"frame_id": "f2", "rank": 2, "score": .01},
    ]
    assert response.json()["warnings"] == ["context unavailable"]


class StubHttpClient:
    def post(self, path: str, *, json: dict):
        assert path == "/search_text"
        assert json == {"query": "seafood", "top_k": 50}
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"frame_id": "f1", "rank": 1, "score": .02},
                    {"frame_id": "f2", "rank": 2, "score": .01},
                ],
                "retrieval_ms": 12.5,
                "warnings": ["context unavailable"],
            },
            request=httpx.Request("POST", "http://127.0.0.1:8002/search_text"),
        )

    def close(self) -> None:
        return None


def test_search_text_client_maps_rank_order_and_diagnostics():
    client = RetrievalHttpClient(RetrievalClientSettings(target="127.0.0.1:8002"))
    client._client.close()
    client._client = StubHttpClient()

    result = client.search_text("seafood", top_k=50)

    assert [(item.frame_id, item.rank, item.score) for item in result.candidates] == [
        ("f1", 1, .02),
        ("f2", 2, .01),
    ]
    assert result.retrieval_ms == 12.5
    assert result.warnings == ("context unavailable",)
