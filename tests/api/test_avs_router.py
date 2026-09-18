from fastapi import FastAPI
from fastapi.testclient import TestClient

from hcmai.api.contracts.avs import (
    AvsSearchLatency,
    AvsSearchResponse,
    AvsSearchResult,
)
from hcmai.api.contracts.search import SearchResultMetadata
from hcmai.api.routers.avs import create_avs_router


class FakeSearchService:
    def search_avs(self, request):
        assert request.query == "seafood"
        assert request.page_size == 80
        return AvsSearchResponse(
            results=[AvsSearchResult(
                candidate_id="f1",
                frame_id="f1",
                video_id="V1",
                frame_idx=10,
                timestamp_ms=1000,
                fps=25.0,
                retrieval_rank=1,
                retrieval_score=.9,
                metadata=SearchResultMetadata(),
            )],
            latency=AvsSearchLatency(
                retrieval_ms=1.0,
                coverage_ms=.1,
                materialization_ms=.1,
                total_ms=1.2,
            ),
            candidate_pool_size=5,
            deduplicated_candidate_count=4,
            unique_videos=1,
            warnings=[],
        )


def test_avs_router_returns_dedicated_contract():
    app = FastAPI()
    app.include_router(create_avs_router({"service": FakeSearchService()}))

    response = TestClient(app).post(
        "/api/v1/avs/search",
        json={"query": "seafood", "page_size": 80},
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["candidate_id"] == "f1"
    assert response.json()["candidate_pool_size"] == 5
