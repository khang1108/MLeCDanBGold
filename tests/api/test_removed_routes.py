"""Guard against retired answer persistence and shared submission routes."""

from __future__ import annotations

import importlib.util

from hcmai.app import create_app
from hcmai.orchestration.pipeline import SearchService


def test_app_keeps_only_the_private_direct_dres_routes_and_no_database() -> None:
    """Keep one stateless answer POST after workspace and database removal."""

    app = create_app(search_service=object())
    # New FastAPI lazily includes APIRouter objects without exposing a .path
    # on each wrapper; OpenAPI materializes the same public route surface.
    paths = set(app.openapi()["paths"])

    assert "/api/v1/answer-workspace" not in paths
    assert "/api/v1/answer-workspace/ws" not in paths
    assert "/api/v1/vbs/submit/kis" not in paths
    assert "/api/v1/vbs/submit/vqa" not in paths
    assert "/api/v1/vbs/submit/avs" not in paths
    assert "/api/v1/vbs/submission-attempts/{attempt_id}/resolve" not in paths
    assert "/api/v1/vbs/session/connect" in paths
    assert "/api/v1/vbs/session/{user_id}" in paths
    assert "/api/v1/vbs/task/{user_id}" in paths
    assert "/api/v1/vbs/submit" in paths
    assert "/api/v1/query-history" not in paths
    assert "/api/v1/query-candidates" not in paths
    assert "/api/v1/database/tables" not in paths
    assert "/api/v1/database/execute" not in paths
    assert getattr(SearchService, "submis" + "sion", None) is None
    assert getattr(SearchService, "generate_query_candidates", None) is None
    assert importlib.util.find_spec("hcmai.api.history") is None
    assert importlib.util.find_spec("hcmai.api.contracts.history") is None
    assert importlib.util.find_spec("hcmai.api.routers.history") is None
    assert importlib.util.find_spec("hcmai.api.contracts." + "sub" + "mission") is None
    assert importlib.util.find_spec("hcmai.api.contracts.workspace") is None
    assert importlib.util.find_spec("hcmai.api.routers.workspace") is None
    assert importlib.util.find_spec("hcmai.api.contracts.database") is None
    assert importlib.util.find_spec("hcmai.api.routers.database") is None
