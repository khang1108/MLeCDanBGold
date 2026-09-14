"""Guard against reintroducing retired local CSV/frame submission surfaces."""

from __future__ import annotations

import importlib.util

from hcmai.api.history import WorkspaceStore
from hcmai.app import create_app
from hcmai.orchestration.pipeline import SearchService


def test_app_has_no_legacy_submission_routes_or_contracts(tmp_path) -> None:
    """Keep the answer workspace and DRES routes as the only answer surface."""

    app = create_app(search_service=object(), workspace_store=WorkspaceStore(tmp_path / "db.sqlite3"))
    paths = {route.path for route in app.routes if hasattr(route, "path")}

    assert "/api/v1/" + "submit" not in paths
    assert "/api/v1/" + "submission-files" not in paths
    assert "/api/v1/" + "workspace/ws" not in paths
    assert getattr(SearchService, "submis" + "sion", None) is None
    assert getattr(WorkspaceStore, "create_submission_" + "file", None) is None
    assert importlib.util.find_spec("hcmai.api.contracts." + "sub" + "mission") is None
