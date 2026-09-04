"""Tests for the active backend route inventory."""

from hcmai.app import create_app


def test_literal_filter_and_competition_routes_are_registered() -> None:
    paths = {
        child.path
        for included in create_app().routes
        for child in getattr(getattr(included, "original_router", None), "routes", (included,))
        if hasattr(child, "path")
    }

    assert {
        "/api/v1/database/tables",
        "/api/v1/database/tables/{table_name}/rows",
        "/api/v1/database/execute",
        "/api/v1/search",
        "/api/v1/filter",
        "/api/v1/trake",
        "/api/v1/query-candidates",
        "/api/v1/videos",
        "/api/v1/videos/{video_id}/stream",
    } <= paths
