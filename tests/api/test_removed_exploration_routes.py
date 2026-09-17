"""Tests verifying legacy exploration routes return 404 and EventTrail routes are active."""

from starlette.testclient import TestClient

from hcmai.app import create_app


def test_exploration_route_returns_404() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.post("/api/v1/exploration", json={})
    assert response.status_code == 404

    # Confirm EventTrail route remains registered (empty body returns 422 unprocessable, not 404)
    et_response = client.post("/api/v1/event-trail/open", json={})
    assert et_response.status_code == 422
