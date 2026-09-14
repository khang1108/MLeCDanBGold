"""Server-side tests for the static, same-origin monitor experience."""

from __future__ import annotations

import re

from fastapi.testclient import TestClient


def test_monitor_page_and_assets_are_delivered_from_the_package(
    client: TestClient,
) -> None:
    """The dashboard HTML, CSS, and JavaScript are available at fixed routes."""

    page = client.get("/monitor")
    css = client.get("/static/monitor.css")
    javascript = client.get("/static/monitor.js")

    assert page.status_code == css.status_code == javascript.status_code == 200
    assert page.headers["X-DRES-Mock"] == "true"
    assert "text/html" in page.headers["content-type"]
    assert "text/css" in css.headers["content-type"]
    assert "javascript" in javascript.headers["content-type"]
    assert "LOCAL MOCK — DO NOT EXPOSE PUBLICLY" in page.text
    assert "Request detail" in page.text


def test_monitor_uses_only_same_origin_routes_and_contains_no_credentials(
    client: TestClient,
) -> None:
    """The initial page and script do not load external resources or secrets."""

    page = client.get("/monitor").text
    javascript = client.get("/static/monitor.js").text
    referenced_paths = re.findall(r"(?:href|src)=\"([^\"]+)\"", page)

    assert set(referenced_paths) == {"/static/monitor.css", "/static/monitor.js", "/docs"}
    assert all(path.startswith("/") for path in referenced_paths)
    assert "/__test/state" in javascript
    assert "/__test/task" in javascript
    assert "/__test/scenario/submission" in javascript
    assert "/__test/scenario/result-log" in javascript
    assert "/__test/reset" in javascript
    assert "https://" not in page + javascript
    assert "password-1" not in page
    assert "password-2" not in page
    assert ".innerHTML" not in javascript
    assert "textContent" in javascript


def test_monitor_has_accessible_controls_and_reduced_motion_support(
    client: TestClient,
) -> None:
    """Status, forms, focus, and motion preferences are represented in markup."""

    page = client.get("/monitor").text
    css = client.get("/static/monitor.css").text

    assert 'aria-live="polite"' in page
    assert "<fieldset>" in page
    assert "<details class=" in page
    assert "aria-pressed=" in page
    assert ":focus-visible" in css
    assert "prefers-reduced-motion" in css
    assert "@media (max-width: 760px)" in css


def test_monitor_asset_routes_reject_unlisted_paths(client: TestClient) -> None:
    """The asset handler only serves the two named local monitor files."""

    response = client.get("/static/secret.env")

    assert response.status_code == 404
    assert response.json()["status"] is False
