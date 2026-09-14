"""Fixtures for black-box tests of the standalone DRES mock server."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dres_mock_server.app import create_app
from dres_mock_server.settings import MockSettings


@pytest.fixture
def settings() -> MockSettings:
    """Build isolated test configuration without reading a local .env file."""

    return MockSettings(_env_file=None)


@pytest.fixture
def app(settings: MockSettings) -> FastAPI:
    """Create a fresh app and in-memory state for each test."""

    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Run the app through Starlette's synchronous HTTP test client."""

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def session_id(client: TestClient) -> str:
    """Log in one example member and return the DRES session token."""

    response = client.post(
        "/api/v2/login",
        json={"username": "member-1", "password": "password-1"},
    )
    assert response.status_code == 200
    return response.json()["sessionId"]
