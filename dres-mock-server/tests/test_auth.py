"""Black-box tests for DRES login, session lookup, and logout behavior."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_login_returns_a_session_without_echoing_password(client: TestClient) -> None:
    """Successful login returns the DRES ApiUser shape and an opaque token."""

    response = client.post(
        "/api/v2/login",
        json={"username": "member-1", "password": "password-1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["username"] == "member-1"
    assert body["role"] == "PARTICIPANT"
    assert body["sessionId"]
    assert "password" not in body
    assert "password-1" not in response.text
    assert response.headers["X-DRES-Mock"] == "true"


def test_failed_login_does_not_echo_or_capture_password(client: TestClient) -> None:
    """Bad credentials use a generic DRES error without disclosing the input."""

    secret = "submitted-password-that-must-stay-private"
    response = client.post(
        "/api/v2/login",
        json={"username": "member-1", "password": secret},
    )

    assert response.status_code == 401
    assert response.json()["status"] is False
    assert secret not in response.text


def test_invalid_login_body_returns_dres_400_without_input_values(
    client: TestClient,
) -> None:
    """Strict schema errors do not echo a malformed password or extra fields."""

    secret = "body-secret-value"
    response = client.post(
        "/api/v2/login",
        json={"username": "member-1", "password": secret, "unexpected": "x"},
    )

    assert response.status_code == 400
    assert response.json()["status"] is False
    assert secret not in response.text


def test_protected_route_rejects_missing_and_unknown_sessions(
    client: TestClient,
) -> None:
    """Missing and expired DRES sessions return a safe unauthorized body."""

    missing = client.get("/api/v2/client/evaluation/list")
    expired_token = "expired-session-token"
    expired = client.get(
        "/api/v2/client/evaluation/list",
        params={"session": expired_token},
    )

    assert missing.status_code == expired.status_code == 401
    assert missing.json() == expired.json()
    assert expired_token not in expired.text


def test_logout_invalidates_only_the_selected_session(client: TestClient) -> None:
    """A logged-out session fails subsequent protected requests."""

    first = client.post(
        "/api/v2/login",
        json={"username": "member-1", "password": "password-1"},
    ).json()["sessionId"]
    second = client.post(
        "/api/v2/login",
        json={"username": "member-2", "password": "password-2"},
    ).json()["sessionId"]

    response = client.get("/api/v2/logout", params={"session": first})

    assert response.status_code == 200
    assert response.json() == {"status": True, "description": "session closed"}
    assert client.get(
        "/api/v2/client/evaluation/list",
        params={"session": first},
    ).status_code == 401
    assert client.get(
        "/api/v2/client/evaluation/list",
        params={"session": second},
    ).status_code == 200
