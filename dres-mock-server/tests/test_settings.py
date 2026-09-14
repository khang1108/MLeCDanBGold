"""Settings tests for the standalone mock server's safe defaults and secrets."""

from __future__ import annotations

import json

import pytest
from pydantic import SecretStr, ValidationError

from dres_mock_server.settings import MockSettings


def test_settings_use_documented_local_defaults() -> None:
    """The default server is local-only and has harmless example accounts."""
    settings = MockSettings(_env_file=None)

    assert settings.host == "127.0.0.1"
    assert settings.port == 8090
    assert settings.evaluation_id == "eval-vbs-local"
    assert settings.evaluation_type == "SYNCHRONOUS"
    assert set(settings.users) == {"member-1", "member-2"}
    assert all(isinstance(password, SecretStr) for password in settings.users.values())
    assert settings.credential_matches("member-1", "password-1")
    assert not settings.credential_matches("member-1", "wrong-password")
    assert not settings.credential_matches("unknown", "password-1")


def test_settings_parse_json_user_map_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The isolated DRES_MOCK_USERS_JSON setting accepts a JSON object."""
    monkeypatch.setenv(
        "DRES_MOCK_USERS_JSON",
        json.dumps({"operator": "new-password", "viewer": "another-secret"}),
    )
    monkeypatch.setenv("DRES_MOCK_HOST", "0.0.0.0")
    monkeypatch.setenv("DRES_MOCK_PORT", "9012")
    monkeypatch.setenv("DRES_MOCK_EVALUATION_ID", "eval-local-test")

    settings = MockSettings(_env_file=None)

    assert settings.host == "0.0.0.0"
    assert settings.port == 9012
    assert settings.evaluation_id == "eval-local-test"
    assert settings.credential_matches("operator", "new-password")
    assert settings.credential_matches("viewer", "another-secret")
    assert not settings.credential_matches("member-1", "password-1")


@pytest.mark.parametrize(
    "users",
    [
        {"": "valid-password"},
        {"   ": "valid-password"},
        {"member": ""},
        {"member": "   "},
        {},
    ],
)
def test_settings_reject_blank_usernames_and_passwords(users: dict[str, str]) -> None:
    """An empty user map or blank username/password cannot enable login."""
    with pytest.raises(ValidationError):
        MockSettings(_env_file=None, users=users)


def test_settings_and_validation_errors_redact_credentials() -> None:
    """Passwords never appear in settings repr or validation error text."""
    secret = "do-not-print-this-password"
    settings = MockSettings(_env_file=None, users={"member": secret})

    assert secret not in repr(settings)
    assert "member" not in repr(settings)
    assert secret not in repr(settings.model_dump())

    with pytest.raises(ValidationError) as error:
        MockSettings(_env_file=None, users={"": secret})

    assert secret not in str(error.value)


def test_settings_reject_unsupported_evaluation_type() -> None:
    """Configured evaluation metadata stays inside DRES's declared enum."""

    with pytest.raises(ValidationError):
        MockSettings(_env_file=None, evaluation_type="VIDEO_RETRIEVAL")
