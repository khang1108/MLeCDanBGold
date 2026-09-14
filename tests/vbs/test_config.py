"""Configuration boundary tests for the backend-only DRES integration."""

from __future__ import annotations

import pytest

from hcmai.vbs.config import DresConfigurationError, DresSettings


def test_dres_settings_defaults_logging_off_and_parses_private_credentials() -> None:
    """Use only configured VBS IDs in FE while keeping passwords redacted."""

    settings = DresSettings.from_env({
        "HCMAI_DRES_BASE_URL": "https://dres.example.test/",
        "HCMAI_DRES_USERS_JSON": (
            '{"member-1":{"username":"judge-user",'
            '"password":"super-secret"}}'
        ),
    })

    assert settings.base_url == "https://dres.example.test"
    assert settings.credentials["member-1"].username == "judge-user"
    assert settings.credentials["member-1"].password.get_secret_value() == "super-secret"
    assert "super-secret" not in repr(settings)
    assert "super-secret" not in repr(settings.credentials["member-1"])


@pytest.mark.parametrize(
    "base_url",
    ["", "ftp://dres.example.test", "https:///missing-host"],
)
def test_dres_settings_rejects_missing_or_non_http_base_url(base_url: str) -> None:
    """Require an HTTP(S) origin before constructing a network adapter."""

    with pytest.raises(DresConfigurationError, match="(?i)base_url"):
        DresSettings.from_env({"HCMAI_DRES_BASE_URL": base_url})


@pytest.mark.parametrize("timeout", ["0", "-1", "31", "not-a-number"])
def test_dres_settings_rejects_unbounded_timeout(timeout: str) -> None:
    """Keep DRES waits bounded so the UI can recover from service outages."""

    with pytest.raises(DresConfigurationError, match="(?i)timeout"):
        DresSettings.from_env({
            "HCMAI_DRES_BASE_URL": "https://dres.example.test",
            "HCMAI_DRES_TIMEOUT_SECONDS": timeout,
        })


def test_dres_settings_accepts_explicit_evaluation_and_exact_prefix_mapping() -> None:
    """Expose optional run selection and exact video-ID prefix handling."""

    settings = DresSettings.from_env({
        "HCMAI_DRES_BASE_URL": "http://127.0.0.1:8080",
        "HCMAI_DRES_EVALUATION_ID": "run-7",
        "HCMAI_DRES_MEDIA_ID_PREFIX_TO_STRIP": "prefix.",
    })

    assert settings.evaluation_id == "run-7"
    assert not hasattr(settings, "logging_enabled")
    assert settings.media_id_prefix_to_strip == "prefix."


def test_malformed_credential_json_is_redacted() -> None:
    """Never echo malformed secret-bearing JSON in startup errors."""

    raw = '{"member":{"username":"u","password":"dont-print-me"}'

    with pytest.raises(DresConfigurationError) as error:
        DresSettings.from_env({
            "HCMAI_DRES_BASE_URL": "https://dres.example.test",
            "HCMAI_DRES_USERS_JSON": raw,
        })

    assert "dont-print-me" not in str(error.value)


def test_credential_entries_require_username_and_password() -> None:
    """Reject partial credential rows without disclosing supplied values."""

    with pytest.raises(DresConfigurationError, match="credential"):
        DresSettings.from_env({
            "HCMAI_DRES_BASE_URL": "https://dres.example.test",
            "HCMAI_DRES_USERS_JSON": '{"member":{"username":"u"}}',
        })
