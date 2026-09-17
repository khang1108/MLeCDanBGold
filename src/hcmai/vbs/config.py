"""Parse backend-only DRES configuration and keep credentials out of reprs.

This module owns environment validation for the DRES adapter. It does not
perform network calls or expose credentials to API contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Mapping
from urllib.parse import urlsplit

from pydantic import SecretStr


class DresConfigurationError(ValueError):
    """Report invalid DRES configuration without including secret input."""


@dataclass(frozen=True)
class DresCredential:
    """One participant's DRES login, with password redacted from diagnostics."""

    username: str
    password: SecretStr = field(repr=False)


@dataclass(frozen=True)
class DresSettings:
    """Validated settings used only by backend DRES services and clients."""

    base_url: str
    timeout_seconds: float = 3.0
    evaluation_id: str | None = None
    credentials: Mapping[str, DresCredential] = field(default_factory=dict, repr=False)
    media_id_prefix_to_strip: str = ""
    admin_credential: DresCredential | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "DresSettings":
        """Build settings from an environment mapping without leaking secret text."""

        values = os.environ if environ is None else environ
        raw_url = values.get("HCMAI_DRES_BASE_URL", "").strip().rstrip("/")
        if raw_url.endswith("/api/v2"):
            raw_url = raw_url[:-len("/api/v2")].rstrip("/")
        parsed_url = urlsplit(raw_url)
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.hostname
            or parsed_url.username is not None
            or parsed_url.password is not None
        ):
            raise DresConfigurationError(
                "HCMAI_DRES_BASE_URL must be an HTTP(S) URL without embedded credentials"
            )

        try:
            timeout = float(values.get("HCMAI_DRES_TIMEOUT_SECONDS", "3"))
        except (TypeError, ValueError):
            raise DresConfigurationError(
                "HCMAI_DRES_TIMEOUT_SECONDS must be between 0 and 30 seconds"
            ) from None
        if not 0 < timeout <= 30:
            raise DresConfigurationError(
                "HCMAI_DRES_TIMEOUT_SECONDS must be between 0 and 30 seconds"
            )

        evaluation_id = values.get("HCMAI_DRES_EVALUATION_ID", "").strip() or None

        credentials = _parse_credentials(values.get("HCMAI_DRES_USERS_JSON", "{}"))
        media_prefix = values.get("HCMAI_DRES_MEDIA_ID_PREFIX_TO_STRIP", "")

        admin_user = values.get("HCMAI_DRES_ADMIN_USERNAME", "").strip()
        admin_pass = values.get("HCMAI_DRES_ADMIN_PASSWORD", "")
        admin_credential = None
        if admin_user and admin_pass:
            admin_credential = DresCredential(
                username=admin_user,
                password=SecretStr(admin_pass),
            )
        elif "admin" in credentials:
            admin_credential = credentials["admin"]

        return cls(
            base_url=raw_url,
            timeout_seconds=timeout,
            evaluation_id=evaluation_id,
            credentials=credentials,
            media_id_prefix_to_strip=media_prefix,
            admin_credential=admin_credential,
        )


def _parse_credentials(raw: str) -> dict[str, DresCredential]:
    """Parse user credentials and replace parser errors with a redacted message."""

    try:
        entries = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        raise DresConfigurationError(
            "HCMAI_DRES_USERS_JSON must be a JSON object keyed by VBS user ID"
        ) from None
    if not isinstance(entries, dict):
        raise DresConfigurationError(
            "HCMAI_DRES_USERS_JSON must be a JSON object keyed by VBS user ID"
        )

    parsed: dict[str, DresCredential] = {}
    for user_id, entry in entries.items():
        if (
            not isinstance(user_id, str)
            or not user_id.strip()
            or not isinstance(entry, dict)
            or set(entry) != {"username", "password"}
            or not isinstance(entry.get("username"), str)
            or not entry["username"].strip()
            or not isinstance(entry.get("password"), str)
            or not entry["password"]
        ):
            raise DresConfigurationError(
                "Each DRES credential must include a non-empty username and password"
            )
        parsed[user_id] = DresCredential(
            username=entry["username"],
            password=SecretStr(entry["password"]),
        )
    return parsed


__all__ = ["DresConfigurationError", "DresCredential", "DresSettings"]
