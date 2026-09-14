"""Configuration for the standalone DRES mock server.

The settings own local bind, credential, evaluation, and default-task values.
Credential data is excluded from model representations and dumps, and this
module has no dependency on HCMAI configuration.
"""

from __future__ import annotations

import hmac
import json
from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


DEFAULT_USERS: dict[str, str] = {
    "member-1": "password-1",
    "member-2": "password-2",
}


class MockSettings(BaseSettings):
    """Settings and credential checks for a local DRES contract simulator.

    Passwords remain wrapped in ``SecretStr`` and the complete user map is
    omitted from representations and serialized settings output.
    """

    model_config = SettingsConfigDict(
        env_prefix="DRES_MOCK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
        populate_by_name=True,
    )

    host: str = "127.0.0.1"
    port: int = Field(default=8090, ge=1, le=65535)
    evaluation_id: str = "eval-vbs-local"
    evaluation_name: str = "Local DRES Mock Evaluation"
    evaluation_type: Literal["SYNCHRONOUS", "ASYNCHRONOUS", "NON_INTERACTIVE"] = "SYNCHRONOUS"
    users: Annotated[dict[str, SecretStr], NoDecode] = Field(
        default_factory=lambda: {
            username: SecretStr(password)
            for username, password in DEFAULT_USERS.items()
        },
        validation_alias="DRES_MOCK_USERS_JSON",
        repr=False,
        exclude=True,
    )

    default_task_name: str = "KIS task"
    default_task_group: str = "KIS"
    default_task_type: str = "KIS"
    default_task_duration: int = Field(default=300, ge=0)

    @field_validator("users", mode="before")
    @classmethod
    def parse_and_validate_users(cls, value: Any) -> dict[str, SecretStr]:
        """Parse the JSON user map while keeping validation messages generic."""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                raise ValueError("DRES_MOCK_USERS_JSON must be a JSON object.") from None

        if not isinstance(value, dict) or not value:
            raise ValueError("DRES_MOCK_USERS_JSON must be a non-empty JSON object.")

        users: dict[str, SecretStr] = {}
        for username, password in value.items():
            if not isinstance(username, str) or not username.strip():
                raise ValueError("DRES_MOCK_USERS_JSON usernames must not be blank.")

            if isinstance(password, SecretStr):
                password_value = password.get_secret_value()
            elif isinstance(password, str):
                password_value = password
            else:
                raise ValueError("DRES_MOCK_USERS_JSON passwords must be strings.")

            if not password_value.strip():
                raise ValueError("DRES_MOCK_USERS_JSON passwords must not be blank.")

            users[username] = SecretStr(password_value)

        return users

    def credential_matches(self, username: str, password: str) -> bool:
        """Return whether the supplied credentials match a configured user."""
        expected = self.users.get(username)
        if expected is None:
            return False

        # Compare encoded bytes so credentials may contain non-ASCII text.
        return hmac.compare_digest(
            expected.get_secret_value().encode("utf-8"),
            password.encode("utf-8"),
        )


__all__ = ["MockSettings"]
