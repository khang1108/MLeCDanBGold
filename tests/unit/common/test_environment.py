"""Tests for repository-local environment precedence."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hcmai.common.environment import load_repository_environment


def test_repository_env_overrides_inherited_process_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checked-in launch value must replace a stale terminal export."""

    environment_path = tmp_path / ".env"
    environment_path.write_text(
        "HCMAI_LLM_CONFIG=llm/config.yaml\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HCMAI_LLM_CONFIG", "thundercompute/config.yaml")

    loaded = load_repository_environment(environment_path)

    assert loaded is True
    assert os.environ["HCMAI_LLM_CONFIG"] == "llm/config.yaml"
