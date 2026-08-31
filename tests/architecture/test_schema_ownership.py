"""Keep generic shared schemas out of runtime and offline implementation code."""

from __future__ import annotations

from pathlib import Path


def grep_python_sources(root: str | Path, needle: str) -> list[Path]:
    """Return non-cache Python files under ``root`` containing ``needle``."""

    return sorted(
        path
        for path in Path(root).rglob("*.py")
        if "__pycache__" not in path.parts and needle in path.read_text()
    )


def test_common_schema_package_is_not_imported() -> None:
    """Require contracts to be imported from their domain owner modules."""

    retired_package = "hcmai.common." + "schemas"
    assert grep_python_sources("src/hcmai", retired_package) == []
    assert grep_python_sources("offline", retired_package) == []
