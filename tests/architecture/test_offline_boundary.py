"""Enforce the one-way boundary between runtime code and offline builders."""

from __future__ import annotations

from pathlib import Path


def grep_python_sources(root: str | Path, needle: str) -> list[Path]:
    """Return non-cache Python files under ``root`` containing ``needle``."""

    return sorted(
        path
        for path in Path(root).rglob("*.py")
        if "__pycache__" not in path.parts and needle in path.read_text()
    )


def test_runtime_does_not_import_offline_package() -> None:
    """Runtime modules must not depend on artifact-producing offline code."""

    assert grep_python_sources("src/hcmai", "from offline") == []
    assert grep_python_sources("src/hcmai", "import offline") == []


def test_offline_directories_are_python_packages() -> None:
    """Every offline package directory has an explicit package initializer."""

    offline_root = Path("offline")
    package_dirs = sorted(
        path
        for path in offline_root.rglob("*")
        if path.is_dir()
        and "__pycache__" not in path.parts
        and any(path.glob("*.py"))
    )
    assert (offline_root / "__init__.py").is_file()
    missing = [path for path in package_dirs if not (path / "__init__.py").is_file()]
    assert missing == []
