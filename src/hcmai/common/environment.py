"""Load repository-local runtime environment with explicit precedence.

The root ``.env`` file is the configuration authority for local HCMAI
launches. Values declared there intentionally replace variables inherited
from a terminal, IDE, service manager, or the operating system.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from hcmai.common.config import REPOSITORY_ROOT


def load_repository_environment(path: str | Path | None = None) -> bool:
    """Load the repository ``.env`` and override inherited process values.

    Variables absent from the file retain their normal process/default
    behavior. This keeps secrets optional while ensuring that values written
    to ``.env`` are not shadowed by stale exports in the launching terminal.
    """

    environment_path = Path(path) if path is not None else REPOSITORY_ROOT / ".env"
    return load_dotenv(environment_path, override=True)
