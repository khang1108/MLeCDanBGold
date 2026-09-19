"""Small local run-state journal; artifacts remain the source of truth."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"version": 1, "stages": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "stages": {}}
    return value if isinstance(value, dict) else {"version": 1, "stages": {}}


def update_stage(path: Path, stage: str, status: str, *, detail: str | None = None) -> None:
    state = load_state(path)
    stages = state.setdefault("stages", {})
    row = {"status": status, "updated_at": _now()}
    if detail:
        row["detail"] = detail[:1000]
    stages[stage] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
