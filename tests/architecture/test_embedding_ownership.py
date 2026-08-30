"""Architecture checks for the runtime/offline embedding ownership boundary."""

from __future__ import annotations

from pathlib import Path


def test_runtime_embedding_package_has_no_artifact_writes() -> None:
    """Keep vector persistence and artifact publication exclusively offline."""
    embedding_root = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "hcmai"
        / "retrieval"
        / "embedding"
    )
    forbidden_calls = (
        "np.save",
        "to_parquet(",
        "write_json(",
        "write_parquet(",
        "write_yaml(",
        "atomic_write(",
        ".mkdir(",
    )

    violations = [
        f"{path.relative_to(embedding_root)}: {call}"
        for path in embedding_root.rglob("*.py")
        for call in forbidden_calls
        if call in path.read_text(encoding="utf-8")
    ]

    assert violations == []
