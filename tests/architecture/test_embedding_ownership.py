"""Architecture checks for the runtime/offline embedding ownership boundary."""

from __future__ import annotations

import ast
from pathlib import Path

from hcmai.retrieval.embedding.pipeline import EmbeddingService


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
        "np.savetxt(",
        "to_parquet(",
        "to_csv(",
        "to_json(",
        "to_pickle(",
        "write_json(",
        "write_parquet(",
        "write_yaml(",
        ".write_text(",
        ".write_bytes(",
        "atomic_write(",
        ".mkdir(",
        "os.replace(",
        "shutil.move(",
        "shutil.copytree(",
    )

    violations = [
        f"{path.relative_to(embedding_root)}: {call}"
        for path in embedding_root.rglob("*.py")
        for call in forbidden_calls
        if call in path.read_text(encoding="utf-8")
    ]

    assert violations == []


def test_runtime_embedding_package_does_not_depend_on_offline_builds() -> None:
    """Prevent runtime query encoding from importing offline artifact builders."""
    embedding_root = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "hcmai"
        / "retrieval"
        / "embedding"
    )
    violations: list[str] = []
    for path in embedding_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [name.name for name in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module] if node.module is not None else []
            else:
                continue
            for module in modules:
                if module == "offline.embeddings" or module.startswith(
                    "offline.embeddings."
                ):
                    violations.append(
                        f"{path.relative_to(embedding_root)}: {module}"
                    )

    assert violations == []


def test_runtime_embedding_service_cannot_build_visual_artifacts() -> None:
    """Keep corpus writes out of the runtime service's public interface."""
    assert not hasattr(EmbeddingService, "build_visual_artifacts")
