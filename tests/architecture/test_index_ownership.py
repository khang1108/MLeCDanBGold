"""Architecture checks for the runtime/offline retrieval-index boundary."""

from __future__ import annotations

import ast
from pathlib import Path

def test_runtime_service_exposes_load_and_search_but_no_index_builders() -> None:
    """Keep corpus writes out of the public runtime retrieval facade."""

    pipeline = (
        Path(__file__).resolve().parents[2]
        / "src/hcmai/retrieval/retriever/pipeline.py"
    )
    service = next(
        node
        for node in ast.parse(pipeline.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == "RetrievalService"
    )
    methods = {
        node.name
        for node in service.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "load_index" in methods
    for name in (
        "build_index",
        "build_text_artifacts",
        "build_context_artifacts",
        "build_asr_segment_artifacts",
        "build_text_embedding_artifacts",
    ):
        assert name not in methods


def test_runtime_retriever_modules_do_not_define_offline_build_functions() -> None:
    """Require all text/segment bundle construction to live under offline.indexes."""

    root = Path(__file__).resolve().parents[2] / "src/hcmai/retrieval/retriever"
    builder_names = {
        "build_text_index",
        "build_context_index",
        "build_text_artifacts",
        "build_context_artifacts",
        "build_asr_segment_index",
        "build_asr_segment_artifacts",
        "build_text_embedding_artifacts",
    }
    definitions = {
        node.name
        for path in (root / "text", root / "segment")
        for source in path.glob("*.py")
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert definitions.isdisjoint(builder_names)


def test_runtime_retrieval_package_does_not_depend_on_offline_index_builds() -> None:
    """Prevent serving modules from importing artifact-producing index builders."""

    root = Path(__file__).resolve().parents[2] / "src/hcmai/retrieval"
    violations: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules = (
                [name.name for name in node.names]
                if isinstance(node, ast.Import)
                else [node.module]
                if isinstance(node, ast.ImportFrom) and node.module is not None
                else []
            )
            for module in modules:
                if module == "offline.indexes" or module.startswith("offline.indexes."):
                    violations.append(f"{path.relative_to(root)}: {module}")

    assert violations == []
