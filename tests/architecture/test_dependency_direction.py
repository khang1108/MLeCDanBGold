"""AST-based guards for runtime and offline dependency direction.

These checks inspect import syntax rather than raw source text so comments and
docstrings cannot accidentally create a reported boundary violation.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _python_files(root: Path) -> list[Path]:
    """Return Python sources below ``root``, excluding generated caches."""

    return sorted(
        path for path in root.rglob("*.py") if "__pycache__" not in path.parts
    )


def _imports(path: Path) -> list[str]:
    """Return absolute module names named by import statements in ``path``."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            # Relative imports have no bearing on the absolute package
            # boundaries enforced here.
            if node.level == 0:
                modules.append(node.module)
    return modules


def _violations(root: Path, forbidden: tuple[str, ...]) -> list[str]:
    """Find imports whose module is one of the forbidden prefixes."""

    violations: list[str] = []
    for path in _python_files(root):
        for module in _imports(path):
            if any(module == prefix or module.startswith(prefix + ".") for prefix in forbidden):
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)}: {module}")
    return violations


def test_runtime_packages_do_not_depend_on_forbidden_boundaries() -> None:
    """Prevent runtime corpus/temporal code from recoupling to API/offline code."""

    assert _violations(
        REPOSITORY_ROOT / "src" / "hcmai" / "corpus",
        ("hcmai.api", "offline"),
    ) == []
    assert _violations(
        REPOSITORY_ROOT / "src" / "hcmai" / "temporal",
        ("fastapi", "hcmai.api.contracts"),
    ) == []


def test_offline_package_does_not_depend_on_orchestration_or_api() -> None:
    """Keep offline builders independent of serving orchestration and HTTP."""

    assert _violations(
        REPOSITORY_ROOT / "offline",
        ("hcmai.orchestration", "hcmai.api"),
    ) == []


def test_runtime_callers_use_the_public_corpus_facade() -> None:
    """Require API, orchestration, and temporal callers to avoid specialist stores."""

    offenders: list[str] = []
    for root_name in ("api", "orchestration", "temporal"):
        root = REPOSITORY_ROOT / "src" / "hcmai" / root_name
        for path in _python_files(root):
            for module in _imports(path):
                if module == "hcmai.corpus.stores" or module.startswith(
                    "hcmai.corpus.stores."
                ) or module == "hcmai.corpus.assets" or module.startswith(
                    "hcmai.corpus.assets."
                ):
                    offenders.append(f"{path.relative_to(REPOSITORY_ROOT)}: {module}")
    assert offenders == []


def _declared_exports(path: Path) -> set[str]:
    """Read a literal ``__all__`` declaration from a package initializer."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    declaration = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    values = ast.literal_eval(declaration)
    assert isinstance(values, list)
    assert all(isinstance(value, str) for value in values)
    return set(values)


def test_package_exports_are_minimal_and_one_way() -> None:
    """Keep package facades limited to runtime contracts and numerical APIs."""

    corpus = _declared_exports(REPOSITORY_ROOT / "src/hcmai/corpus/__init__.py")
    assert corpus == {"Corpus", "Frame", "TranscriptSegment", "VideoMetadata"}

    temporal = _declared_exports(REPOSITORY_ROOT / "src/hcmai/temporal/__init__.py")
    assert temporal == {
        "DPPath",
        "AlignedPath",
        "align_video",
        "cluster_starts",
        "rank_paths",
        "split_query_events",
    }

    retrieval = _declared_exports(REPOSITORY_ROOT / "src/hcmai/retrieval/__init__.py")
    assert {"RetrievalService", "RetrievalCandidate", "RetrievalResult", "RetrievalSource"} <= retrieval
    assert not any("offline" in name.lower() for name in retrieval)

    offline = REPOSITORY_ROOT / "offline/__init__.py"
    assert _imports(offline) == []
