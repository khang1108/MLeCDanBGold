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


def _package_name(path: Path) -> str:
    """Return the dotted package containing a Python source file."""

    for root_name in ("src", "offline"):
        root = REPOSITORY_ROOT / root_name
        try:
            relative = path.relative_to(root).with_suffix("")
        except ValueError:
            continue
        return ".".join(relative.parts[:-1])
    return ""


def _imports(path: Path) -> list[str]:
    """Return canonical modules named by absolute and relative imports."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            package = _package_name(path)
            if node.level:
                parent = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join(parent)
            else:
                base = node.module or ""
            for alias in node.names:
                imported = ".".join(part for part in (base, alias.name) if part)
                if imported:
                    modules.append(imported)
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


def test_offline_package_does_not_depend_on_runtime_serving_or_corpus() -> None:
    """Keep offline construction independent of runtime serving and corpus readers."""

    assert _violations(
        REPOSITORY_ROOT / "offline",
        ("hcmai.orchestration", "hcmai.api", "hcmai.corpus"),
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
        "plan_query_events",
        "rank_paths",
        "split_query_events",
    }

    retrieval_path = REPOSITORY_ROOT / "src/hcmai/retrieval/__init__.py"
    retrieval = _declared_exports(retrieval_path)
    assert retrieval == {
        "RetrievalService",
        "RetrievalCandidate",
        "RetrievalResult",
        "RetrievalSource",
    }
    assert _violations(retrieval_path.parent, ("offline",)) == []

    offline = REPOSITORY_ROOT / "offline/__init__.py"
    assert _imports(offline) == []
