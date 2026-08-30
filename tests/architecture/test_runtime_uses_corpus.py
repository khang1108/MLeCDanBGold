"""Guard the one-way runtime dependency on the public Corpus facade."""

from __future__ import annotations

from pathlib import Path


def grep_python_sources(root: str | Path, needle: str) -> list[Path]:
    """Return Python source files containing ``needle``, excluding caches."""

    return sorted(
        path
        for path in Path(root).rglob("*.py")
        if "__pycache__" not in path.parts and needle in path.read_text()
    )


def test_runtime_does_not_reference_legacy_data_facade() -> None:
    """Keep deleted mutable data composition out of runtime modules."""

    legacy_facade = "Data" + "Service"
    offenders = grep_python_sources("src/hcmai", legacy_facade)
    assert offenders == []


def test_runtime_callers_do_not_import_corpus_implementations() -> None:
    """Require API, orchestration, and temporal code to use only Corpus."""

    roots = (
        Path("src/hcmai/api"),
        Path("src/hcmai/orchestration"),
        Path("src/hcmai/temporal"),
    )
    needles = ("hcmai.corpus.assets", "hcmai.corpus.stores")
    offenders = [
        path
        for root in roots
        for needle in needles
        for path in grep_python_sources(root, needle)
    ]
    assert sorted(set(offenders)) == []
