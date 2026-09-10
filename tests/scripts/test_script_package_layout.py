"""Repository contracts for the domain-oriented script layout."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SCRIPT_DOMAINS: dict[str, tuple[str, ...]] = {
    "corpus": (
        "extract_custom_keyframes.py",
        "export_corpus_jsonl.py",
        "ingest_btc_keyframes.py",
        "materialize_custom_frames.py",
        "prepare_custom_pipeline.py",
    ),
    "enrichment": (
        "build_frame_context.py",
        "detect_objects.py",
        "generate_enrichment.py",
        "generate_ocr_enrichment.py",
        "prepare_transcripts.py",
        "translate.py",
    ),
    "indexing": ("build_retrieval_indexes.py",),
    "evaluation": (
        "benchmark_remote.py",
        "evaluate_benchmark.py",
        "evaluate_benchmark_v2.py",
        "evaluate_temporal_p0.py",
    ),
    "diagnostics": (
        "debug_temporal_evidence.py",
        "diagnose_temporal_quality.py",
    ),
    "review": (
        "build_overview.py",
        "build_review_sheets.py",
        "manual_label_unlabeled.py",
    ),
    "automation": ("watch_github_main.sh",),
}
PYTHON_MODULES = tuple(
    f"scripts.{domain}.{Path(filename).stem}"
    for domain, filenames in SCRIPT_DOMAINS.items()
    for filename in filenames
    if filename.endswith(".py")
)


def test_scripts_are_grouped_by_domain_without_flat_wrappers() -> None:
    """Every executable has one owning domain and no old root-level wrapper."""

    expected_names = {
        filename
        for filenames in SCRIPT_DOMAINS.values()
        for filename in filenames
    }
    assert len(expected_names) == 22

    for domain, filenames in SCRIPT_DOMAINS.items():
        domain_root = SCRIPTS_ROOT / domain
        assert domain_root.is_dir(), domain
        for filename in filenames:
            assert (domain_root / filename).is_file(), f"{domain}/{filename}"
            assert not (SCRIPTS_ROOT / filename).exists(), filename

    assert {path.name for path in SCRIPTS_ROOT.glob("*.py")} == {"__init__.py"}


def test_python_script_modules_import_from_their_domains() -> None:
    """Moved modules resolve without relying on a flat scripts namespace."""

    failures: list[str] = []
    for module_name in PYTHON_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as error:  # pragma: no cover - assertion reports details
            failures.append(f"{module_name}: {type(error).__name__}: {error}")

    assert failures == []


def test_moved_scripts_keep_repository_relative_defaults() -> None:
    """Adding one path component must not relocate repository-level defaults."""

    from scripts.corpus import prepare_custom_pipeline
    from scripts.enrichment import (
        build_frame_context,
        generate_ocr_enrichment,
        prepare_transcripts,
    )

    expected = PROJECT_ROOT / "configs" / "prepare.yaml"
    assert prepare_custom_pipeline.PROJECT_ROOT == PROJECT_ROOT
    assert prepare_custom_pipeline.DEFAULT_CONFIG == expected
    assert build_frame_context.DEFAULT_CONFIG == expected
    assert generate_ocr_enrichment.DEFAULT_CONFIG == expected
    assert prepare_transcripts.DEFAULT_CONFIG == expected


def test_custom_pipeline_runs_specialist_stages_as_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipeline subprocesses use importable modules, not removed flat files."""

    from scripts.corpus import prepare_custom_pipeline

    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> None:
        captured["command"] = command
        captured.update(kwargs)

    monkeypatch.setattr(prepare_custom_pipeline.subprocess, "run", fake_run)

    prepare_custom_pipeline._run_python(
        "scripts.enrichment.generate_enrichment",
        ["--config", "configs/prepare.yaml"],
    )

    assert captured["command"] == [
        sys.executable,
        "-m",
        "scripts.enrichment.generate_enrichment",
        "--config",
        "configs/prepare.yaml",
    ]
    assert captured["cwd"] == PROJECT_ROOT
    assert captured["check"] is True
    assert captured["shell"] is False
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert str(PROJECT_ROOT) in environment["PYTHONPATH"].split(":")
    assert str(PROJECT_ROOT / "src") in environment["PYTHONPATH"].split(":")


@pytest.mark.parametrize("module_name", PYTHON_MODULES)
def test_python_script_module_exposes_help(module_name: str) -> None:
    """Every moved Python executable remains runnable through ``python -m``."""

    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{PROJECT_ROOT}:{PROJECT_ROOT / 'src'}"
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, f"{module_name}: {result.stderr}"
    assert "usage:" in result.stdout.lower()
