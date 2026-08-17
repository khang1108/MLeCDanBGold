"""Tests for the Thunder batch launcher shell script."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def launcher_path() -> Path:
    path = Path(__file__).resolve().parents[2] / "scripts" / "thunder_batch_launcher.sh"
    assert path.is_file(), f"Launcher script not found at {path}"
    return path


def test_launcher_rejects_unknown_argument(launcher_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(launcher_path), "--unknown-arg"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Unknown argument: --unknown-arg" in result.stdout


def test_launcher_declares_optimized_run_switches(launcher_path: Path) -> None:
    text = launcher_path.read_text(encoding="utf-8")
    for option in ("--config", "--cache-only", "--no-resume", "--skip-install"):
        assert option in text


def test_launcher_has_valid_bash_syntax(launcher_path: Path) -> None:
    result = subprocess.run(
        ["bash", "-n", str(launcher_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_launcher_requires_hf_token(
    launcher_path: Path,
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake_nvidia_smi = bin_dir / "nvidia-smi"
    fake_nvidia_smi.write_text("#!/bin/bash\necho fake-gpu")
    fake_nvidia_smi.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["AWS_ACCESS_KEY_ID"] = "test"
    env["AWS_SECRET_ACCESS_KEY"] = "test"
    env.pop("HF_TOKEN", None)
    
    result = subprocess.run(
        ["bash", str(launcher_path)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 1
    assert "ERROR: HF_TOKEN not found in environment." in result.stdout


def test_launcher_cache_only_does_not_require_hf_token(
    launcher_path: Path,
) -> None:
    text = launcher_path.read_text(encoding="utf-8")
    assert '[[ $CACHE_ONLY -eq 0 && -z "${HF_TOKEN:-}" ]]' in text
