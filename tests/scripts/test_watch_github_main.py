"""Test the unattended, fast-forward-only GitHub branch watcher.

These tests use local Git repositories and never contact GitHub. They verify
the CLI contract and the real fetch/merge path without touching user data.
"""

from __future__ import annotations

from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "watch_github_main.sh"


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run Git with isolated author identity for a local test repository."""

    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=HCMAI Test",
            "-c",
            "user.email=hcmai-test@example.invalid",
            *args,
        ],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def test_help_documents_safe_update_policy() -> None:
    """Expose the default GitHub target and non-destructive update rules."""

    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "khang1108/MLeCDanBGold" in result.stdout
    assert "fast-forward-only" in result.stdout
    assert "dirty worktree is never modified" in result.stdout


def test_once_fast_forwards_a_clean_clone(tmp_path: Path) -> None:
    """Fetch and apply one new main commit from a local remote in one pass."""

    remote = tmp_path / "remote.git"
    source = tmp_path / "source"
    deployed = tmp_path / "deployed"

    _git("init", "--bare", "--initial-branch=main", str(remote))
    _git("init", "--initial-branch=main", str(source))
    (source / "version.txt").write_text("v1\n", encoding="utf-8")
    _git("add", "version.txt", cwd=source)
    _git("commit", "-m", "initial", cwd=source)
    _git("remote", "add", "origin", str(remote), cwd=source)
    _git("push", "-u", "origin", "main", cwd=source)
    _git("clone", str(remote), str(deployed))

    (source / "version.txt").write_text("v2\n", encoding="utf-8")
    _git("add", "version.txt", cwd=source)
    _git("commit", "-m", "update", cwd=source)
    _git("push", "origin", "main", cwd=source)

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--repo-dir",
            str(deployed),
            "--once",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (deployed / "version.txt").read_text(encoding="utf-8") == "v2\n"
    assert "Updated main" in result.stderr


def test_once_preserves_dirty_worktree(tmp_path: Path) -> None:
    """Leave both local content and HEAD unchanged when an update is available."""

    remote = tmp_path / "remote.git"
    source = tmp_path / "source"
    deployed = tmp_path / "deployed"

    _git("init", "--bare", "--initial-branch=main", str(remote))
    _git("init", "--initial-branch=main", str(source))
    (source / "version.txt").write_text("v1\n", encoding="utf-8")
    _git("add", "version.txt", cwd=source)
    _git("commit", "-m", "initial", cwd=source)
    _git("remote", "add", "origin", str(remote), cwd=source)
    _git("push", "-u", "origin", "main", cwd=source)
    _git("clone", str(remote), str(deployed))
    original_head = _git("rev-parse", "HEAD", cwd=deployed).stdout.strip()

    (source / "remote.txt").write_text("remote\n", encoding="utf-8")
    _git("add", "remote.txt", cwd=source)
    _git("commit", "-m", "remote update", cwd=source)
    _git("push", "origin", "main", cwd=source)
    (deployed / "local.txt").write_text("do not overwrite\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(SCRIPT), "--repo-dir", str(deployed), "--once"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert _git("rev-parse", "HEAD", cwd=deployed).stdout.strip() == original_head
    assert (deployed / "local.txt").read_text(encoding="utf-8") == "do not overwrite\n"
    assert "worktree has local changes" in result.stderr
