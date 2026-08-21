from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path


def test_extract_zips_script_extracts_all_archives(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    archive = data_dir / "sample.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("sample/hello.txt", "hello world")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/extract_zips.py",
            "--data-dir",
            str(data_dir),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "Archives processed: 1" in completed.stdout
    assert not archive.exists()
    assert (data_dir / "sample/hello.txt").read_text() == "hello world"
