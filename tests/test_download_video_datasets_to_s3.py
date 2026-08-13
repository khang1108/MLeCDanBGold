from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/download_video_datasets_to_s3.py"
SPEC = importlib.util.spec_from_file_location("download_video_datasets_to_s3", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_read_links_and_extract_expected_layout(tmp_path: Path) -> None:
    links = tmp_path / "links.tsv"
    links.write_text("Videos_L21_a.zip\thttps://example.test/a.zip\n", encoding="utf-8")
    assert module.read_links(links) == [
        ("Videos_L21_a.zip", "https://example.test/a.zip")
    ]

    archive = tmp_path / "Videos_L21_a.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("nested/video/L21_V001.mp4", b"video")
        output.writestr("nested/metadata.txt", b"ignored")

    batch = module.extract_batch(archive, tmp_path, "Videos_L21_a")

    assert (batch / "videos" / "L21_V001.mp4").read_bytes() == b"video"
    assert not archive.exists()


def test_extract_rejects_duplicate_flattened_video_names(tmp_path: Path) -> None:
    archive = tmp_path / "batch.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("a/video.mp4", b"a")
        output.writestr("b/video.mp4", b"b")

    with pytest.raises(ValueError, match="Duplicate video filename"):
        module.extract_batch(archive, tmp_path, "batch")

    assert archive.exists()
    assert not (tmp_path / "batch").exists()


def test_upload_preserves_batch_and_videos_prefix(tmp_path: Path) -> None:
    video_dir = tmp_path / "Videos_L21_a" / "videos"
    video_dir.mkdir(parents=True)
    (video_dir / "L21_V001.mp4").write_bytes(b"video")

    calls = []

    class FakeS3:
        def upload_file(self, filename, bucket, key, Callback):
            calls.append((filename, bucket, key))
            Callback(Path(filename).stat().st_size)

    count = module.upload_batch(FakeS3(), video_dir.parent, "dataset-bucket", "hcmai")

    assert count == 1
    assert calls == [
        (
            str(video_dir / "L21_V001.mp4"),
            "dataset-bucket",
            "hcmai/Videos_L21_a/videos/L21_V001.mp4",
        )
    ]
