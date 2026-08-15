"""Download model sources and weights used by data and transcript pipelines."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from importlib import resources
from pathlib import Path
from typing import Any, Sequence

import yaml

DEFAULT_CONFIG = Path(__file__).with_name("model_sources.yaml")
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024


class _RemoteFile(io.RawIOBase):
    """Provide seekable HTTP range reads for a remote ZIP archive."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.position = 0
        request = urllib.request.Request(
            url,
            headers={
                "Range": "bytes=0-0",
                "User-Agent": "hcmai-model-setup",
            },
        )
        with urllib.request.urlopen(request) as response:
            content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or "/" not in content_range:
            raise RuntimeError("Model provider does not support range downloads")
        self.size = int(content_range.rsplit("/", 1)[1])

    def readable(self) -> bool:
        """Return that the remote file supports reads."""

        return True

    def seekable(self) -> bool:
        """Return that the remote file supports seeks."""

        return True

    def tell(self) -> int:
        """Return the current remote byte position."""

        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        """Move the current remote byte position."""

        if whence == os.SEEK_CUR:
            offset += self.position
        elif whence == os.SEEK_END:
            offset += self.size
        self.position = max(0, offset)
        return self.position

    def read(self, size: int = -1) -> bytes:
        """Read one byte range from the provider."""

        if size == 0 or self.position >= self.size:
            return b""
        end = self.size - 1 if size < 0 else min(
            self.position + size - 1, self.size - 1
        )
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={self.position}-{end}",
                "User-Agent": "hcmai-model-setup",
            },
        )
        with urllib.request.urlopen(request) as response:
            if response.status != 206:
                raise RuntimeError("Model provider ignored the requested byte range")
            data = response.read()
        self.position += len(data)
        return data


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid(path: Path, expected: str) -> bool:
    """Return whether a file exists with the expected digest."""

    return path.is_file() and _sha256(path) == expected


def _checkout(path: Path, settings: dict[str, Any], verify_only: bool) -> None:
    """Clone and pin one official source repository."""

    revision = settings["revision"]
    if not (path / ".git").is_dir():
        if verify_only:
            raise FileNotFoundError(f"Missing repository: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ | {"GIT_LFS_SKIP_SMUDGE": "1"}
        subprocess.run(
            ["git", "clone", "--quiet", settings["repository"], str(path)],
            check=True,
            env=environment,
        )
    current = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if current != revision:
        if verify_only:
            raise RuntimeError(f"Unexpected revision in {path}: {current}")
        subprocess.run(
            ["git", "-C", str(path), "fetch", "--depth", "1", "origin", revision],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(path), "checkout", "--detach", revision],
            check=True,
            env=os.environ | {"GIT_LFS_SKIP_SMUDGE": "1"},
        )


def _download(url: str, path: Path, digest: str, verify_only: bool) -> None:
    """Download one file atomically and verify its digest."""

    if _valid(path, digest):
        return
    if verify_only:
        raise RuntimeError(f"Missing or invalid model file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f"{path.suffix}.partial")
    request = urllib.request.Request(url, headers={"User-Agent": "hcmai-model-setup"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    if not _valid(partial, digest):
        raise RuntimeError(f"Checksum mismatch: {path}")
    partial.replace(path)


def _setup_transnet(root: Path, settings: dict[str, Any], verify_only: bool) -> None:
    """Install TransNetV2 source and official Git-LFS weights."""

    repository = root / "TransNetV2"
    _checkout(repository, settings, verify_only)
    for weight in settings["weights"]:
        _download(
            weight["url"],
            repository / weight["path"],
            weight["sha256"],
            verify_only,
        )
    print("TransNetV2: READY")


def _archive_member(archive: zipfile.ZipFile, path: str) -> zipfile.ZipInfo:
    """Find one configured file in the official checkpoint archive."""

    matches = [item for item in archive.infolist() if item.filename.endswith(path)]
    if len(matches) != 1:
        raise RuntimeError(f"Checkpoint archive does not contain exactly one {path}")
    return matches[0]


def _setup_efficientgebd(
    root: Path, settings: dict[str, Any], verify_only: bool,
) -> None:
    """Install EfficientGEBD source and the configured official checkpoint."""

    repository = root / "EfficientGEBD"
    _checkout(repository, settings, verify_only)
    outputs = (settings["config"], settings["checkpoint"])
    missing = [
        item for item in outputs
        if not _valid(repository / item["path"], item["sha256"])
    ]
    if not missing:
        print("EfficientGEBD: READY")
        return
    if verify_only:
        raise RuntimeError("EfficientGEBD checkpoint is missing or invalid")

    with tempfile.TemporaryDirectory(dir=repository) as temporary:
        staging = Path(temporary)
        for item in (item for item in missing if "source" in item):
            target = staging / Path(item["path"]).name
            shutil.copyfile(Path(__file__).with_name(item["source"]), target)
            _publish_model_file(repository, target, item)

        remote_items = [item for item in missing if "archive_path" in item]
        if remote_items:
            print("EfficientGEBD: downloading required checkpoint")
            remote = _RemoteFile(settings["checkpoint_url"])
            with zipfile.ZipFile(remote) as archive:
                for item in remote_items:
                    target = staging / Path(item["path"]).name
                    member = _archive_member(archive, item["archive_path"])
                    with archive.open(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output, DOWNLOAD_CHUNK_SIZE)
                    _publish_model_file(repository, target, item)
    print("EfficientGEBD: READY")


def _publish_model_file(
    repository: Path, source: Path, settings: dict[str, Any],
) -> None:
    """Verify and publish one extracted model file."""

    if not _valid(source, settings["sha256"]):
        raise RuntimeError(f"Checksum mismatch: {settings['path']}")
    destination = repository / settings["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)


def _setup_huggingface(
    home: Path, models: list[dict[str, Any]], verify_only: bool,
) -> None:
    """Download pinned Hugging Face snapshots into the configured cache."""

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    token = os.getenv("HF_TOKEN")
    cache = home / "hub"
    for model in models:
        options = {
            "repo_id": model["repository"],
            "revision": model["revision"],
            "cache_dir": cache,
            "token": token,
        }
        try:
            snapshot_download(**options, local_files_only=True)
        except LocalEntryNotFoundError:
            if verify_only:
                raise RuntimeError(f"Missing model: {model['repository']}")
            if model.get("gated") and not token:
                raise RuntimeError(
                    "HF_TOKEN is required for Pyannote Community-1"
                )
            snapshot_download(**options)
        print(f"{model['name']}: READY")


def _verify_silero(settings: dict[str, Any]) -> None:
    """Verify the Silero weight bundled by the installed package."""

    path = Path(str(resources.files("silero_vad.data").joinpath(settings["filename"])))
    if not _valid(path, settings["sha256"]):
        raise RuntimeError("Installed Silero VAD weight is missing or invalid")
    print("Silero VAD: READY")


def _write_preprocessing_config(
    root: Path, settings: dict[str, Any], output: Path,
) -> None:
    """Write a ready-to-edit preprocessing configuration."""

    values = dict(settings)
    values.update({
        "transnet_repo": str((root / "TransNetV2").resolve()),
        "transnet_weights": str(
            (root / "TransNetV2/inference/transnetv2-weights").resolve()
        ),
        "efficientgebd_repo": str((root / "EfficientGEBD").resolve()),
        "efficientgebd_config": str((
            root / "EfficientGEBD/output/x2x3x4_r50_eff/"
            "baseline_end_to_end_diff_former.yaml"
        ).resolve()),
        "efficientgebd_checkpoint": str((
            root / "EfficientGEBD/output/x2x3x4_r50_eff/model_best.pth"
        ).resolve()),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump({"preprocessing": values}, sort_keys=False),
        encoding="utf-8",
    )


def setup_models(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    root: str | Path | None = None,
    verify_only: bool = False,
) -> Path:
    """Install or verify every model used by preprocessing and transcripts."""

    with Path(config_path).open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    model_root = Path(
        root or os.getenv("HCMAI_MODELS_ROOT") or settings["models_root"]
    ).expanduser().resolve()
    huggingface_home = Path(
        os.getenv("HF_HOME") or settings["huggingface_home"]
    ).expanduser().resolve()
    _setup_transnet(model_root, settings["transnetv2"], verify_only)
    _setup_efficientgebd(model_root, settings["efficientgebd"], verify_only)
    _setup_huggingface(huggingface_home, settings["huggingface"], verify_only)
    _verify_silero(settings["silero_vad"])
    output = model_root / "preprocessing.yaml"
    if not verify_only:
        _write_preprocessing_config(model_root, settings["preprocessing"], output)
    print(f"Models root: {model_root}")
    print(f"Hugging Face cache: {huggingface_home}")
    print("Status: PASSED")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse model setup arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run model setup."""

    args = parse_args(argv)
    setup_models(args.config, root=args.root, verify_only=args.verify_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
