from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from hcmai.common.schemas import CaptionEvidence, FrameEnrichment
from hcmai.data.enrichment.caption.config import CaptionConfig
from hcmai.data.enrichment.caption.generator import generate_captions


class FakeCaptionAdapter:
    def __init__(self, results: list[object]) -> None:
        self._results = results
        self.calls: list[int] = []
        self.resolved_revision: str | None = None

    def resolve_revision(self) -> str:
        self.resolved_revision = "fake-revision"
        return self.resolved_revision

    def caption_batch(self, images: list[object]) -> list[object]:
        self.calls.append(len(images))
        results, self._results = self._results[: len(images)], self._results[len(images) :]
        return results


def _config() -> CaptionConfig:
    return CaptionConfig(
        model_checkpoint="fake/captioner",
        revision="fake-revision",
        prompt="<CAPTION>",
        decoding={"max_new_tokens": 8},
        device="cpu",
        precision="fp32",
        dtype="float32",
        image_size=16,
        batch_size=3,
        enrichment_version="caption-test-v1",
        write_interval=3,
        dataset_version="test-v1",
    )


def _frames(root: Path) -> Path:
    rows = []
    for index in range(3):
        image_path = root / f"frame-{index}.jpg"
        Image.new("RGB", (16, 12)).save(image_path)
        rows.append(
            {
                "frame_id": f"f{index}",
                "video_id": "v1",
                "frame_idx": index,
                "timestamp_ms": index * 1000,
                "image_path": image_path.name,
                "width": 16,
                "height": 12,
            }
        )
    path = root / "frames.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_caption_artifact_keeps_failure_identity_and_retries_only_incomplete_rows(
    tmp_path: Path,
) -> None:
    frames_path = _frames(tmp_path)
    output = tmp_path / "captions"
    first_adapter = FakeCaptionAdapter(
        ["A person runs.", RuntimeError("oom"), ""]
    )

    first = generate_captions(
        frames_path,
        output,
        _config(),
        first_adapter,
        dataset_root=tmp_path,
        frame_store_id="btc-test-v1",
    )

    table = pd.read_parquet(output / "captions.parquet")
    assert table.loc[0, "video_id"] == "v1"
    assert table.loc[0, "frame_idx"] == 0
    assert table.loc[0, "timestamp_ms"] == 0
    assert table.loc[0, "text"] == "A person runs."
    assert table.loc[1, "status"] == "failed"
    assert table.loc[1, "error_code"] == "RuntimeError"
    assert table.loc[2, "error_code"] == "EmptyCaption"
    assert first["artifact_version"] == "caption-test-v1"
    assert table["artifact_version"].eq("caption-test-v1").all()
    assert first["source_artifact"] == "captions.parquet"

    second_adapter = FakeCaptionAdapter(["Recovered one.", "Recovered two."])
    second = generate_captions(
        frames_path,
        output,
        _config(),
        second_adapter,
        dataset_root=tmp_path,
        frame_store_id="btc-test-v1",
    )

    assert second_adapter.calls == [2]
    assert second["skipped_count"] == 1
    assert second["retried_count"] == 2
    assert pd.read_parquet(output / "captions.parquet")["status"].eq("completed").all()
    legacy = pd.read_parquet(output / "frame_enrichment.parquet")
    assert legacy["objects"].map(len).eq(0).all()


def test_resume_retries_caption_with_mismatched_canonical_timestamp(
    tmp_path: Path,
) -> None:
    """Never reuse a caption row aligned to a different source timestamp."""

    frames_path = _frames(tmp_path)
    output = tmp_path / "captions"
    generate_captions(
        frames_path,
        output,
        _config(),
        FakeCaptionAdapter(["zero", "one", "two"]),
        dataset_root=tmp_path,
        frame_store_id="btc-test-v1",
    )
    rows = pd.read_parquet(output / "captions.parquet")
    rows.loc[1, "timestamp_ms"] = 999
    rows.to_parquet(output / "captions.parquet", index=False)

    retry = FakeCaptionAdapter(["one fixed"])
    report = generate_captions(
        frames_path,
        output,
        _config(),
        retry,
        dataset_root=tmp_path,
        frame_store_id="btc-test-v1",
    )

    rows = pd.read_parquet(output / "captions.parquet")
    assert retry.calls == [1]
    assert report["retried_count"] == 1
    assert rows.timestamp_ms.tolist() == [0, 1_000, 2_000]


def test_empty_caption_bundle_keeps_explicit_canonical_schemas(
    tmp_path: Path,
) -> None:
    """Publish readable zero-row source and compatibility Parquet tables."""

    frames_path = tmp_path / "frames.parquet"
    pd.DataFrame(
        columns=[
            "frame_id",
            "video_id",
            "frame_idx",
            "timestamp_ms",
            "image_path",
            "width",
            "height",
        ]
    ).to_parquet(frames_path, index=False)

    report = generate_captions(
        frames_path,
        tmp_path / "captions",
        _config(),
        FakeCaptionAdapter([]),
        dataset_root=tmp_path,
        frame_store_id="btc-test-v1",
    )

    assert report["input_frame_count"] == 0
    assert list(pd.read_parquet(tmp_path / "captions/captions.parquet")) == list(
        CaptionEvidence.model_fields
    )
    assert list(
        pd.read_parquet(tmp_path / "captions/frame_enrichment.parquet")
    ) == list(FrameEnrichment.model_fields)


@pytest.mark.parametrize(
    "failed_name", ["frame_enrichment.parquet", "manifest.json"]
)
def test_caption_publication_failure_restores_prior_complete_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_name: str,
) -> None:
    """Publish Caption data as one manifest-committed rollback-safe bundle."""

    frames_path = _frames(tmp_path)
    output = tmp_path / "captions"
    generate_captions(
        frames_path,
        output,
        _config(),
        FakeCaptionAdapter(["zero", RuntimeError("retry"), "two"]),
        dataset_root=tmp_path,
        frame_store_id="btc-test-v1",
    )
    names = (
        "captions.parquet",
        "failures.json",
        "frame_enrichment.parquet",
        "manifest.json",
    )
    before = {name: (output / name).read_bytes() for name in names}
    assert json.loads(before["manifest.json"])["failed_count"] == 1

    original_replace = Path.replace
    injected = False

    def fail_publish_once(source: Path, target: Path) -> Path:
        nonlocal injected
        if source.name == f".{failed_name}.staged" and not injected:
            injected = True
            raise OSError(f"injected {failed_name} publication failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_publish_once)

    with pytest.raises(OSError, match="injected"):
        generate_captions(
            frames_path,
            output,
            _config(),
            FakeCaptionAdapter(["one recovered"]),
            dataset_root=tmp_path,
            frame_store_id="btc-test-v1",
        )

    assert injected
    assert {name: (output / name).read_bytes() for name in names} == before
    assert {
        path.name
        for path in output.iterdir()
        if path.name.endswith((".staged", ".backup", ".tmp"))
    } == set()
