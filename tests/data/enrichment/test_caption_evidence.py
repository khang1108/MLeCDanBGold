from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

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
    assert table.loc[0, "text"] == "A person runs."
    assert table.loc[1, "status"] == "failed"
    assert table.loc[1, "error_code"] == "RuntimeError"
    assert table.loc[2, "error_code"] == "EmptyCaption"
    assert first["artifact_version"] == "caption-evidence.v1"
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
