from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from hcmai.common.utils.io import write_yaml


def test_btc_import_does_not_require_preprocessing_fields(
    tmp_path, monkeypatch
):
    # Importing BTC-provided metadata must remain independent of preprocessing.
    monkeypatch.setitem(sys.modules, "hcmai.data.preprocessing", None)
    from hcmai.data.ingestion.btc import (
        BTCIngestionConfig,
        import_btc_frame_store,
    )

    source = tmp_path / "btc"
    (source / "metadata").mkdir(parents=True)
    keyframe = source / "keyframes" / "L01_V001" / "0000.jpg"
    keyframe.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24)).save(keyframe)
    pd.DataFrame(
        [
            {
                "frame_id": "L01_V001:0000",
                "video_id": "L01_V001",
                "frame_idx": 120,
                "keyframe_order": 1,
                "timestamp_ms": 4000,
                "image_path": "keyframes/L01_V001/0000.jpg",
                "width": 32,
                "height": 24,
            }
        ]
    ).to_parquet(source / "metadata" / "frames.parquet", index=False)

    output = import_btc_frame_store(
        BTCIngestionConfig(
            btc_root=source,
            data_root=source,
            output_root=tmp_path / "frame_store",
            frame_store_id="btc-test-v1",
        )
    )

    row = pd.read_parquet(output).iloc[0]
    assert row["frame_id"] == "L01_V001:0000"
    assert row["frame_idx"] == 120
    assert row["shot_id"] is None
    assert row["event_id"] is None
    assert row["pts"] is None
    assert row["time_base"] is None
    assert list(row["selection_reasons"]) == ["btc_keyframe"]

    manifest = json.loads((output.parent / "manifest.json").read_text())
    assert manifest["pipeline_version"] == "btc-keyframe-ingestion-v1"
    assert manifest["source"] == "btc_provided_keyframes"
    assert manifest["frame_store_id"] == "btc-test-v1"
    assert manifest["video_count"] == 1
    assert manifest["frame_count"] == 1


def test_ingestion_cli_delegates_to_reusable_importer(
    tmp_path, monkeypatch, capsys
):
    import scripts.ingest_btc_keyframes as cli

    captured = []
    expected_output = tmp_path / "out" / "frames.parquet"

    def fake_import(config):
        captured.append(config)
        return expected_output

    monkeypatch.setattr(cli, "import_btc_frame_store", fake_import)
    exit_code = cli.main(
        [
            "--btc-root",
            str(tmp_path / "btc"),
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "out"),
            "--frame-store-id",
            "btc-cli-v1",
        ]
    )

    assert exit_code == 0
    assert len(captured) == 1
    assert captured[0].btc_root == tmp_path / "btc"
    assert captured[0].data_root == tmp_path / "data"
    assert captured[0].output_root == tmp_path / "out"
    assert captured[0].frame_store_id == "btc-cli-v1"
    assert capsys.readouterr().out.strip() == str(expected_output)


def test_data_service_prepare_uses_btc_dataset_config(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "hcmai.data.preprocessing", None)
    from hcmai.data.pipeline import DataService

    source = tmp_path / "btc"
    (source / "metadata").mkdir(parents=True)
    keyframe = source / "keyframes" / "L01_V001" / "0000.jpg"
    keyframe.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24)).save(keyframe)
    pd.DataFrame(
        [
            {
                "frame_id": "L01_V001:0000",
                "video_id": "L01_V001",
                "frame_idx": 120,
                "keyframe_order": 1,
                "timestamp_ms": 4000,
                "image_path": "keyframes/L01_V001/0000.jpg",
                "width": 32,
                "height": 24,
            }
        ]
    ).to_parquet(source / "metadata" / "frames.parquet", index=False)

    output_root = tmp_path / "frame_store"
    config_path = tmp_path / "enrichment.yaml"
    write_yaml(
        {
            "dataset": {
                "version": "hcmai2026-test-v1",
                "source": "btc_keyframes",
                "btc_root": str(source),
                "data_root": str(source),
                "frame_store_id": "btc-service-v1",
                "frames_path": str(output_root / "frames.parquet"),
                "frame_store_output": str(output_root),
            }
        },
        config_path,
    )

    output = DataService.prepare(config_path)

    assert output == output_root / "frames.parquet"
    assert pd.read_parquet(output).iloc[0]["frame_idx"] == 120
    manifest = json.loads((output_root / "manifest.json").read_text())
    assert manifest["frame_store_id"] == "btc-service-v1"


def test_data_service_prepare_rejects_non_btc_source(tmp_path):
    from hcmai.data.pipeline import DataService

    output_root = tmp_path / "frame_store"
    config_path = tmp_path / "enrichment.yaml"
    write_yaml(
        {
            "dataset": {
                "version": "hcmai2026-test-v1",
                "source": "custom_videos",
                "btc_root": str(tmp_path / "btc"),
                "data_root": str(tmp_path / "btc"),
                "frame_store_id": "custom-v1",
                "frames_path": str(output_root / "frames.parquet"),
                "frame_store_output": str(output_root),
            }
        },
        config_path,
    )

    with pytest.raises(ValueError, match="expected 'btc_keyframes'"):
        DataService.prepare(config_path)


def test_prepare_data_cli_uses_btc_enrichment_config(tmp_path):
    source = tmp_path / "btc"
    (source / "metadata").mkdir(parents=True)
    keyframe = source / "keyframes" / "L01_V001" / "0000.jpg"
    keyframe.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24)).save(keyframe)
    pd.DataFrame(
        [
            {
                "frame_id": "L01_V001:0000",
                "video_id": "L01_V001",
                "frame_idx": 120,
                "keyframe_order": 1,
                "timestamp_ms": 4000,
                "image_path": "keyframes/L01_V001/0000.jpg",
                "width": 32,
                "height": 24,
            }
        ]
    ).to_parquet(source / "metadata" / "frames.parquet", index=False)

    output_root = tmp_path / "frame_store"
    config_path = tmp_path / "enrichment.yaml"
    write_yaml(
        {
            "dataset": {
                "version": "hcmai2026-test-v1",
                "source": "btc_keyframes",
                "btc_root": str(source),
                "data_root": str(source),
                "frame_store_id": "btc-cli-prepare-v1",
                "frames_path": str(output_root / "frames.parquet"),
                "frame_store_output": str(output_root),
            }
        },
        config_path,
    )

    project_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_data.py",
            "--config",
            str(config_path),
            "--dataset-root",
            str(source),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Videos: 1" in completed.stdout
    assert "Frames: 1" in completed.stdout
    assert "Status: PASSED" in completed.stdout
    assert (output_root / "frames.parquet").is_file()


def test_prepare_data_cli_delegates_config_and_optional_dataset_root(
    tmp_path, monkeypatch, capsys
):
    import scripts.prepare_data as cli

    config_path = tmp_path / "enrichment.yaml"
    dataset_root = tmp_path / "btc"
    output = tmp_path / "frame_store" / "frames.parquet"
    calls = []

    class FakeData:
        @staticmethod
        def iter_frames():
            return iter(())

    def fake_prepare(config):
        calls.append(("prepare", config))
        return output

    def fake_load(frames_path, *, dataset_root=None):
        calls.append(("load", frames_path, dataset_root))
        return FakeData()

    monkeypatch.setattr(cli.DataService, "prepare", fake_prepare)
    monkeypatch.setattr(cli.DataService, "load", fake_load)

    exit_code = cli.main(
        [
            "--config",
            str(config_path),
            "--dataset-root",
            str(dataset_root),
        ]
    )

    assert exit_code == 0
    assert calls == [
        ("prepare", config_path),
        ("load", output, dataset_root),
    ]
    assert "Status: PASSED" in capsys.readouterr().out
