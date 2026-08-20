from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from hcmai.common.utils.io import write_yaml


def _valid_btc_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "frame_id": "L01_V001:0000",
        "video_id": "L01_V001",
        "frame_idx": 120,
        "keyframe_order": 1,
        "timestamp_ms": 4000,
        "image_path": "keyframes/L01_V001/0000.jpg",
        "width": 32,
        "height": 24,
    }
    row.update(updates)
    return row


def _write_btc_metadata(
    root: Path, rows: list[dict[str, object]] | pd.DataFrame
) -> None:
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(metadata / "frames.parquet", index=False)


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
    assert not (output.parent / ".frames.parquet.staged").exists()
    assert not (output.parent / ".manifest.json.staged").exists()
    assert not (output.parent / ".frames.parquet.backup").exists()
    assert not (output.parent / ".manifest.json.backup").exists()


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
    data = DataService.load(output)
    assert data.contains_submission("L01_V001", 120)
    assert not data.contains_submission("L01_V001", 1)
    manifest = json.loads((output_root / "manifest.json").read_text())
    assert manifest["frame_store_id"] == "btc-service-v1"


def test_data_service_prepare_resolves_btc_paths_from_project_root(
    tmp_path, monkeypatch
):
    """Keep active BTC output paths independent of the command's CWD."""

    from hcmai.data.pipeline import DataService

    project_root = Path(__file__).resolve().parents[2]
    config_path = tmp_path / "enrichment.yaml"
    write_yaml(
        {
            "dataset": {
                "version": "hcmai2026-test-v1",
                "source": "btc_keyframes",
                "btc_root": "fixture/btc",
                "data_root": "fixture/data",
                "frame_store_id": "btc-cwd-v1",
                "frames_path": "fixture/artifacts/frame_store/frames.parquet",
                "frame_store_output": "fixture/artifacts/frame_store",
            }
        },
        config_path,
    )
    captured = []

    def fake_import(config):
        """Expose the resolved BTC configuration without filesystem ingestion."""

        captured.append(config)
        return config.output_root / "frames.parquet"

    monkeypatch.setattr("hcmai.data.pipeline.import_btc_frame_store", fake_import)
    monkeypatch.chdir(tmp_path)

    output = DataService.prepare(config_path)

    assert output == project_root / "fixture/artifacts/frame_store/frames.parquet"
    assert captured[0].btc_root == project_root / "fixture/btc"
    assert captured[0].data_root == project_root / "fixture/data"
    assert captured[0].output_root == project_root / "fixture/artifacts/frame_store"


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


def test_data_service_prepare_adaptive_keeps_legacy_local_entry(
    tmp_path, monkeypatch
):
    from hcmai.data.pipeline import DataService

    config_path = tmp_path / "preprocessing.yaml"
    write_yaml(
        {
            "preprocessing": {
                "videos_root": str(tmp_path / "videos"),
                "output_root": str(tmp_path / "frame_store"),
            }
        },
        config_path,
    )
    expected = tmp_path / "legacy-frames.parquet"
    calls = []

    def fake_prepare(config, *, resume, limit):
        calls.append((config, resume, limit))
        return expected

    monkeypatch.setattr(
        "hcmai.data.preprocessing.prepare_frame_store", fake_prepare
    )

    output = DataService.prepare_adaptive(config_path, resume=False, limit=2)

    assert output == expected
    assert len(calls) == 1
    config, resume, limit = calls[0]
    assert config.videos_root == tmp_path / "videos"
    assert config.s3 is None
    assert resume is False
    assert limit == 2


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


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"frame_id": None}, "frame_id"),
        ({"video_id": "  "}, "video_id"),
        ({"frame_idx": -1}, "frame_idx"),
        ({"timestamp_ms": -1}, "timestamp_ms"),
        ({"keyframe_order": 0}, "keyframe_order"),
        ({"image_path": ["not-a-scalar.jpg"]}, "image_path"),
        ({"width": 0}, "width"),
        ({"height": -1}, "height"),
    ],
)
def test_btc_import_rejects_invalid_canonical_rows(
    tmp_path, updates, message
):
    from hcmai.data.ingestion import BTCIngestionConfig, import_btc_frame_store

    source = tmp_path / "btc"
    _write_btc_metadata(source, [_valid_btc_row(**updates)])

    with pytest.raises(ValueError, match=message):
        import_btc_frame_store(
            BTCIngestionConfig(
                btc_root=source,
                data_root=source,
                output_root=tmp_path / "frame_store",
                frame_store_id="btc-invalid-v1",
            )
        )


def test_btc_import_rejects_duplicate_frame_id(tmp_path):
    from hcmai.data.ingestion import BTCIngestionConfig, import_btc_frame_store

    source = tmp_path / "btc"
    _write_btc_metadata(
        source,
        [
            _valid_btc_row(),
            _valid_btc_row(frame_idx=121, timestamp_ms=4100),
        ],
    )

    with pytest.raises(ValueError, match="Duplicate frame_id"):
        import_btc_frame_store(
            BTCIngestionConfig(
                btc_root=source,
                data_root=source,
                output_root=tmp_path / "frame_store",
                frame_store_id="btc-duplicate-v1",
            )
        )


def test_btc_import_preserves_distinct_frames_at_one_submission_coordinate(
    tmp_path,
):
    from hcmai.data.ingestion import BTCIngestionConfig, import_btc_frame_store

    source = tmp_path / "btc"
    _write_btc_metadata(
        source,
        [
            _valid_btc_row(),
            _valid_btc_row(
                frame_id="L01_V001:0001",
                keyframe_order=2,
                timestamp_ms=4040,
                image_path="keyframes/L01_V001/0001.jpg",
            ),
        ],
    )

    output = import_btc_frame_store(
        BTCIngestionConfig(
            btc_root=source,
            data_root=source,
            output_root=tmp_path / "frame_store",
            frame_store_id="btc-collision-v1",
        )
    )

    frames = pd.read_parquet(output)
    assert frames["frame_id"].tolist() == ["L01_V001:0000", "L01_V001:0001"]
    assert frames["frame_idx"].tolist() == [120, 120]
    manifest = json.loads((output.parent / "manifest.json").read_text())
    assert manifest["duplicate_submission_coordinate_groups"] == 1
    assert manifest["duplicate_submission_coordinate_rows"] == 2
    assert manifest["maximum_submission_coordinate_multiplicity"] == 2


def test_btc_import_rejects_empty_metadata(tmp_path):
    from hcmai.data.ingestion import BTCIngestionConfig, import_btc_frame_store

    source = tmp_path / "btc"
    _write_btc_metadata(
        source,
        pd.DataFrame(columns=list(_valid_btc_row())),
    )

    with pytest.raises(ValueError, match="must contain at least one frame"):
        import_btc_frame_store(
            BTCIngestionConfig(
                btc_root=source,
                data_root=source,
                output_root=tmp_path / "frame_store",
                frame_store_id="btc-empty-v1",
            )
        )


def test_manifest_staging_failure_preserves_published_bundle(
    tmp_path, monkeypatch
):
    import hcmai.data.ingestion.btc as btc

    source = tmp_path / "btc"
    _write_btc_metadata(source, [_valid_btc_row()])
    output_root = tmp_path / "frame_store"
    output_root.mkdir()
    frames_path = output_root / "frames.parquet"
    manifest_path = output_root / "manifest.json"
    frames_path.write_bytes(b"previous frames")
    manifest_path.write_text('{"frame_store_id":"previous"}\n')

    def fail_manifest_write(*_args, **_kwargs):
        raise OSError("simulated manifest staging failure")

    monkeypatch.setattr(btc, "write_json", fail_manifest_write)

    with pytest.raises(OSError, match="manifest staging failure"):
        btc.import_btc_frame_store(
            btc.BTCIngestionConfig(
                btc_root=source,
                data_root=source,
                output_root=output_root,
                frame_store_id="btc-new-v1",
            )
        )

    assert frames_path.read_bytes() == b"previous frames"
    assert manifest_path.read_text() == '{"frame_store_id":"previous"}\n'
    assert not (output_root / ".frames.parquet.staged").exists()
    assert not (output_root / ".manifest.json.staged").exists()


def test_manifest_publish_failure_restores_previous_bundle(
    tmp_path, monkeypatch
):
    import hcmai.data.ingestion.btc as btc

    source = tmp_path / "btc"
    _write_btc_metadata(source, [_valid_btc_row()])
    output_root = tmp_path / "frame_store"
    output_root.mkdir()
    frames_path = output_root / "frames.parquet"
    manifest_path = output_root / "manifest.json"
    previous_frames = b"previous frames"
    previous_manifest = '{"frame_store_id":"previous"}\n'
    frames_path.write_bytes(previous_frames)
    manifest_path.write_text(previous_manifest)

    original_replace = btc.Path.replace

    def fail_final_manifest_replace(path, target):
        if (
            path.name == ".manifest.json.staged"
            and btc.Path(target) == manifest_path
        ):
            raise OSError("simulated final manifest replacement failure")
        return original_replace(path, target)

    monkeypatch.setattr(btc.Path, "replace", fail_final_manifest_replace)

    with pytest.raises(OSError, match="final manifest replacement failure"):
        btc.import_btc_frame_store(
            btc.BTCIngestionConfig(
                btc_root=source,
                data_root=source,
                output_root=output_root,
                frame_store_id="btc-new-v1",
            )
        )

    assert frames_path.read_bytes() == previous_frames
    assert manifest_path.read_text() == previous_manifest
    assert not (output_root / ".frames.parquet.staged").exists()
    assert not (output_root / ".manifest.json.staged").exists()
    assert not (output_root / ".frames.parquet.backup").exists()
    assert not (output_root / ".manifest.json.backup").exists()
