"""Smoke tests for caption, OCR, and ASR artifact stores."""

from pathlib import Path

import pandas as pd
import pytest

from hcmai.common.schemas import FrameEnrichment, RetrievalSource
from hcmai.data import ASRStore, CaptionStore, OCRStore
from hcmai.data import evidence


@pytest.fixture
def enrichment_path(tmp_path: Path) -> Path:
    rows = [
        {
            "frame_id": "L01_V001_00000010",
            "caption": "A cook holds a pan.",
            "ocr_text": "BƠ",
            "asr_text": "Cho bơ vào chảo.",
            "model_name": "fixture",
            "status": "completed",
        },
        {
            "frame_id": "L01_V001_00000020",
            "caption": "A failed frame.",
            "ocr_text": None,
            "asr_text": float("nan"),
            "model_name": "fixture",
            "status": "failed",
            "error_message": "backend error",
        },
    ]
    path = tmp_path / "frame_enrichment.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


@pytest.mark.parametrize(
    ("store_type", "source", "expected"),
    [
        (CaptionStore, RetrievalSource.CAPTION, "A cook holds a pan."),
        (OCRStore, RetrievalSource.OCR, "BƠ"),
        (ASRStore, RetrievalSource.ASR, "Cho bơ vào chảo."),
    ],
)
def test_source_stores_return_validated_text(
    enrichment_path: Path,
    store_type: type[CaptionStore | OCRStore | ASRStore],
    source: RetrievalSource,
    expected: str,
) -> None:
    store = store_type(enrichment_path)

    assert store.source == source
    assert isinstance(store.get("L01_V001_00000010"), FrameEnrichment)
    assert store.get_text("L01_V001_00000010") == expected
    assert store.get("L01_V001_00000020").asr_text is None
    assert store.get_text("L01_V001_00000020") is None


def test_store_reads_once_and_preserves_requested_order(
    enrichment_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = evidence.pd.read_parquet
    calls = 0

    def read_once(path: Path) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(evidence.pd, "read_parquet", read_once)
    store = CaptionStore(enrichment_path)
    requested = ["L01_V001_00000020", "L01_V001_00000010"] * 2

    assert [row.frame_id for row in store.get_many(requested)] == requested
    assert list(store.iter_records())
    assert calls == 1


def test_store_rejects_duplicate_frame_ids(enrichment_path: Path) -> None:
    table = pd.read_parquet(enrichment_path)
    pd.concat([table, table.iloc[:1]]).to_parquet(enrichment_path, index=False)

    with pytest.raises(ValueError, match="Duplicate frame_id"):
        CaptionStore(enrichment_path)


def test_store_requires_its_source_column(tmp_path: Path) -> None:
    path = tmp_path / "missing.parquet"
    pd.DataFrame([{"frame_id": "frame-1", "model_name": "fixture"}]).to_parquet(
        path, index=False
    )

    with pytest.raises(ValueError, match="asr_text"):
        ASRStore(path)
