"""Smoke tests for caption, OCR, and ASR artifact stores."""

from pathlib import Path

import pandas as pd
import pytest

from hcmai.corpus.stores.evidence import ASRStore, CaptionStore, OCRStore
from hcmai.corpus.stores import evidence
from hcmai.retrieval.models import RetrievalSource
from offline.enrichment.caption.models import CaptionEvidence
from offline.enrichment.models import FrameEnrichment
from offline.enrichment.ocr.models import OCREvidence


@pytest.fixture
def evidence_paths(tmp_path: Path) -> dict[RetrievalSource, Path]:
    caption_path = tmp_path / "captions.parquet"
    caption_rows = [
        CaptionEvidence(
            frame_id="L01_V001_00000010",
            video_id="L01_V001",
            frame_idx=10,
            timestamp_ms=1_000,
            text="A cook holds a pan.",
            artifact_version="caption-v1",
            model_name="fixture",
        ),
        CaptionEvidence(
            frame_id="L01_V001_00000020",
            video_id="L01_V001",
            frame_idx=20,
            timestamp_ms=2_000,
            text="A failed frame.",
            artifact_version="caption-v1",
            model_name="fixture",
            status="failed",
            error_code="BackendError",
            error_message="backend error",
        ),
    ]
    pd.DataFrame(
        [row.model_dump(mode="json") for row in caption_rows]
    ).to_parquet(caption_path, index=False)

    ocr_path = tmp_path / "ocr.parquet"
    ocr_rows = [
        OCREvidence(
            frame_id="L01_V001_00000010",
            video_id="L01_V001",
            frame_idx=10,
            timestamp_ms=1_000,
            raw_text="BƠ",
            normalized_text="BƠ",
            quality_score=1.0,
            region_count=1,
            artifact_version="ocr-v1",
            model_name="fixture",
        ),
        OCREvidence(
            frame_id="L01_V001_00000020",
            video_id="L01_V001",
            frame_idx=20,
            timestamp_ms=2_000,
            artifact_version="ocr-v1",
            model_name="fixture",
            status="failed",
            error_code="BackendError",
            error_message="backend error",
        ),
    ]
    pd.DataFrame([row.model_dump(mode="json") for row in ocr_rows]).to_parquet(
        ocr_path, index=False
    )

    asr_path = tmp_path / "frame_enrichment.parquet"
    pd.DataFrame(
        [
            {
                "frame_id": "L01_V001_00000010",
                "asr_text": "Cho bơ vào chảo.",
                "model_name": "fixture",
                "status": "completed",
            },
            {
                "frame_id": "L01_V001_00000020",
                "asr_text": float("nan"),
                "model_name": "fixture",
                "status": "failed",
                "error_message": "backend error",
            },
        ]
    ).to_parquet(asr_path, index=False)
    return {
        RetrievalSource.CAPTION: caption_path,
        RetrievalSource.OCR: ocr_path,
        RetrievalSource.ASR: asr_path,
    }


@pytest.mark.parametrize(
    ("store_type", "source", "contract", "expected"),
    [
        (
            CaptionStore,
            RetrievalSource.CAPTION,
            CaptionEvidence,
            "A cook holds a pan.",
        ),
        (OCRStore, RetrievalSource.OCR, OCREvidence, "BƠ"),
        (ASRStore, RetrievalSource.ASR, None, "Cho bơ vào chảo."),
    ],
)
def test_source_stores_return_validated_text(
    evidence_paths: dict[RetrievalSource, Path],
    store_type: type[CaptionStore | OCRStore | ASRStore],
    source: RetrievalSource,
    contract: type[CaptionEvidence | OCREvidence] | None,
    expected: str,
) -> None:
    store = store_type(evidence_paths[source])

    assert store.source == source
    record = store.get("L01_V001_00000010")
    if contract is not None:
        # Runtime readers validate at their own private boundary and must not
        # expose offline producer contracts through corpus APIs.
        assert not isinstance(record, contract)
        assert record.frame_id == "L01_V001_00000010"
    else:
        assert record.frame_id == "L01_V001_00000010"
        assert not isinstance(record, FrameEnrichment)
    assert store.get_text("L01_V001_00000010") == expected
    assert store.get_text("L01_V001_00000020") is None


def test_store_reads_once_and_preserves_requested_order(
    evidence_paths: dict[RetrievalSource, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = evidence.pd.read_parquet
    calls = 0

    def read_once(path: Path) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(evidence.pd, "read_parquet", read_once)
    store = CaptionStore(evidence_paths[RetrievalSource.CAPTION])
    requested = ["L01_V001_00000020", "L01_V001_00000010"] * 2

    assert [row.frame_id for row in store.get_many(requested)] == requested
    assert list(store.iter_records())
    assert calls == 1


def test_store_rejects_duplicate_frame_ids(
    evidence_paths: dict[RetrievalSource, Path],
) -> None:
    path = evidence_paths[RetrievalSource.CAPTION]
    table = pd.read_parquet(path)
    pd.concat([table, table.iloc[:1]]).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="Duplicate frame_id"):
        CaptionStore(path)


def test_store_requires_its_source_column(tmp_path: Path) -> None:
    path = tmp_path / "missing.parquet"
    pd.DataFrame([{"frame_id": "frame-1", "model_name": "fixture"}]).to_parquet(
        path, index=False
    )

    with pytest.raises(ValueError, match="asr_text"):
        ASRStore(path)
