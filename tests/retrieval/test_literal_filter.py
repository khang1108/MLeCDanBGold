"""Focused tests for direct literal matching over runtime evidence."""

from __future__ import annotations

import pytest

from hcmai.api.contracts import FilterRequest
from hcmai.corpus.models import Frame, TranscriptSegment
from hcmai.orchestration.pipeline import SearchService
from hcmai.retrieval.evidence.literal import LiteralTextIndex
from hcmai.retrieval.models import RetrievalSource


class _Corpus:
    """Provide a tiny hand-checkable corpus without loading model artifacts."""

    def __init__(self) -> None:
        self.records = (
            Frame("L21_V001_f0", "L21_V001", 0, 0, "0.jpg", fps=25),
            Frame("L21_V001_f1", "L21_V001", 3, 100, "1.jpg", fps=25),
            Frame("L21_V001_f2", "L21_V001", 5, 200, "2.jpg", fps=25),
            Frame("L22_V002_f0", "L22_V002", 0, 0, "3.jpg", fps=30),
        )
        self.captions = {
            "L21_V001_f0": "Người mặc áo đỏ",
            "L22_V002_f0": "A blue bicycle",
        }
        self.ocrs = {"L21_V001_f2": "ÁO ĐỎ - BIỂN BÁO"}
        self.counts = {"L22_V002_f0": {"person": 2, "bicycle": 1}}

    def iter_frames(self):
        """Return frames in canonical artifact order."""

        return iter(self.records)

    def has_titles(self):
        """Expose the title source."""

        return True

    def title(self, video_id):
        """Return one shared title for the first video."""

        return "Bản tin sáng" if video_id == "L21_V001" else None

    def has_evidence(self, source):
        """Expose all three text evidence stores."""

        return source in {
            RetrievalSource.CAPTION,
            RetrievalSource.OCR,
            RetrievalSource.ASR,
        }

    def caption(self, frame_id):
        """Return optional caption text."""

        return self.captions.get(frame_id)

    def ocr(self, frame_id):
        """Return optional OCR text."""

        return self.ocrs.get(frame_id)

    def transcript_segments_for_video(self, video_id):
        """Return one half-open ASR segment for the first video."""

        if video_id != "L21_V001":
            return ()
        return (
            TranscriptSegment("s1", video_id, 0, 100, 200, "Xin chào lớp học"),
        )

    def has_object_counts(self):
        """Expose object counts."""

        return True

    def object_counts(self, frame_id):
        """Return object multiplicity for one frame."""

        return self.counts.get(frame_id, {})


def test_literal_filter_ands_populated_text_sources_and_preserves_order() -> None:
    """Require every source predicate while retaining raw evidence and identity."""

    index = LiteralTextIndex(_Corpus())
    total, hits = index.search(
        text_filters={"title": "ban tin", "caption": "AO do"},
        object_filters={},
        folder_id=None,
        video_id=None,
        page_id=1,
        page_size=20,
    )

    assert total == 1
    assert [frame.frame_id for frame, _, _ in hits] == ["L21_V001_f0"]
    assert hits[0][2] == {
        "title": "Bản tin sáng",
        "caption": "Người mặc áo đỏ",
    }


def test_literal_filter_projects_asr_by_half_open_segment_time() -> None:
    """Attach ASR at its start boundary but not at its end boundary."""

    _, hits = LiteralTextIndex(_Corpus()).search(
        text_filters={"title": "ban tin", "asr": "xin chao"},
        object_filters={},
        folder_id=None,
        video_id=None,
        page_id=1,
        page_size=20,
    )

    assert [frame.frame_id for frame, _, _ in hits] == ["L21_V001_f1"]
    assert hits[0][2] == {
        "title": "Bản tin sáng",
        "asr": "Xin chào lớp học",
    }


@pytest.mark.parametrize(
    ("text_filters", "frame_id", "source"),
    [
        ({"title": "ban tin"}, "L21_V001_f0", "title"),
        ({"caption": "blue bicycle"}, "L22_V002_f0", "caption"),
        ({"ocr": "bien bao"}, "L21_V001_f2", "ocr"),
        ({"asr": "xin chao"}, "L21_V001_f1", "asr"),
    ],
)
def test_literal_filter_searches_each_available_text_source(
    text_filters: dict[str, str],
    frame_id: str,
    source: str,
) -> None:
    """Apply each configured text matcher to its corresponding source only."""

    _, hits = LiteralTextIndex(_Corpus()).search(
        text_filters=text_filters,
        object_filters={},
        folder_id=None,
        video_id=None,
        page_id=1,
        page_size=20,
    )

    assert hits[0][0].frame_id == frame_id
    assert source in hits[0][2]


def test_literal_filter_requires_exact_object_name_and_minimum_count() -> None:
    """Keep object names exact while accepting counts at or above the threshold."""

    index = LiteralTextIndex(_Corpus())
    total, hits = index.search(
        text_filters={},
        object_filters={"person": 2, "bicycle": 1},
        folder_id=None,
        video_id=None,
        page_id=1,
        page_size=20,
    )
    insufficient_total, _ = index.search(
        text_filters={},
        object_filters={"person": 3},
        folder_id=None,
        video_id=None,
        page_id=1,
        page_size=20,
    )
    wrong_label_total, _ = index.search(
        text_filters={},
        object_filters={"people": 1},
        folder_id=None,
        video_id=None,
        page_id=1,
        page_size=20,
    )

    assert total == 1
    assert hits[0][0].frame_id == "L22_V002_f0"
    assert hits[0][2] == {"objects": "bicycle: 1, person: 2"}
    assert insufficient_total == 0
    assert wrong_label_total == 0


def test_literal_filter_combines_backend_scopes_and_paginates() -> None:
    """Apply backend-only scopes with AND before slicing a result page."""

    index = LiteralTextIndex(_Corpus())
    total, hits = index.search(
        text_filters={"caption": "ao do"},
        object_filters={},
        folder_id="L21",
        video_id="L21_V001",
        page_id=2,
        page_size=1,
    )
    empty_total, _ = index.search(
        text_filters={"caption": "ao do"},
        object_filters={},
        folder_id="L21",
        video_id="L22_V002",
        page_id=1,
        page_size=20,
    )

    assert total == 1
    assert hits == []
    assert empty_total == 0


def test_literal_filter_returns_all_frames_without_evidence_predicates() -> None:
    """Allow browsing canonical frames when no evidence field is populated."""

    total, hits = LiteralTextIndex(_Corpus()).search(
        text_filters={},
        object_filters={},
        folder_id=None,
        video_id=None,
        page_id=1,
        page_size=20,
    )

    assert total == 4
    assert [frame.frame_id for frame, _, _ in hits] == [
        "L21_V001_f0",
        "L21_V001_f1",
        "L21_V001_f2",
        "L22_V002_f0",
    ]


def test_filter_service_applies_scope_pagination_and_complete_metadata() -> None:
    """Materialize one scoped page without inventing ranking scores."""

    corpus = _Corpus()
    service = SearchService(
        corpus=corpus,
        retrieval=None,
        literal_text=LiteralTextIndex(corpus),
    )

    response = service.filter_frames(FilterRequest(
        metadata_filters={"objects": {"person": 2}},
        folder_id="L22",
        frames_per_pages=20,
        page_id=1,
    ))

    assert response.total_results == 1
    assert response.total_pages == 1
    assert response.available_sources == ["title", "caption", "ocr", "asr", "objects"]
    assert response.results[0].model_dump() == {
        "frame_id": "L22_V002_f0",
        "video_id": "L22_V002",
        "frame_idx": 0,
        "timestamp_ms": 0,
        "fps": 30.0,
        "folder_id": "L22",
        "title": None,
        "caption": "A blue bicycle",
        "ocr": None,
        "objects": {"person": 2, "bicycle": 1},
        "asr": None,
        "matches": {"objects": "bicycle: 1, person: 2"},
    }
