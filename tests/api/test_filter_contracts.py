"""Tests for strict public Filter request and response contracts."""

import pytest

from pydantic import ValidationError

from hcmai.api.contracts import FilterRequest, FilterResponse


def test_filter_request_defaults_and_keeps_frontend_page_field() -> None:
    """Retain the merged frontend's intentionally plural page-size field."""

    request = FilterRequest.model_validate({})

    assert request.frames_per_pages == 12
    assert request.page_id == 1
    assert request.folder_id is None
    assert request.video_id is None
    assert request.metadata_filters.objects == {}


@pytest.mark.parametrize("value", [0, 49])
def test_filter_request_rejects_page_sizes_outside_v1_bound(value: int) -> None:
    """Bound each page so one request cannot expand Filter memory arbitrarily."""

    with pytest.raises(ValidationError):
        FilterRequest(frames_per_pages=value)


def test_filter_request_rejects_page_zero() -> None:
    """Use one-based page coordinates consistently with the UI."""

    with pytest.raises(ValidationError):
        FilterRequest(page_id=0)


def test_filter_request_normalizes_text_and_object_labels() -> None:
    """Normalize every textual predicate again at the trusted API boundary."""

    request = FilterRequest(
        metadata_filters={
            "title": "  Cảnh ÁO Đỏ ",
            "objects": {" Người ": 3},
        }
    )

    assert request.metadata_filters.title == "canh ao do"
    assert request.metadata_filters.objects == {"nguoi": 3}


@pytest.mark.parametrize(
    "metadata_filters",
    [
        {"objects": {"": 1}},
        {"objects": {"person": -1}},
        {"objects": {"person": 1.5}},
        {"objects": {"Người": 1, "nguoi": 2}},
        {"title": "x" * 501},
    ],
)
def test_filter_request_rejects_ambiguous_or_unbounded_filters(
    metadata_filters: dict[str, object],
) -> None:
    """Reject inputs that would change exact matching or resource bounds."""

    with pytest.raises(ValidationError):
        FilterRequest(metadata_filters=metadata_filters)


def test_filter_request_rejects_unknown_fields() -> None:
    """Prevent silent contract drift between Filter clients and the backend."""

    with pytest.raises(ValidationError):
        FilterRequest(unexpected=True)


def test_filter_response_preserves_complete_canonical_metadata() -> None:
    """Return enough display metadata to avoid per-frame detail requests."""

    response = FilterResponse.model_validate(
        {
            "page_id": 1,
            "frames_per_pages": 12,
            "total_results": 1,
            "total_pages": 1,
            "results": [
                {
                    "frame_id": "L21_V001_000001",
                    "video_id": "L21_V001",
                    "frame_idx": 25,
                    "timestamp_ms": 1000,
                    "folder_id": "L21",
                    "title": "Title",
                    "caption": "Caption",
                    "ocr": "OCR",
                    "objects": {"person": 3},
                    "asr": "Speech",
                }
            ],
        }
    )

    assert response.results[0].frame_id == "L21_V001_000001"
    assert response.results[0].frame_idx == 25
    assert response.results[0].objects == {"person": 3}
