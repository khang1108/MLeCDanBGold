"""Tests for the deterministic Filter text-normalization contract."""

import pytest

from hcmai.filtering.normalization import normalize_filter_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Cảnh  có ÁO đỏ ", "canh co ao do"),
        ("ĐƯỜNG phố", "duong pho"),
        ("person\n\tcar", "person car"),
        ("", ""),
    ],
)
def test_normalize_filter_text_matches_frontend_contract(
    raw: str,
    expected: str,
) -> None:
    """Keep backend matching identical to the frontend request serializer."""

    assert normalize_filter_text(raw) == expected

