"""Row selection and merge-back must not shift a caption onto the wrong frame."""

from __future__ import annotations

import pyarrow as pa
import pytest

from scripts.translate import merge, pending, rewrite


def _table() -> pa.Table:
    return pa.table(
        {
            "status": pa.array(["completed", "failed", "completed", "completed"]),
            "frame_id": pa.array(["a", "b", "c", "d"]),
            "text": pa.array(["a dog", "", "a cat", ""]),
            "artifact_version": pa.array(["caption_qwen_vl_v1"] * 4),
        }
    )


def test_pending_skips_failed_and_empty() -> None:
    assert pending(_table()) == [0, 2]


def test_merge_keeps_untranslated_rows_in_place() -> None:
    texts = ["a dog", "", "a cat", ""]

    assert merge(texts, [0, 2], ["một con chó", "một con mèo"]) == [
        "một con chó",
        "",
        "một con mèo",
        "",
    ]


def test_merge_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        merge(["a dog", "a cat"], [0, 1], ["một con chó"])


def test_rewrite_preserves_schema_and_bumps_version() -> None:
    table = _table()
    out = rewrite(table, ["một con chó", "", "một con mèo", ""])

    assert out.schema == table.schema
    assert out.column("text").to_pylist()[0] == "một con chó"
    assert set(out.column("artifact_version").to_pylist()) == {"caption_qwen_vl_vi_v1"}
