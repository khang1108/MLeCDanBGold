"""CLI validation tests for the YOLOE object detection entry point."""

from __future__ import annotations

import pytest

from scripts.detect_objects import parse_args


@pytest.mark.parametrize("option", ("--top-k", "--batch-size", "--limit"))
@pytest.mark.parametrize("value", ("0", "-1"))
def test_parse_args_rejects_non_positive_work_limits(option: str, value: str) -> None:
    """Reject values that would make slicing or batching silently do no work."""

    with pytest.raises(SystemExit) as error:
        parse_args([option, value])

    assert error.value.code == 2


@pytest.mark.parametrize("value", ("-0.01", "1.01", "nan", "inf", "-inf"))
def test_parse_args_rejects_invalid_confidence(value: str) -> None:
    """Reject confidence thresholds outside the finite YOLO unit interval."""

    with pytest.raises(SystemExit) as error:
        parse_args(["--min-confidence", value])

    assert error.value.code == 2


@pytest.mark.parametrize("value", ("0", "1"))
def test_parse_args_accepts_confidence_boundaries(value: str) -> None:
    """Allow both inclusive confidence boundaries supported by the CLI contract."""

    args = parse_args(["--min-confidence", value])

    assert args.min_confidence == float(value)


def test_parse_args_keeps_positive_defaults() -> None:
    """Keep safe defaults for all work-size arguments."""

    args = parse_args([])

    assert args.top_k == 30
    assert args.batch_size == 32
    assert args.limit is None
    assert args.min_confidence == 0.20
