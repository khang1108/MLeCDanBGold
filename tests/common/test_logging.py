"""Tests for the shared logging configuration."""

from __future__ import annotations

import logging

from hcmai.common.utils.logging import configure_logging, get_logger


def test_console_is_colored_and_contains_call_site(capsys) -> None:
    configure_logging("DEBUG", force=True)

    get_logger("hcmai.test").warning("Index is unavailable")

    output = capsys.readouterr().err
    assert "\033[33mWARNING\033[0m" in output
    assert "hcmai.test" in output
    assert "test_logging.py:" in output
    assert "(test_console_is_colored_and_contains_call_site" in output
    assert "Index is unavailable" in output


def test_file_log_has_details_without_ansi_codes(tmp_path) -> None:
    log_path = tmp_path / "pipeline.log"
    configure_logging(logging.INFO, log_file=log_path, force=True)

    get_logger("hcmai.pipeline").error("Retrieval failed")

    output = log_path.read_text(encoding="utf-8")
    assert "\033[" not in output
    assert "ERROR" in output
    assert "hcmai.pipeline" in output
    assert "test_logging.py:" in output
    assert "(test_file_log_has_details_without_ansi_codes)" in output
    assert "Retrieval failed" in output
