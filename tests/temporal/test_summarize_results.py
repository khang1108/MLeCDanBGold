"""Unit tests for motion graph result summarizer and LaTeX table generator (SP-15)."""

import csv
import json
from pathlib import Path

import pytest

from scripts.evaluation.summarize_motion_graph_results import (
    build_efficiency_table_rows,
    build_main_table_rows,
    format_csv,
    format_float,
    format_latex_efficiency_table,
    format_latex_main_table,
    format_pct,
    load_result_json,
)


def test_load_result_json(tmp_path: Path):
    assert load_result_json(tmp_path / "non_existent.json") is None

    test_file = tmp_path / "data.json"
    test_file.write_text('{"key": 123}', encoding="utf-8")
    assert load_result_json(test_file) == {"key": 123}


def test_format_helpers():
    assert format_pct(0.8524) == "85.2"
    assert format_pct(None) == "--"
    assert format_float(0.12345) == "0.123"
    assert format_float(None) == "--"


def test_format_csv():
    headers = ["Col1", "Col2"]
    rows = [["a", "1"], ["b", "2"]]
    csv_str = format_csv(headers, rows)

    reader = list(csv.reader(csv_str.strip().splitlines()))
    assert reader[0] == ["Col1", "Col2"]
    assert reader[1] == ["a", "1"]
    assert reader[2] == ["b", "2"]


def test_build_main_table_empty(tmp_path: Path):
    rows = build_main_table_rows(tmp_path)
    assert len(rows) == 5
    assert rows[0]["method"] == "B1 Independent"
    assert rows[1]["method"] == "B2 Static DP"
    assert rows[4]["method"] == "Ours (Motion Graph)"
    assert rows[1]["r1"] is None


def test_build_main_table_with_mock_results(tmp_path: Path):
    # Mock result files
    b2_data = {"method_name": "B2 Static DP", "num_queries": 10, "r1": 0.50, "r5": 0.70, "mrr": 0.60, "event_hit": 0.75, "all_hit": 0.60}
    b3_data = {"method_name": "B3 Top-K", "num_queries": 10, "r1": 0.55, "r5": 0.75, "mrr": 0.65, "event_hit": 0.80, "all_hit": 0.65}
    b4_data = {"method_name": "B4 Visual", "num_queries": 10, "r1": 0.58, "r5": 0.78, "mrr": 0.68, "event_hit": 0.82, "all_hit": 0.68}
    ours_data = {"method_name": "Ours", "num_queries": 10, "r1": 0.72, "r5": 0.88, "mrr": 0.79, "event_hit": 0.90, "all_hit": 0.82}
    rev_data = {"experiment": "reverse_counterfactual", "reverse_accuracy": 0.85}

    (tmp_path / "baseline_dp.json").write_text(json.dumps(b2_data), encoding="utf-8")
    (tmp_path / "topk_no_edges.json").write_text(json.dumps(b3_data), encoding="utf-8")
    (tmp_path / "visual_continuity.json").write_text(json.dumps(b4_data), encoding="utf-8")
    (tmp_path / "transition_delta.json").write_text(json.dumps(ours_data), encoding="utf-8")
    (tmp_path / "reverse.json").write_text(json.dumps(rev_data), encoding="utf-8")

    rows = build_main_table_rows(tmp_path)
    assert len(rows) == 5

    # Check B2
    assert rows[1]["r1"] == 0.50
    assert rows[1]["all_hit"] == 0.60

    # Check Ours
    ours_row = rows[4]
    assert ours_row["r1"] == 0.72
    assert ours_row["reverse_acc"] == 0.85

    # Check LaTeX formatting
    latex = format_latex_main_table(rows)
    assert r"\begin{table}" in latex
    assert r"\label{tab:main_ablation}" in latex
    assert r"\textbf{Ours (Motion Graph)}" in latex
    assert "85.0" in latex  # Reverse acc for Ours


def test_efficiency_table(tmp_path: Path):
    topk_data = {"candidate_k": 32, "transition_weight": 0.0, "all_hit": 0.65, "avg_latency_ms": 12.5}
    ours_data = {"candidate_k": 32, "transition_weight": 0.25, "all_hit": 0.82, "avg_latency_ms": 14.8}

    (tmp_path / "topk_no_edges.json").write_text(json.dumps(topk_data), encoding="utf-8")
    (tmp_path / "transition_delta.json").write_text(json.dumps(ours_data), encoding="utf-8")

    rows = build_efficiency_table_rows(tmp_path, candidate_recalls={32: 0.915})
    assert len(rows) == 2
    assert rows[0]["k"] == 32
    assert rows[0]["beta"] == 0.0
    assert rows[1]["beta"] == 0.25

    latex = format_latex_efficiency_table(rows)
    assert r"\label{tab:candidate_efficiency}" in latex
    assert "91.5" in latex
    assert "14.8" in latex
