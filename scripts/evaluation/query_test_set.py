"""Compile competition query files and verified labels into one test-set artifact.

The compiler owns artifact validation, portable serialization, and deep schema
verification. It does not run retrieval or expose labels to a request payload.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from hcmai.temporal.planner import plan_query_events, split_query_events


TEST_SET_SCHEMA_VERSION = "hcmai-query-test-set-v1"
TEST_SET_SCHEMA_VERSION_V2 = "hcmai-query-test-set-v2"
SUPPORTED_SCHEMA_VERSIONS = {TEST_SET_SCHEMA_VERSION, TEST_SET_SCHEMA_VERSION_V2}
ALLOWED_TOP_LEVEL_KEYS = {
    "schema_version",
    "split",
    "ground_truth_source",
    "qa_mode",
    "source_sha256",
    "cases",
}
ALLOWED_CASE_KEYS = {
    "query_id",
    "kind",
    "source_query_file",
    "raw_query",
    "retrieval_query",
    "events",
    "answer",
    "ground_truth",
}
ALLOWED_KINDS = {"kis", "qa", "trake"}
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_TRAKE_EVENT = re.compile(r"^(?:E\d+|Cảnh\s*\d+)\s*:?\s*(.+)$", re.IGNORECASE)


def validate_query_test_set(value: Any) -> None:
    """Deeply validate a query test-set dictionary against the v1 or v2 schema."""

    if not isinstance(value, dict):
        raise ValueError("query test set must be a JSON object")

    schema_version = value.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported query test-set schema: {schema_version!r}"
        )

    keys = set(value.keys())
    if keys != ALLOWED_TOP_LEVEL_KEYS:
        missing = ALLOWED_TOP_LEVEL_KEYS - keys
        extra = keys - ALLOWED_TOP_LEVEL_KEYS
        parts: list[str] = []
        if missing:
            parts.append(f"missing {sorted(missing)}")
        if extra:
            parts.append(f"unexpected {sorted(extra)}")
        raise ValueError(f"invalid test-set keys: {', '.join(parts)}")

    if not isinstance(value["split"], str) or not value["split"].strip():
        raise ValueError("split must be a non-empty string")

    if not isinstance(value["ground_truth_source"], str) or not value["ground_truth_source"].strip():
        raise ValueError("ground_truth_source must be a non-empty string")

    if value["qa_mode"] != "retrieval_only":
        raise ValueError(f"unsupported qa_mode: {value['qa_mode']!r}, expected 'retrieval_only'")

    if not isinstance(value["source_sha256"], str) or not _HEX_64.match(value["source_sha256"]):
        raise ValueError(f"source_sha256 must be a 64-character lowercase hex string: {value['source_sha256']!r}")

    cases = value["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list")

    seen_ids: set[str] = set()
    seen_files: set[str] = set()

    for idx, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case at index {idx} must be a JSON object")

        case_keys = set(case.keys())
        if case_keys != ALLOWED_CASE_KEYS:
            missing = ALLOWED_CASE_KEYS - case_keys
            extra = case_keys - ALLOWED_CASE_KEYS
            parts = []
            if missing:
                parts.append(f"missing {sorted(missing)}")
            if extra:
                parts.append(f"unexpected {sorted(extra)}")
            raise ValueError(f"invalid case keys at index {idx}: {', '.join(parts)}")

        query_id = case["query_id"]
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(f"case at index {idx}: query_id must be a non-empty string")
        if query_id in seen_ids:
            raise ValueError(f"duplicate query_id: {query_id!r}")
        seen_ids.add(query_id)

        kind = case["kind"]
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"{query_id}: unsupported kind: {kind!r}")

        source_file = case["source_query_file"]
        if not isinstance(source_file, str) or not source_file.strip():
            raise ValueError(f"{query_id}: source_query_file must be a non-empty string")
        if source_file in seen_files:
            raise ValueError(f"duplicate source_query_file: {source_file!r}")
        seen_files.add(source_file)

        raw_query = case["raw_query"]
        if not isinstance(raw_query, str) or not raw_query.strip():
            raise ValueError(f"{query_id}: raw_query must be a non-empty string")

        retrieval_query = case["retrieval_query"]
        if not isinstance(retrieval_query, str) or not retrieval_query.strip():
            raise ValueError(f"{query_id}: retrieval_query must be a non-empty string")

        events = case["events"]
        if not isinstance(events, list) or not events:
            raise ValueError(f"{query_id}: events must be a non-empty list of strings")
        for event_idx, event in enumerate(events):
            if not isinstance(event, str) or not event.strip():
                raise ValueError(f"{query_id}: event {event_idx} must be a non-empty string")

        answer = case["answer"]
        if kind == "qa":
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"{query_id}: QA case must have a non-empty string answer")
            expected_retrieval = " ".join(events)
            if retrieval_query != expected_retrieval:
                raise ValueError(f"{query_id}: QA retrieval_query must equal joined events")
            if raw_query.strip() == retrieval_query.strip():
                raise ValueError(f"{query_id}: QA retrieval_query must not equal raw_query")
        else:
            if answer is not None:
                raise ValueError(f"{query_id}: non-QA case must have null answer, got {answer!r}")
            if kind == "kis" and retrieval_query != raw_query:
                raise ValueError(f"{query_id}: KIS retrieval_query must equal raw_query")
            elif kind == "trake" and retrieval_query != " ".join(events):
                raise ValueError(f"{query_id}: TRAKE retrieval_query must equal joined events")

        gt = case["ground_truth"]
        if not isinstance(gt, list) or not gt:
            raise ValueError(f"{query_id}: ground_truth must be a non-empty list")

        seen_gt: set[tuple[str, Any]] = set()
        for gt_idx, entry in enumerate(gt):
            if not isinstance(entry, dict):
                raise ValueError(f"{query_id}: ground_truth entry {gt_idx} must be a JSON object")

            vid = entry.get("video_id")
            if not isinstance(vid, str) or not vid.strip():
                raise ValueError(f"{query_id}: ground_truth entry {gt_idx} video_id must be a non-empty string")

            if schema_version == TEST_SET_SCHEMA_VERSION:
                if set(entry.keys()) != {"video_id", "frame_idxs"}:
                    raise ValueError(f"{query_id}: ground_truth entry {gt_idx} must have video_id and frame_idxs")
                frame_idxs = entry["frame_idxs"]
                if not isinstance(frame_idxs, list) or not frame_idxs:
                    raise ValueError(f"{query_id}: ground_truth entry {gt_idx} frame_idxs must be a non-empty list")
                for f_idx in frame_idxs:
                    if isinstance(f_idx, bool) or not isinstance(f_idx, int) or f_idx < 0:
                        raise ValueError(f"{query_id}: ground_truth entry {gt_idx} frame_idx must be non-negative integer, got {f_idx!r}")

                if kind in ("kis", "qa") and len(frame_idxs) != 1:
                    raise ValueError(f"{query_id}: {kind.upper()} ground_truth entry must contain exactly 1 frame_idx, got {len(frame_idxs)}")
                elif kind == "trake":
                    if len(frame_idxs) != len(events):
                        raise ValueError(
                            f"{query_id}: TRAKE ground_truth entry frame count ({len(frame_idxs)}) "
                            f"must match event count ({len(events)})"
                        )
                    if tuple(sorted(frame_idxs)) != tuple(frame_idxs):
                        raise ValueError(f"{query_id}: TRAKE ground_truth entry frame_idxs must be chronological")

                gt_key = (vid, tuple(frame_idxs))
                if gt_key in seen_gt:
                    raise ValueError(f"{query_id}: duplicate ground_truth entry at index {gt_idx}")
                seen_gt.add(gt_key)
            else:
                # Schema v2: window-based ground truth
                allowed_v2_keys = {
                    "video_id",
                    "time_windows_ms",
                    "frame_windows",
                    "representative_frame_idx",
                    "frame_idxs",
                    "description",
                    "event_windows",
                }
                extra_keys = set(entry.keys()) - allowed_v2_keys
                if extra_keys:
                    raise ValueError(f"{query_id}: unexpected ground_truth entry keys in v2: {sorted(extra_keys)}")

                if "time_windows_ms" in entry:
                    tw = entry["time_windows_ms"]
                    if not isinstance(tw, list) or not tw:
                        raise ValueError(f"{query_id}: time_windows_ms must be a non-empty list")
                    for w in tw:
                        if not isinstance(w, (list, tuple)) or len(w) != 2 or w[0] < 0 or w[1] < w[0]:
                            raise ValueError(f"{query_id}: invalid time window {w!r}")

                if "frame_windows" in entry:
                    fw = entry["frame_windows"]
                    if not isinstance(fw, list) or not fw:
                        raise ValueError(f"{query_id}: frame_windows must be a non-empty list")
                    for w in fw:
                        if not isinstance(w, (list, tuple)) or len(w) != 2 or w[0] < 0 or w[1] < w[0]:
                            raise ValueError(f"{query_id}: invalid frame window {w!r}")

                if "representative_frame_idx" in entry:
                    rf = entry["representative_frame_idx"]
                    if isinstance(rf, bool) or not isinstance(rf, int) or rf < 0:
                        raise ValueError(f"{query_id}: invalid representative_frame_idx: {rf!r}")

                if "event_windows" in entry:
                    ew = entry["event_windows"]
                    if not isinstance(ew, list) or len(ew) != len(events):
                        raise ValueError(f"{query_id}: event_windows count must match events count")

                gt_key = (vid, str(entry.get("frame_windows")), str(entry.get("time_windows_ms")), str(entry.get("frame_idxs")))
                if gt_key in seen_gt:
                    raise ValueError(f"{query_id}: duplicate ground_truth entry at index {gt_idx}")
                seen_gt.add(gt_key)


def build_query_test_set(
    query_root: Path,
    split: str,
    *,
    ground_truth_source: str,
) -> dict[str, Any]:
    """Build a deterministic retrieval test set from one artifact split."""

    split_root = query_root / split
    for label_path in sorted((split_root / "ground_truth").glob("*.csv")):
        query_path = split_root / f"{label_path.stem}.txt"
        if not query_path.is_file():
            raise ValueError(f"missing query: {query_path}")

    cases: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    for query_path in sorted(split_root.glob("*.txt"), key=_natural_path_key):
        kind = _kind_from_path(query_path)
        label_path = split_root / "ground_truth" / f"{query_path.stem}.csv"
        if not label_path.is_file():
            raise ValueError(f"missing ground truth: {label_path}")
        source_paths.extend((query_path, label_path))
        raw_query = query_path.read_text(encoding="utf-8-sig").strip()
        labels = _read_rows(label_path)
        events = (
            _parse_trake_events(raw_query)
            if kind == "trake"
            else [
                event.rstrip(".!?").rstrip()
                for event in plan_query_events(raw_query)
            ]
        )
        if kind == "qa":
            parts = split_query_events(raw_query)
            final_part = parts[-1].rstrip(".!?").rstrip()
            if len(parts) < 2 or any(final_part in event for event in events):
                raise ValueError(
                    f"{query_path}: QA query must contain scene text before the question"
                )
        _validate_source_rows(label_path, kind, len(events), labels)
        cases.append(
            {
                "query_id": query_path.stem,
                "kind": kind,
                "source_query_file": f"{split}/{query_path.name}",
                "raw_query": raw_query,
                "retrieval_query": raw_query if kind == "kis" else " ".join(events),
                "events": events,
                "answer": labels[0][2] if kind == "qa" else None,
                "ground_truth": [
                    {
                        "video_id": row[0],
                        "frame_idxs": [
                            int(value)
                            for value in (row[1:] if kind == "trake" else row[1:2])
                        ],
                    }
                    for row in labels
                ],
            }
        )

    result = {
        "schema_version": TEST_SET_SCHEMA_VERSION,
        "split": split,
        "ground_truth_source": ground_truth_source,
        "qa_mode": "retrieval_only",
        "source_sha256": _source_sha256(query_root, source_paths),
        "cases": cases,
    }
    validate_query_test_set(result)
    return result


def write_query_test_set(path: Path, value: dict[str, Any]) -> None:
    """Write a test-set artifact as deterministic UTF-8 JSON atomically."""

    validate_query_test_set(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        temp_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def load_query_test_set(path: Path) -> dict[str, Any]:
    """Load and validate a test-set JSON object from disk."""

    value = json.loads(path.read_text(encoding="utf-8"))
    validate_query_test_set(value)
    return value


def _validate_source_rows(
    path: Path,
    kind: str,
    event_count: int,
    rows: list[tuple[str, ...]],
) -> None:
    expected_width = event_count + 1 if kind == "trake" else 3 if kind == "qa" else 2
    if not rows:
        raise ValueError(f"{path}: ground truth must contain at least one row")
    seen: set[tuple[str, ...]] = set()
    for row_number, row in enumerate(rows, start=1):
        if len(row) != expected_width:
            raise ValueError(
                f"{path}: expected {expected_width} columns, got {len(row)} at row {row_number}"
            )
        if not row[0].strip():
            raise ValueError(f"{path}: video_id must be non-empty at row {row_number}")
        frame_values = row[1:-1] if kind == "qa" else row[1:]
        try:
            frame_indexes = tuple(int(value) for value in frame_values)
        except ValueError:
            frame_indexes = ()
        if len(frame_indexes) != len(frame_values) or any(value < 0 for value in frame_indexes):
            raise ValueError(
                f"{path}: frame index must be a non-negative integer at row {row_number}"
            )
        if kind == "trake" and tuple(sorted(frame_indexes)) != frame_indexes:
            raise ValueError(
                f"{path}: TRAKE frame indexes must be chronological at row {row_number}"
            )
        if row in seen:
            raise ValueError(f"{path}: duplicate ground truth at row {row_number}")
        seen.add(row)
    if kind == "qa" and any(not row[2].strip() for row in rows):
        raise ValueError(f"{path}: QA answer must be non-empty")
    if kind == "qa" and len({row[2] for row in rows}) != 1:
        raise ValueError(f"{path}: QA answers must be identical")


def _natural_path_key(path: Path) -> tuple[str | int, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def _source_sha256(query_root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative_path = path.relative_to(query_root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _parse_trake_events(query: str) -> list[str]:
    events = [
        match.group(1).strip()
        for line in query.splitlines()
        if (match := _TRAKE_EVENT.match(line.strip()))
    ]
    if not events:
        raise ValueError("no TRAKE events found")
    return events


def _kind_from_path(path: Path) -> str:
    for kind in ("kis", "qa", "trake"):
        if path.name.lower().endswith(f"-{kind}.txt"):
            return kind
    raise ValueError(f"{path}: unsupported query filename")


def _read_rows(path: Path) -> list[tuple[str, ...]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [tuple(row) for row in csv.reader(handle) if row]


def main() -> None:
    """CLI to compile and write a query test-set fixture."""

    parser = argparse.ArgumentParser(description="Compile query test-set fixture.")
    parser.add_argument("--query-root", type=Path, default=Path("artifacts/query"))
    parser.add_argument("--split", default="002")
    parser.add_argument("--source", default="human_verified")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/query_002.json"),
    )
    args = parser.parse_args()
    value = build_query_test_set(args.query_root, args.split, ground_truth_source=args.source)
    write_query_test_set(args.output, value)
    print(
        f"Wrote {len(value['cases'])} cases to {args.output} "
        f"(source_sha256: {value['source_sha256']})"
    )


if __name__ == "__main__":
    main()
