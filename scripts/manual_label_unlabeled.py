"""Persist conservative visual-review labels for benchmark queries without GT.

The benchmark has no official CSV labels for split 001 and split 003.  This
module records a human visual check of the displayed top-three candidate
windows, while preserving the API's canonical candidate identity.  These
labels are weak review signals and must not be reported as official recall.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# A top-three hit means that at least one displayed candidate window visibly
# supports the query's defining object/action.  Mere topical similarity is not
# enough, which keeps this manual signal conservative.
LABELS: dict[str, dict[str, Any]] = {
    # Split 001 (query-p1-*): no official GT in this benchmark run.
    "query-p1-1-kis": {
        "label": "near_miss", "confidence": 0.82, "top3_ranks": [2, 3],
        "reason": "Top-1 shows an exercise segment but the requested >5-person toe-touch formation and hat/glasses counts are not visible; ranks 2-3 show closer group exercise.",
    },
    "query-p1-2-kis": {
        "label": "relevant", "confidence": 0.97, "top3_ranks": [1, 2, 3],
        "reason": "Top-1 clearly shows water being released from a dam; adjacent and other top candidates show the same dam/spillway event.",
    },
    "query-p1-4-kis": {
        "label": "near_miss", "confidence": 0.98, "top3_ranks": [2, 3],
        "reason": "Top-1 is a zoo animal-feeding shot rather than the requested lions/measurement; rank 2 shows lions and rank 3 shows staff measuring an animal.",
    },
    "query-p1-5-kis": {
        "label": "relevant", "confidence": 0.86, "top3_ranks": [1, 3],
        "reason": "Top-1 window shows the pan and vegetables/squid preparation; rank 3 visibly includes the pan-shaking motion, while fine ingredient details are partly obscured.",
    },
    "query-p1-6-kis": {
        "label": "relevant", "confidence": 0.99, "top3_ranks": [1],
        "reason": "Top-1 is the Botswana diamond news shot with the suited man, woman in a magenta head covering, and rough gemstone; the mine context is in the same story.",
    },
    "query-p1-7-kis": {
        "label": "near_miss", "confidence": 0.86, "top3_ranks": [1, 2],
        "reason": "Top-1 shows star-shaped carrot preparation, and rank 2 shows the plated vegetables, but the requested boiling-in-mesh step is not visible in the reviewed windows.",
    },
    "query-p1-8-kis": {
        "label": "relevant", "confidence": 0.96, "top3_ranks": [1],
        "reason": "Top-1 neighboring frames show the chef mixing and arranging bar/flower-shaped ingredients in the steaming dish, including the central soft mixture.",
    },
    "query-p1-10-kis": {
        "label": "relevant", "confidence": 0.99, "top3_ranks": [1, 2, 3],
        "reason": "Top-1 and its neighboring frames clearly show black-handled shears cutting a grape bunch on the vine; all three candidates remain in the same grape-cutting sequence.",
    },
    "query-p1-11-kis": {
        "label": "near_miss", "confidence": 0.84, "top3_ranks": [],
        "reason": "The candidates show cyclists, but no reviewed frame clearly establishes the finish line, slow-motion finish, or the requested first/second/third jersey order.",
    },
    "query-p1-12-kis": {
        "label": "relevant", "confidence": 0.99, "top3_ranks": [1, 2],
        "reason": "Top-1 shows several motorbike riders at a fuel station and the fuel-price lower third, matching the defining scene.",
    },
    "query-p1-13-kis": {
        "label": "near_miss", "confidence": 0.85, "top3_ranks": [1, 2, 3],
        "reason": "Top-1 shows a fishing net at night and the other top windows show people/boats/net work and dawn water; the full flashlight-to-dawn-to-filming sequence is not all visible in one window.",
    },
    "query-p1-14-kis": {
        "label": "relevant", "confidence": 0.97, "top3_ranks": [1],
        "reason": "Top-1 window covers mixing the soft ingredient and arranging the bar/flower-shaped pieces in the steaming dish, matching the described preparation.",
    },
    "query-p1-16-trake": {
        "label": "near_miss", "confidence": 0.85, "top3_ranks": [1],
        "reason": "Top-1 path is clearly the lion-dance-on-poles performance, but the displayed frames do not prove every requested ordered event (two dragons, completed spin, and gong contact).",
    },
    "query-p1-18-kis": {
        "label": "relevant", "confidence": 0.95, "top3_ranks": [1],
        "reason": "Top-1 neighboring frames clearly show the finished noodle soup with meat, herbs, and chili; the garnish/plating context is visible.",
    },
    "query-p1-19-kis": {
        "label": "near_miss", "confidence": 0.90, "top3_ranks": [],
        "reason": "Top-1 shows a lion balancing/jumping on poles, but the requested pumpkin-and-yellow-flower bite is not visible in the reviewed top-three windows.",
    },
    "query-p1-20-kis": {
        "label": "wrong", "confidence": 0.98, "top3_ranks": [],
        "reason": "Top-1 is a car-crash/pond news card; the reviewed candidates do not show the requested three people, umbrellas, bear raincoat, and walk toward a house.",
    },
    "query-p1-21-kis": {
        "label": "near_miss", "confidence": 0.87, "top3_ranks": [3],
        "reason": "Top-1 is a plated seafood shot, while rank 3 shows shrimp being cut; the full bread-placement and shrimp-grilling sequence is not visible at top-1.",
    },
    "query-p1-22-kis": {
        "label": "relevant", "confidence": 0.99, "top3_ranks": [1, 2],
        "reason": "Top-1 and rank 2 show the woman in pink teaching the requested remember constructions on the board.",
    },
    "query-p1-23-kis": {
        "label": "wrong", "confidence": 0.93, "top3_ranks": [],
        "reason": "Top-1 is a green chemistry slide and the reviewed alternatives do not show the requested male teacher plus the described three-tier diagram.",
    },
    "query-p1-24-kis": {
        "label": "wrong", "confidence": 0.97, "top3_ranks": [],
        "reason": "Top-1 is a betel-leaf preparation scene, not the requested water-hyacinth craft products and tea-cup conversation; no top-three frame shows that defining scene.",
    },
    "query-p1-25-kis": {
        "label": "near_miss", "confidence": 0.90, "top3_ranks": [1, 3],
        "reason": "Top-1 shows school students performing with instruments and rank 3 shows a stage/drum, but the two MCs, red scarves, and piano are not all confirmed.",
    },
    # Split 003 (query-p2-*): query IDs are p2-prefixed, but these files are
    # in folder 003 and have no official GT in this benchmark run.
    "query-p2-1-kis": {
        "label": "relevant", "confidence": 0.93, "top3_ranks": [1, 3],
        "reason": "Top-1 shows the soup pot with mushroom/bamboo-like strips and a liquid being poured; rank 3 shows the corresponding stirring step, though tofu is not obvious.",
    },
    "query-p2-2-kis": {
        "label": "relevant", "confidence": 0.99, "top3_ranks": [1, 2, 3],
        "reason": "Top-1 and neighboring candidates show large student lines, class placards, covered corridors, and movement toward stairs/buildings.",
    },
    "query-p2-3-kis": {
        "label": "relevant", "confidence": 0.95, "top3_ranks": [1, 2, 3],
        "reason": "Top-1 window shows ingredients being added to boiling soup; ranks 2-3 show the carrot/green/mushroom additions in the same preparation pattern.",
    },
    "query-p2-4-kis": {
        "label": "relevant", "confidence": 0.99, "top3_ranks": [1, 3],
        "reason": "Top-1 clearly shows the female teacher and the two requested past-tense example sentences on the green board.",
    },
    "query-p2-7-kis": {
        "label": "near_miss", "confidence": 0.90, "top3_ranks": [2],
        "reason": "Top-1 shows a motorbike at night but not both riders; rank 2 clearly shows two young men in the unsafe prone posture described by the query.",
    },
    "query-p2-9-kis": {
        "label": "wrong", "confidence": 0.98, "top3_ranks": [],
        "reason": "Top-1 is an aerial flower carpet and the alternatives show unrelated stage/decorations; no reviewed frame shows the requested cream fashion and fabric crafts.",
    },
    "query-p2-10-kis": {
        "label": "relevant", "confidence": 0.99, "top3_ranks": [1, 2, 3],
        "reason": "Top-1 and neighboring windows clearly show vintage-style amphibious cars moving through a canal and under/near bridges.",
    },
    "query-p2-11-kis": {
        "label": "near_miss", "confidence": 0.84, "top3_ranks": [],
        "reason": "Top-1 is an intermediate red-sauce/cylinder cooking shot; the requested expanded white stick-like finished product is not visible, so only broad preparation similarity is supported.",
    },
    "query-p2-12-kis": {
        "label": "relevant", "confidence": 0.96, "top3_ranks": [1],
        "reason": "Top-1 neighboring frames show liquid from a white bowl entering the red pan over flame, matching the defining pour-and-pan action.",
    },
    "query-p2-13-kis": {
        "label": "relevant", "confidence": 0.99, "top3_ranks": [1, 3],
        "reason": "Top-1 visibly shows large rows/piles of woven bamboo fish traps beside coconut trees and river boats.",
    },
    "query-p2-15-kis": {
        "label": "relevant", "confidence": 0.94, "top3_ranks": [1],
        "reason": "Top-1 shows the chef coating the prepared green ingredient in batter; nearby frames show the same red-pan cooking setup.",
    },
    "query-p2-16-kis": {
        "label": "relevant", "confidence": 0.98, "top3_ranks": [1, 2],
        "reason": "Top-1 clearly shows a dark liquid being poured into a transparent pot over a gas flame, consistent with the described broth seasoning sequence.",
    },
    "query-p2-17-kis": {
        "label": "near_miss", "confidence": 0.84, "top3_ranks": [2, 3],
        "reason": "Top-1 shows cutting a white segmented ingredient; ranks 2-3 show related radish cutting tools, but frying/halving-with-core-removal is not confirmed.",
    },
    "query-p2-19-kis": {
        "label": "relevant", "confidence": 0.94, "top3_ranks": [1, 2],
        "reason": "Top-1 shows the yellow lion balanced on a pole beside the purple/pink flower target; nearby frames remain in the same pole-and-flower performance.",
    },
    "query-p2-20-kis": {
        "label": "wrong", "confidence": 0.99, "top3_ranks": [],
        "reason": "Top-1 is a polyamide/polyester condensation-reaction slide, whereas the query asks for a compound-group slide with two grain variants and a linked-chain classification; the other reviewed slides are unrelated.",
    },
    "query-p2-21-trake": {
        "label": "near_miss", "confidence": 0.87, "top3_ranks": [1],
        "reason": "Top-1 path is the correct shrimp-and-orange recipe and includes cutting/plating frames, but the displayed path does not prove every ordered event and exact first/fourth placement timing.",
    },
    "query-p2-22-kis": {
        "label": "relevant", "confidence": 0.99, "top3_ranks": [1, 2],
        "reason": "Top-1 clearly shows the electric multicopter taxi flying over landscaped grounds; the lower-third identifies the French test.",
    },
    "query-p2-23-kis": {
        "label": "relevant", "confidence": 0.96, "top3_ranks": [1, 2],
        "reason": "Top-1 shows buffalo racing through a muddy field and rank 2 shows the requested light/white pair, supporting both parts of the scene.",
    },
    "query-p2-25-kis": {
        "label": "near_miss", "confidence": 0.86, "top3_ranks": [1, 2],
        "reason": "Top-1 shows a ladle/chopsticks over boiling soup and rank 2 shows meat and broth preparation, but the exact thin-meat-to-noodle-bowl sequence is not fully visible.",
    },
    "query-p2-26-kis": {
        "label": "relevant", "confidence": 0.99, "top3_ranks": [1],
        "reason": "Top-1 neighboring frames show the perpendicular point-to-plane diagram, intersecting planes, and later solid-geometry diagrams described by the query.",
    },
    "query-p2-28-kis": {
        "label": "relevant", "confidence": 0.99, "top3_ranks": [1, 2],
        "reason": "Top-1 clearly shows the glasses-wearing woman holding a white bag with a green logo among hanging clothes/bags.",
    },
    "query-p2-29-kis": {
        "label": "relevant", "confidence": 0.93, "top3_ranks": [1, 2],
        "reason": "Top-1 shows soup being ladled from a large pot into a smaller bowl; rank 2 shows the striped cloth/wooden-tray serving context, although the count of exactly two scoops is not provable from stills.",
    },
    "query-p2-30-kis": {
        "label": "near_miss", "confidence": 0.90, "top3_ranks": [2, 3],
        "reason": "Top-1 shows a cyclist on the tree-lined course; ranks 2-3 show the peloton, but the finish flag and bystander holding phone/selfie stick are not simultaneously visible.",
    },
    "query-p2-31-kis": {
        "label": "near_miss", "confidence": 0.84, "top3_ranks": [1, 3],
        "reason": "Top-1 shows a student and fan in a bamboo setting and rank 3 shows another child with a fan, but the light-haired foreign participant packing leaf-wrapped cakes is not visible.",
    },
    "query-p2-32-kis": {
        "label": "near_miss", "confidence": 0.87, "top3_ranks": [],
        "reason": "Top-1 shows a child's face on a phone, but the mother/son conversation, later son interview, and black woman passing behind are not established by the reviewed top-three frames.",
    },
    "query-p2-33-kis": {
        "label": "near_miss", "confidence": 0.88, "top3_ranks": [],
        "reason": "Top-1 is a cyclist with a white helmet and the alternatives show cycling, but no frame clearly shows the two leading riders and black-versus-white helmet comparison.",
    },
    "query-p2-34-trake": {
        "label": "near_miss", "confidence": 0.92, "top3_ranks": [],
        "reason": "Top-1 path is lion dancing on poles but the visible lion is yellow rather than the requested red lion, and the ordered event details are not confirmed.",
    },
    "query-p2-36-kis": {
        "label": "near_miss", "confidence": 0.88, "top3_ranks": [],
        "reason": "Top-1 shows an artist using a stylus on a round surface, but the requested opening handshake-with-Ho-Chi-Minh image and special-material portrait are not visible.",
    },
}


def _candidate_summary(candidate: dict[str, Any], rank: int) -> dict[str, Any]:
    """Keep canonical candidate identity and timing for auditability."""

    return {
        "rank": rank,
        "video_id": candidate.get("video_id"),
        "frame_id": candidate.get("frame_id"),
        "frame_ids": candidate.get("frame_ids", []),
        "frame_idx": candidate.get("frame_idx"),
        "frame_idxs": candidate.get("frame_idxs", []),
        "timestamp_ms": candidate.get("timestamp_ms"),
        "timestamps_ms": candidate.get("timestamps_ms", []),
        "score": candidate.get("score"),
    }


def build_records(response_paths: list[Path]) -> list[dict[str, Any]]:
    """Join static visual judgments to the saved top-three API candidates."""

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for response_path in response_paths:
        payload = json.loads(response_path.read_text(encoding="utf-8"))
        for item in payload:
            query_file = str(item["query_file"])
            query_id = Path(query_file).stem
            if query_id not in LABELS or query_id in seen:
                continue
            seen.add(query_id)
            body = item.get("response") or {}
            candidates = body.get("paths", []) if item.get("type") == "trake" else body.get("results", [])
            judgment = LABELS[query_id]
            split = Path(query_file).parent.name
            records.append(
                {
                    "query_id": query_id,
                    "query_file": query_file,
                    "split": split,
                    "type": item.get("type"),
                    "review_sheet": f"artifacts/benchmark/2026-09-06/review_sheets/{query_id}.jpg",
                    "manual_label": judgment["label"],
                    "confidence": judgment["confidence"],
                    "top3_hit": bool(judgment["top3_ranks"]),
                    "top3_hit_ranks": judgment["top3_ranks"],
                    "reason": judgment["reason"],
                    "status_code": item.get("status_code"),
                    "top3_candidates": [_candidate_summary(c, rank) for rank, c in enumerate(candidates[:3], 1)],
                }
            )
    return sorted(records, key=lambda row: row["query_id"])


def write_outputs(records: list[dict[str, Any]], output_json: Path, output_csv: Path) -> None:
    """Write JSON audit records and a compact CSV review table."""

    labels = Counter(row["manual_label"] for row in records)
    by_split: dict[str, dict[str, int]] = {}
    for row in records:
        by_split.setdefault(row["split"], Counter())
        by_split[row["split"]][row["manual_label"]] += 1
    envelope = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "weak manual visual review; not official ground truth",
        "reviewer_method": "Viewed every generated review sheet (top-1 plus displayed neighboring frames and top-2/top-3 windows).",
        "top3_hit_definition": "At least one of ranks 1-3 visibly supports the defining object/action; topical similarity alone is not a hit.",
        "label_definitions": {
            "relevant": "Top-1 window visibly supports the defining query event/content.",
            "near_miss": "Top-1 is temporally/semantically related but misses a requested detail or ordered action.",
            "wrong": "Top-1 does not support the defining event/content.",
            "uncertain": "Evidence is insufficient to decide visually.",
        },
        "counts": {
            "reviewed_sheets": len(records),
            "by_split": {split: dict(counts) for split, counts in sorted(by_split.items())},
            "by_label": dict(labels),
            "top3_hit": sum(1 for row in records if row["top3_hit"]),
        },
        "records": records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "query_id", "query_file", "split", "type", "manual_label", "confidence",
        "top3_hit", "top3_hit_ranks", "status_code", "review_sheet", "reason",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            compact = {field: row.get(field) for field in fields}
            compact["top3_hit_ranks"] = ",".join(str(rank) for rank in row["top3_hit_ranks"])
            writer.writerow(compact)


def main() -> None:
    """Generate weak-label artifacts from the completed response files."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", nargs="+", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    records = build_records(args.responses)
    missing = sorted(set(LABELS) - {row["query_id"] for row in records})
    if missing:
        raise SystemExit(f"Missing response records for manual labels: {missing}")
    write_outputs(records, args.output_json, args.output_csv)
    print(json.dumps({"records": len(records), "labels": dict(Counter(r["manual_label"] for r in records))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
