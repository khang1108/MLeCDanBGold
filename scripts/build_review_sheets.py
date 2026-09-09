"""Build contact sheets for manual inspection of benchmark candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    """Create one annotated JPEG sheet per response/query family."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet("artifacts/frame_store/frames.parquet")
    records = {
        str(row.frame_id): (Path("artifacts/custom-raw1fps-v1") / str(row.image_path), str(row.video_id), int(row.timestamp_ms))
        for row in table.itertuples()
    }
    font = ImageFont.load_default()
    for response_path in args.responses:
        for item in json.loads(Path(response_path).read_text(encoding="utf-8")):
            if item.get("status_code") != 200:
                continue
            body = item.get("response") or {}
            candidates = body.get("paths", []) if item.get("type") == "trake" else body.get("results", [])
            if not candidates:
                continue
            query_name = Path(str(item["query_file"])).stem
            tiles: list[tuple[Image.Image, str]] = []
            for rank, candidate in enumerate(candidates[: args.top_k], 1):
                ids = candidate.get("frame_ids", [])
                if item.get("type") == "trake":
                    ids = candidate.get("frame_ids", [])
                else:
                    ids = [candidate.get("frame_id"), *ids]
                for event_index, frame_id in enumerate(dict.fromkeys(str(value) for value in ids if value)):
                    record = records.get(frame_id)
                    if record is None or not record[0].is_file():
                        continue
                    try:
                        image = Image.open(record[0]).convert("RGB")
                    except OSError:
                        continue
                    image.thumbnail((320, 180))
                    tile = Image.new("RGB", (340, 220), "white")
                    tile.paste(image, ((340 - image.width) // 2, 4))
                    draw = ImageDraw.Draw(tile)
                    label = f"r{rank} e{event_index + 1} {record[1]} {record[2] / 1000:.1f}s"
                    draw.text((5, 190), label, fill="black", font=font)
                    tiles.append((tile, label))
            if not tiles:
                continue
            columns = 4
            rows = (len(tiles) + columns - 1) // columns
            sheet = Image.new("RGB", (columns * 340, rows * 220), "#dddddd")
            for index, (tile, _) in enumerate(tiles):
                sheet.paste(tile, ((index % columns) * 340, (index // columns) * 220))
            safe_name = query_name.replace("/", "_")
            sheet.save(output_dir / f"{safe_name}.jpg", quality=88)


if __name__ == "__main__":
    main()
