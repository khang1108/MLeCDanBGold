#!/usr/bin/env python3
"""Ingest BTC-provided keyframes into the canonical frame store format.

Đọc data/metadata/frames.parquet (format của BTC) và convert sang
frames.parquet chuẩn mà enrichment pipeline (Caption, OCR, Visual Index) yêu cầu.

Cách dùng:
    python3 scripts/ingest_btc_keyframes.py \\
        --btc-root data/ \\
        --output-root runs/btc-keyframes-v1/artifacts/frame_store \\
        [--data-root /home/phuckhang/MyWorkspace/HCMAI_2026]

Sau khi chạy xong, output_root sẽ chứa:
    frames.parquet   ← file canonical mà enrichment pipeline đọc
    manifest.json    ← metadata về nguồn gốc dữ liệu
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_fps_per_video(df: pd.DataFrame) -> dict[str, float]:
    """Ước lượng FPS từ frame_idx / timestamp_ms của từng video."""
    fps_map: dict[str, float] = {}
    for video_id, grp in df.groupby("video_id"):
        # Lấy các frame có timestamp > 0 để tránh chia cho 0
        valid = grp[grp["timestamp_ms"] > 0]
        if len(valid) < 2:
            fps_map[str(video_id)] = 30.0  # fallback
            continue
        # fps ≈ frame_idx / (timestamp_ms / 1000)
        estimates = valid["frame_idx"] / (valid["timestamp_ms"] / 1000.0)
        fps = float(estimates.median())
        # Làm tròn về giá trị chuẩn phổ biến
        for standard in [24.0, 25.0, 30.0]:
            if abs(fps - standard) < 1.5:
                fps = standard
                break
        fps_map[str(video_id)] = fps
    return fps_map


def ingest(btc_root: Path, output_root: Path, data_root: Path) -> None:
    btc_parquet = btc_root / "metadata" / "frames.parquet"
    if not btc_parquet.exists():
        raise FileNotFoundError(f"BTC metadata not found: {btc_parquet}")

    logger.info(f"Reading BTC metadata: {btc_parquet}")
    btc = pd.read_parquet(btc_parquet)
    btc = btc.sort_values(["video_id", "timestamp_ms"]).reset_index(drop=True)
    logger.info(f"Loaded {len(btc):,} frames across {btc['video_id'].nunique()} videos")

    # ── FPS per video ──────────────────────────────────────────
    logger.info("Computing FPS per video...")
    fps_map = compute_fps_per_video(btc)

    # ── Build canonical frame records ──────────────────────────
    logger.info("Building canonical FrameRecord rows...")

    def make_image_path(rel_path: str) -> str:
        """Convert relative BTC path → absolute path."""
        abs_path = (data_root / rel_path).resolve()
        if not abs_path.exists():
            logger.warning(f"Image not found: {abs_path}")
        return str(abs_path)

    rows = []
    for _, row in btc.iterrows():
        video_id = str(row["video_id"])
        fps = fps_map.get(video_id, 30.0)
        frame_idx = int(row["frame_idx"])
        timestamp_ms = int(row["timestamp_ms"])

        # pts: approximate frame position in video stream
        pts = frame_idx

        rows.append({
            "frame_id":         str(row["frame_id"]),
            "video_id":         video_id,
            "frame_idx":        frame_idx,
            "keyframe_order":   int(row["keyframe_order"]),
            "timestamp_ms":     timestamp_ms,
            "fps":              fps,
            "image_path":       make_image_path(str(row["image_path"])),
            "thumbnail_path":   None,
            "width":            int(row["width"]),
            "height":           int(row["height"]),
            "shot_id":          None,
            "event_id":         None,
            "is_anchor":        True,
            "pts":              pts,
            "time_base":        f"1/{int(fps)}",
            "motion_score":     0.0,
            "shot_score":       0.0,
            "event_score":      0.0,
            "selection_reasons": ("btc_keyframe",),
        })

    canonical = pd.DataFrame(rows)

    # ── Validate images exist ───────────────────────────────────
    missing = canonical[~canonical["image_path"].apply(lambda p: Path(p).exists())]
    if len(missing):
        logger.warning(f"{len(missing):,} image files not found on disk! First 5:")
        for p in missing["image_path"].head(5):
            logger.warning(f"  {p}")
    else:
        logger.info("✅ All image files verified on disk")

    # ── Write output ────────────────────────────────────────────
    output_root.mkdir(parents=True, exist_ok=True)
    output_parquet = output_root / "frames.parquet"

    # Atomic write (write partial, then rename)
    partial = output_parquet.with_suffix(".parquet.partial")
    canonical.to_parquet(partial, index=False)
    partial.replace(output_parquet)
    logger.info(f"✅ Wrote frames.parquet: {output_parquet}")
    logger.info(f"   {len(canonical):,} frames | {canonical['video_id'].nunique()} videos")

    # ── Write manifest.json ─────────────────────────────────────
    manifest = {
        "pipeline_version": "btc-keyframe-ingestion-v1",
        "source": "btc_provided_keyframes",
        "btc_root": str(btc_root.resolve()),
        "video_count": int(canonical["video_id"].nunique()),
        "frame_count": int(len(canonical)),
        "limited_run": False,
        "resume_enabled": False,
        "fps_map_sample": dict(list(fps_map.items())[:5]),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info(f"✅ Wrote manifest.json: {manifest_path}")

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("INGESTION COMPLETE")
    print("=" * 55)
    print(f"Output:          {output_root}")
    print(f"Total frames:    {len(canonical):,}")
    print(f"Total videos:    {canonical['video_id'].nunique()}")
    fps_counts = pd.Series(fps_map.values()).round(0).value_counts()
    print(f"FPS distribution: {fps_counts.to_dict()}")
    print("\nNext step — chạy enrichment pipeline:")
    print(f"  python scripts/generate_enrichment.py \\")
    print(f"    --frames-parquet {output_parquet} \\")
    print(f"    --config configs/enrichment.yaml")
    print("=" * 55)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest BTC keyframes into canonical frame store")
    parser.add_argument(
        "--btc-root", type=Path, default=Path("data"),
        help="Root folder chứa metadata/ và keyframes/ của BTC (default: data/)",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("runs/btc-keyframes-v1/artifacts/frame_store"),
        help="Nơi ghi frames.parquet output (default: runs/btc-keyframes-v1/artifacts/frame_store)",
    )
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("data"),
        help="Root để resolve relative image paths (default: data/)",
    )
    args = parser.parse_args()

    ingest(
        btc_root=args.btc_root.resolve(),
        output_root=args.output_root.resolve(),
        data_root=args.data_root.resolve(),
    )


if __name__ == "__main__":
    main()
