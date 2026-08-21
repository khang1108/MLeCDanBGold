#!/usr/bin/env python3
"""Benchmark BTC keyframe temporal coverage vs our pipeline's output.

So sánh:
- Gap thời gian trung bình / lớn nhất giữa các keyframe
- Số lượng keyframe / phút
- Nếu có pipeline checkpoint: so sánh trực tiếp BTC vs pipeline trên cùng video
"""
from __future__ import annotations
k
import glob
from pathlib import Path

import numpy as np
import pandas as pd


BTC_FRAMES = Path("data/metadata/frames.parquet")
PIPELINE_WORK_ROOT = Path("runs")  # thư mục checkpoint của pipeline


# ─────────────────────────────────────────────
# 1. Load BTC metadata
# ─────────────────────────────────────────────
print("=" * 60)
print("BTC KEYFRAME COVERAGE ANALYSIS")
print("=" * 60)

btc = pd.read_parquet(BTC_FRAMES)
btc = btc.sort_values(["video_id", "timestamp_ms"])

print(f"\nTổng số keyframes: {len(btc):,}")
print(f"Tổng số video: {btc['video_id'].nunique()}")

# ─────────────────────────────────────────────
# 2. Gap statistics per video
# ─────────────────────────────────────────────
def gap_stats(df: pd.DataFrame) -> pd.Series:
    ts = df["timestamp_ms"].values
    if len(ts) < 2:
        return pd.Series({"gap_mean_s": 0.0, "gap_max_s": 0.0, "gap_p95_s": 0.0,
                          "frame_count": len(ts), "duration_s": 0.0, "frames_per_min": 0.0})
    gaps = np.diff(ts) / 1000.0
    duration_s = (ts[-1] - ts[0]) / 1000.0
    return pd.Series({
        "gap_mean_s": gaps.mean(),
        "gap_max_s": gaps.max(),
        "gap_p95_s": np.percentile(gaps, 95),
        "frame_count": len(ts),
        "duration_s": duration_s,
        "frames_per_min": len(ts) / max(duration_s / 60, 1e-6),
    })

stats = btc.groupby("video_id").apply(gap_stats, include_groups=False).reset_index()

print("\n── BTC Gap Statistics ──────────────────────────────────")
print(f"Gap trung bình:   {stats['gap_mean_s'].mean():.1f}s")
print(f"Gap lớn nhất:     {stats['gap_max_s'].max():.1f}s  (worst: {stats.loc[stats['gap_max_s'].idxmax(), 'video_id']})")
print(f"Gap P95:          {stats['gap_p95_s'].mean():.1f}s")
print(f"Frames/phút:      {stats['frames_per_min'].mean():.1f} avg  |  {stats['frames_per_min'].min():.1f} min  |  {stats['frames_per_min'].max():.1f} max")

# Cảnh báo video có gap > 30s
BAD_GAP_THRESH = 30.0
bad_videos = stats[stats["gap_max_s"] > BAD_GAP_THRESH].sort_values("gap_max_s", ascending=False)
if len(bad_videos):
    print(f"\n⚠️  {len(bad_videos)} video có gap > {BAD_GAP_THRESH}s:")
    print(bad_videos[["video_id", "gap_max_s", "frames_per_min"]].head(10).to_string(index=False))
else:
    print(f"\n✅ Không có video nào có gap > {BAD_GAP_THRESH}s")

# ─────────────────────────────────────────────
# 3. So sánh với pipeline checkpoint (nếu có)
# ─────────────────────────────────────────────
checkpoint_files = sorted(glob.glob(str(PIPELINE_WORK_ROOT / "**/*.parquet"), recursive=True))
checkpoint_files = [f for f in checkpoint_files if "artifacts" not in f]

if checkpoint_files:
    print(f"\n── So sánh BTC vs Pipeline Checkpoint ──────────────────")
    print(f"Tìm thấy {len(checkpoint_files)} video checkpoint từ pipeline")

    comparisons = []
    for ckpt_path in checkpoint_files[:10]:
        video_id = Path(ckpt_path).stem
        btc_v = btc[btc["video_id"] == video_id]
        if btc_v.empty:
            continue
        try:
            pipe_v = pd.read_parquet(ckpt_path)
            meta_cols = [c for c in pipe_v.columns if c.startswith("_")]
            pipe_v = pipe_v.drop(columns=meta_cols, errors="ignore")
        except Exception:
            continue

        btc_gaps = np.diff(btc_v["timestamp_ms"].values) / 1000.0 if len(btc_v) > 1 else np.array([0.0])
        pipe_gaps = np.diff(pipe_v["timestamp_ms"].values) / 1000.0 if len(pipe_v) > 1 else np.array([0.0])

        comparisons.append({
            "video_id": video_id,
            "btc_frames": len(btc_v),
            "pipe_frames": len(pipe_v),
            "btc_gap_max_s": btc_gaps.max(),
            "pipe_gap_max_s": pipe_gaps.max(),
            "btc_gap_mean_s": btc_gaps.mean(),
            "pipe_gap_mean_s": pipe_gaps.mean(),
        })

    if comparisons:
        cmp = pd.DataFrame(comparisons)
        print(cmp.to_string(index=False))
        print(f"\nBTC avg frames: {cmp['btc_frames'].mean():.0f} vs Pipeline: {cmp['pipe_frames'].mean():.0f}")
        print(f"BTC avg max-gap: {cmp['btc_gap_max_s'].mean():.1f}s vs Pipeline: {cmp['pipe_gap_max_s'].mean():.1f}s")

        worse = cmp[cmp["btc_gap_max_s"] > cmp["pipe_gap_max_s"] * 1.5]
        if len(worse):
            print(f"\n⚠️  {len(worse)} video BTC có gap lớn hơn pipeline >50%:")
            print(worse[["video_id", "btc_gap_max_s", "pipe_gap_max_s"]].to_string(index=False))
        else:
            print("\n✅ BTC coverage tương đương hoặc tốt hơn pipeline")
else:
    print("\n(Không tìm thấy pipeline checkpoint để so sánh trực tiếp)")

print("\n" + "=" * 60)
print("KHUYẾN NGHỊ:")
if stats["gap_max_s"].max() < 60:
    print("✅ BTC keyframes đủ tốt. Proceed với ingestion pipeline.")
else:
    n_bad = (stats["gap_max_s"] >= 60).sum()
    print(f"⚠️  {n_bad} video có gap >= 60s. Xem xét bổ sung thêm frames.")
print("=" * 60)
