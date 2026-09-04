#!/usr/bin/env python3
"""Export multimodal corpus metadata and specialist enrichments to JSONL.

This script joins canonical FrameStore coordinates with specialist enrichments
(Caption, OCR, Object detections, and frame-aligned ASR transcripts) along with
video metadata from organizer media-info JSON files into a structured JSON / JSONL.

Output format per record:
{
  "videos": {
    "L24_V001": {
      "folder_id": "L24",
      "title": "...",
      "fps": 25.0
    }
  },
  "frames": [
    {
      "frame_id": "L24_V001_keyframe_000001",
      "frame_idx": 0,
      "video_id": "L24_V001",
      "timestamp": 0.0,
      "metadata": {
        "caption": "...",
        "asr": "...",
        "ocr": "...",
        "objects": {"car": 2, "person": 3}
      }
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence, cast

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("export_jsonl")


def _clean_caption(text: Any) -> str:
    """Clean model generation artifacts such as padding tokens and excess spaces."""
    if not text or pd.isna(text):
        return ""
    clean = str(text).replace("<pad>", "").replace("</s>", "").strip()
    return " ".join(clean.split())


def _clean_ocr(text: Any) -> str:
    """Clean and normalize OCR text."""
    if not text or pd.isna(text):
        return ""
    return " ".join(str(text).split())


def _load_media_info(media_info_dir: Path | None) -> dict[str, dict[str, Any]]:
    """Load video metadata from organizer media-info directory if available."""
    metadata_by_video: dict[str, dict[str, Any]] = {}
    if media_info_dir is None or not media_info_dir.is_dir():
        logger.warning("Media-info directory not found or not provided: %s", media_info_dir)
        return metadata_by_video

    for json_file in media_info_dir.glob("*.json"):
        video_id = json_file.stem
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            metadata_by_video[video_id] = {
                "title": data.get("title", video_id),
                "author": data.get("author", ""),
                "channel_id": data.get("channel_id", ""),
                "description": data.get("description", ""),
                "keywords": data.get("keywords", []),
                "publish_date": data.get("publish_date", ""),
                "watch_url": data.get("watch_url", ""),
            }
        except Exception as e:
            logger.debug("Failed reading media-info for %s: %s", video_id, e)
    logger.info("Loaded media-info for %d videos", len(metadata_by_video))
    return metadata_by_video


def _find_file(root: Path, relative_paths: Sequence[str]) -> Path | None:
    """Find the first existing path among candidates relative to root."""
    for rel in relative_paths:
        candidate = root / rel
        if candidate.is_file():
            return candidate
    return None


def _load_objects_map(artifacts_root: Path, video_filter: Sequence[str] | None) -> dict[str, dict[str, int]]:
    """Load object counts per frame_id from detections or frames parquet."""
    objects_by_frame: dict[str, dict[str, int]] = defaultdict(dict)

    # 1. Prefer counts_json from objects/frames.parquet (official summary)
    frames_obj_path = _find_file(
        artifacts_root,
        [
            "enrichment/objects/frames.parquet",
            "objects/frames.parquet",
            "custom-raw1fps-v1/enrichment/objects/frames.parquet",
        ],
    )
    if frames_obj_path and frames_obj_path.is_file():
        logger.info("Loading object counts from: %s", frames_obj_path)
        df_obj = pd.read_parquet(frames_obj_path, columns=["frame_id", "video_id", "counts_json"])
        if video_filter is not None:
            df_obj = df_obj[df_obj["video_id"].isin(list(video_filter))]
        for _, row in df_obj.iterrows():
            fid = str(row["frame_id"])
            if bool(pd.notna(row.get("counts_json"))):
                try:
                    counts = json.loads(str(row["counts_json"]))
                    if isinstance(counts, dict):
                        objects_by_frame[fid] = {str(k): int(v) for k, v in counts.items()}
                except Exception:
                    pass
        return objects_by_frame

    # 2. Fallback to detections table
    detections_path = _find_file(
        artifacts_root,
        [
            "enrichment/objects/detections.parquet",
            "objects/detections.parquet",
            "custom-raw1fps-v1/enrichment/objects/detections.parquet",
        ],
    )
    if detections_path and detections_path.is_file():
        logger.info("Loading object counts from detections: %s", detections_path)
        det_df = pd.read_parquet(detections_path, columns=["frame_id", "video_id", "label"])
        if video_filter is not None:
            det_df = det_df[det_df["video_id"].isin(list(video_filter))]
        
        for frame_id, group in det_df.groupby("frame_id", sort=False):
            counts: dict[str, int] = {}
            for lbl in group["label"]:
                if pd.notna(lbl):
                    lbl_str = str(lbl).strip()
                    if lbl_str:
                        counts[lbl_str] = counts.get(lbl_str, 0) + 1
            objects_by_frame[str(frame_id)] = counts

    return objects_by_frame


def _load_transcripts(
    transcripts_root: Path,
    video_filter: Sequence[str] | None,
) -> dict[str, list[tuple[int, int, str]]]:
    """Load transcript segments (start_ms, end_ms, text) grouped by video_id."""
    transcripts_by_video: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    if not transcripts_root.is_dir():
        logger.warning("Transcripts root does not exist: %s", transcripts_root)
        return transcripts_by_video

    transcript_files = list(transcripts_root.rglob("*.parquet"))
    logger.info("Scanning %d transcript files from %s", len(transcript_files), transcripts_root)

    filter_set = set(video_filter) if video_filter is not None else None
    for f in transcript_files:
        video_id = f.stem
        if filter_set is not None and video_id not in filter_set:
            continue
        try:
            df = pd.read_parquet(f)
            if len(df) == 0 or "text" not in df.columns:
                continue
            segments = []
            for _, row in df.iterrows():
                text = row.get("text")
                if bool(pd.notna(text)) and str(text).strip():
                    segments.append((
                        int(cast(Any, row.get("start_ms", 0))),
                        int(cast(Any, row.get("end_ms", 0))),
                        str(text).strip(),
                    ))
            segments.sort(key=lambda s: (s[0], s[1]))
            transcripts_by_video[video_id] = segments
        except Exception as e:
            logger.debug("Failed reading transcript for %s: %s", video_id, e)

    return transcripts_by_video


def _match_asr_for_timestamp(
    segments: list[tuple[int, int, str]],
    timestamp_ms: int,
    window_ms: int,
) -> str:
    """Find and concatenate unique ASR segments within the temporal window."""
    if not segments:
        return ""
    start_bound = max(0, timestamp_ms - window_ms)
    end_bound = timestamp_ms + window_ms
    matching_texts: list[str] = []
    seen: set[str] = set()

    for seg_start, seg_end, text in segments:
        if seg_start <= end_bound and seg_end >= start_bound:
            key = text.lower()
            if key not in seen:
                seen.add(key)
                matching_texts.append(text)

    return " ".join(matching_texts)


def export_corpus_jsonl(
    artifacts_root: Path,
    media_info_dir: Path | None,
    output_path: Path,
    group_by: str = "video",
    asr_window_ms: int = 2000,
    video_ids: list[str] | None = None,
    limit_videos: int | None = None,
    pretty: bool = False,
) -> None:
    """Export the enriched multimodal dataset into JSONL format."""
    # 1. Resolve FrameStore
    frames_path = _find_file(
        artifacts_root,
        [
            "frame_store/frames.parquet",
            "frames.parquet",
            "custom-raw1fps-v1/frame_store/frames.parquet",
        ],
    )
    if not frames_path or not frames_path.is_file():
        raise FileNotFoundError(f"frames.parquet not found in {artifacts_root}")

    logger.info("Reading canonical frames from: %s", frames_path)
    frames_df = pd.read_parquet(frames_path)
    all_videos = sorted(frames_df["video_id"].unique().tolist())

    if video_ids:
        target_video_set = set(video_ids)
        all_videos = [v for v in all_videos if v in target_video_set]
    if limit_videos and limit_videos > 0:
        all_videos = all_videos[:limit_videos]

    video_filter = list(dict.fromkeys(all_videos))
    frames_df = frames_df[frames_df["video_id"].isin(video_filter)]
    logger.info("Processing %d videos (%d total frames)", len(all_videos), len(frames_df))

    # 2. Load Captions
    captions_map: dict[str, str] = {}
    captions_path = _find_file(
        artifacts_root,
        [
            "enrichment/captions/captions.parquet",
            "captions/captions.parquet",
            "custom-raw1fps-v1/enrichment/captions/captions.parquet",
        ],
    )
    if captions_path and captions_path.is_file():
        logger.info("Reading captions from: %s", captions_path)
        cap_df = pd.read_parquet(captions_path, columns=["frame_id", "video_id", "text"])
        cap_df = cap_df[cap_df["video_id"].isin(video_filter)]
        for _, row in cap_df.iterrows():
            captions_map[str(row["frame_id"])] = _clean_caption(row["text"])

    # 3. Load OCR
    ocr_map: dict[str, str] = {}
    ocr_path = _find_file(
        artifacts_root,
        [
            "enrichment/ocr/frames.parquet",
            "ocr/frames.parquet",
            "custom-raw1fps-v1/enrichment/ocr/frames.parquet",
        ],
    )
    if ocr_path and ocr_path.is_file():
        logger.info("Reading OCR from: %s", ocr_path)
        ocr_cols = ["frame_id", "video_id", "normalized_text", "raw_text"]
        actual_cols = [c for c in ocr_cols if c in pd.read_parquet(ocr_path, columns=None).columns]
        ocr_df = pd.read_parquet(ocr_path, columns=actual_cols)
        ocr_df = ocr_df[ocr_df["video_id"].isin(video_filter)]
        for _, row in ocr_df.iterrows():
            txt = row.get("normalized_text") if bool(pd.notna(row.get("normalized_text"))) else row.get("raw_text")
            ocr_map[str(row["frame_id"])] = _clean_ocr(txt)

    # 4. Load Objects
    objects_map = _load_objects_map(artifacts_root, video_filter)

    # 5. Load Transcripts (ASR)
    transcripts_root = artifacts_root / "enrichment" / "transcripts"
    if not transcripts_root.exists():
        transcripts_root = artifacts_root / "transcripts"
    transcripts_map = _load_transcripts(transcripts_root, video_filter)

    # 6. Load Media-info metadata
    media_info_map = _load_media_info(media_info_dir)

    # 7. Group frames by video
    logger.info("Grouping frames by video...")
    frames_by_video: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for _, row in tqdm(frames_df.iterrows(), total=len(frames_df), desc="Formatting frames"):
        vid = str(row["video_id"])
        fid = str(row["frame_id"])
        ts_ms = int(cast(Any, row.get("timestamp_ms", 0)))
        ts_sec = round(ts_ms / 1000.0, 4)

        # Match ASR
        video_segments = transcripts_map.get(vid, [])
        asr_text = _match_asr_for_timestamp(video_segments, ts_ms, asr_window_ms)

        frame_data = {
            "frame_id": fid,
            "frame_idx": int(cast(Any, row.get("frame_idx", 0))),
            "video_id": vid,
            "timestamp": ts_sec,
            "metadata": {
                "caption": captions_map.get(fid, ""),
                "asr": asr_text,
                "ocr": ocr_map.get(fid, ""),
                "objects": objects_map.get(fid, {}),
            },
        }
        frames_by_video[vid].append(frame_data)

    # Helper to construct video info header
    def make_video_header(v_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        header = {}
        for vid in v_ids:
            folder_id = vid.split("_")[0] if "_" in vid else vid
            info = media_info_map.get(vid, {})
            # Read fps from first frame of video
            v_frames = cast(pd.DataFrame, frames_df[frames_df["video_id"] == vid])
            fps_val = float(v_frames.iloc[0].get("fps", 25.0)) if len(v_frames) > 0 else 25.0
            
            header[vid] = {
                "folder_id": folder_id,
                "title": info.get("title", vid),
                "fps": int(fps_val) if fps_val.is_integer() else fps_val,
            }
        return header

    # 8. Export records
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing JSONL records (group_by='%s') to: %s", group_by, output_path)

    with output_path.open("w", encoding="utf-8") as out_file:
        if group_by == "corpus":
            # 1 single document containing the whole corpus
            all_video_headers = make_video_header(all_videos)
            all_frames_flat = [frame for vid in all_videos for frame in frames_by_video.get(vid, [])]
            doc = {
                "videos": all_video_headers,
                "frames": all_frames_flat,
            }
            if pretty:
                json.dump(doc, out_file, ensure_ascii=False, indent=2)
            else:
                out_file.write(json.dumps(doc, ensure_ascii=False) + "\n")

        elif group_by == "folder":
            # 1 line per folder_id (e.g. L21, L22, ...)
            folders: defaultdict[str, list[str]] = defaultdict(list)
            for vid in all_videos:
                folder_id = vid.split("_")[0] if "_" in vid else vid
                folders[folder_id].append(vid)

            for folder_id, v_ids in tqdm(sorted(folders.items()), desc="Writing folders"):
                folder_video_headers = make_video_header(v_ids)
                folder_frames = [frame for vid in v_ids for frame in frames_by_video.get(vid, [])]
                record = {
                    "videos": folder_video_headers,
                    "frames": folder_frames,
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        else:
            # group_by == "video" (default): 1 line per video
            for vid in tqdm(all_videos, desc="Writing video JSONL lines"):
                vid_header = make_video_header([vid])
                vid_frames = frames_by_video.get(vid, [])
                record = {
                    "videos": vid_header,
                    "frames": vid_frames,
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Export completed successfully: %s", output_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts"),
        help="Root directory containing frame_store and enrichment parquets (default: artifacts)",
    )
    parser.add_argument(
        "--media-info-dir",
        type=Path,
        default=Path("data/media-info-aic25-b1/media-info"),
        help="Directory containing organizer media-info JSON files (default: data/media-info-aic25-b1/media-info)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("artifacts/corpus_export.jsonl"),
        help="Output JSONL / JSON file path (default: artifacts/corpus_export.jsonl)",
    )
    parser.add_argument(
        "--group-by",
        choices=("video", "folder", "corpus"),
        default="video",
        help="Chunking unit per JSON line: 'video' (1 line/video), 'folder' (1 line/batch folder), 'corpus' (all in 1 document).",
    )
    parser.add_argument(
        "--asr-window-ms",
        type=int,
        default=2000,
        help="Temporal matching window in ms for ASR transcript alignment (default: 2000ms)",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        dest="video_ids",
        help="Filter specific video IDs (can be repeated)",
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        help="Limit number of videos to export (for testing)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Format JSON with indentation (applies when --group-by corpus)",
    )

    args = parser.parse_args(argv)
    export_corpus_jsonl(
        artifacts_root=args.artifacts_root,
        media_info_dir=args.media_info_dir,
        output_path=args.output,
        group_by=args.group_by,
        asr_window_ms=args.asr_window_ms,
        video_ids=args.video_ids,
        limit_videos=args.limit_videos,
        pretty=args.pretty,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
