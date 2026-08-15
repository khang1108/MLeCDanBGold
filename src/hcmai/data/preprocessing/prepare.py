"""Video Preprocessing & Frame Extraction Module.

Module này chịu trách nhiệm chuẩn bị dữ liệu (FrameStore) từ các raw videos.
Các chức năng chính bao gồm:
1. Phân tích video (Shot/Event boundary detection) sử dụng TransNet và GEBD.
2. Trích xuất frame (Keyframe extraction) kết hợp thuật toán khử trùng lặp (Deduplication) qua DinoEncoder.
3. Cơ chế cache/checkpoint (Resume-friendly) cho phép lưu tạm các frame đã xử lý để không phải chạy lại khi gặp lỗi.
4. Hỗ trợ chạy đa luồng (Multi-threading) để tối ưu hoá khả năng giải mã phần cứng và tận dụng tối đa GPU.

Toàn bộ siêu dữ liệu của các frame đã được chuẩn bị sẽ được tổng hợp vào file `frames.parquet`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import concurrent.futures

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from hcmai.common.schemas.frame import FrameRecord
from hcmai.data.preprocessing.config import PreprocessingConfig
from hcmai.data.preprocessing.models import EfficientGEBDDetector, TransNetDetector
from hcmai.data.preprocessing.selection import (
    CandidateFrame,
    DinoEncoder,
    deduplicate,
    restore_maximum_gap,
    select_candidates,
)
from hcmai.data.preprocessing.video import (
    analyze_video,
    discover_videos,
    iter_source_frames,
)

_CHECKPOINT_COLUMNS = ["_config_hash", "_source_size", "_source_mtime_ns"]
_PIPELINE_VERSION = "adaptive-frame-store-v2"


class FramePreparationSession:
    """Prepare staged videos incrementally and finalize one canonical store."""

    def __init__(
        self,
        config: PreprocessingConfig,
        *,
        shot_detector: Any | None = None,
        event_detector: Any | None = None,
        encoder: Any | None = None,
        resume: bool = True,
    ) -> None:
        self.config = config
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        self.model_fingerprints = _model_fingerprints(config)
        self.config_hash = _config_hash(config, self.model_fingerprints)
        self.resume = resume and config.dino_revision is not None
        self.shot_detector = shot_detector or TransNetDetector(config)
        self.event_detector = event_detector or EfficientGEBDDetector(config)
        self.encoder = encoder or DinoEncoder(config)

    def prepare_video(
        self,
        path: str | Path,
        *,
        source_version: str | None = None,
    ) -> pd.DataFrame:
        """Prepare or resume one already-local source video."""

        return _prepare_video(
            Path(path),
            self.config,
            self.shot_detector,
            self.event_detector,
            self.encoder,
            self.resume,
            self.config_hash,
            source_version,
        )

    def finalize(
        self,
        tables: list[pd.DataFrame],
        *,
        limited_run: bool = False,
        source: dict[str, object] | None = None,
    ) -> Path:
        """Publish metadata only after every requested video is prepared."""

        return _finalize_frame_store(
            self.config,
            tables,
            config_hash=self.config_hash,
            model_fingerprints=self.model_fingerprints,
            limited_run=limited_run,
            resume_enabled=self.resume,
            source=source,
        )


def _path_fingerprint(path: Path) -> str:
    """Hash a model artifact or source tree by path and file contents."""

    digest = hashlib.sha256()
    if not path.exists():
        digest.update(b"missing")
        return digest.hexdigest()
    
    # Nếu là file thì hash file, nếu là directory thì hash tất cả file trong directory
    paths = [path] if path.is_file() else sorted(
        item for item in path.rglob("*")
        if item.is_file()
        and ".git" not in item.parts
        and "__pycache__" not in item.parts
    )
    for item in paths:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _model_fingerprints(config: PreprocessingConfig) -> dict[str, str]:
    """Capture every configured local model input plus the remote revision."""

    local = {
        "transnet_repo": config.transnet_repo,
        "transnet_weights": config.transnet_weights,
        "efficientgebd_repo": config.efficientgebd_repo,
        "efficientgebd_config": config.efficientgebd_config,
        "efficientgebd_checkpoint": config.efficientgebd_checkpoint,
    }
    fingerprints = {
        name: _path_fingerprint(path.expanduser().resolve())
        for name, path in local.items()
    }
    fingerprints["dino_source"] = hashlib.sha256(
        f"{config.dino_model}@{config.dino_revision or 'unresolved'}".encode()
    ).hexdigest()
    return fingerprints


def _config_hash(
    config: PreprocessingConfig,
    model_fingerprints: dict[str, str] | None = None,
) -> str:
    """Hash settings that affect selected frames."""
    values = config.model_dump(
        mode="json",
        exclude={"videos_root", "output_root", "s3"},
    )
    values["pipeline_version"] = _PIPELINE_VERSION
    values["model_fingerprints"] = (
        model_fingerprints or _model_fingerprints(config)
    )
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_parquet(table: pd.DataFrame, path: Path) -> None:
    """Publish Parquet only after writing it completely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f"{path.suffix}.partial")
    table.to_parquet(partial, index=False)
    partial.replace(path)


def _write_json(values: dict[str, object], path: Path) -> None:
    """Atomically publish a deterministic JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f"{path.suffix}.partial")
    partial.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _checkpoint_path(config: PreprocessingConfig, video_id: str) -> Path:
    """Return the private checkpoint for one video."""
    return config.work_root / video_id.split("_", 1)[0] / f"{video_id}.parquet"


def _load_checkpoint(
    config: PreprocessingConfig,
    video_path: Path,
    resume: bool,
    config_hash: str,
    source_version: str | None = None,
) -> pd.DataFrame | None:
    """Reuse a checkpoint when its source, config, and images still match."""
    path = _checkpoint_path(config, video_path.stem)
    if not resume or not path.is_file():
        return None
    table = pd.read_parquet(path)
    stat = video_path.stat()
    version_matches = source_version is None or (
        "_source_version" in table
        and str(table["_source_version"].iloc[0]) == source_version
    )
    valid = (
        not table.empty
        and table["_config_hash"].iloc[0] == config_hash
        and int(table["_source_size"].iloc[0]) == stat.st_size
        and int(table["_source_mtime_ns"].iloc[0]) == stat.st_mtime_ns
        and version_matches
        and all(
            (config.output_root / value).is_file()
            for value in table["image_path"]
        )
    )
    checkpoint_columns = [
        column for column in (*_CHECKPOINT_COLUMNS, "_source_version")
        if column in table
    ]
    return table.drop(columns=checkpoint_columns) if valid else None


def _encode_images(paths: list[Path], encoder: Any, batch_size: int) -> np.ndarray:
    """Encode candidate images in small batches."""
    chunks = []

    # Batch size nhỏ giúp giảm tải bộ nhớ
    for start in range(0, len(paths), batch_size):
        images = []
        for path in paths[start : start + batch_size]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        chunks.append(np.asarray(encoder.encode(images), dtype=np.float32))
    return np.concatenate(chunks)


def _record(candidate: CandidateFrame, image_path: Path) -> dict[str, object]:
    """Convert one candidate to canonical frame metadata."""
    frame = candidate.frame
    return FrameRecord(
        frame_id=f"{frame.video_id}_frame_{frame.decode_index:09d}",
        video_id=frame.video_id,
        frame_idx=frame.frame_idx,
        timestamp_ms=frame.timestamp_ms,
        image_path=image_path.as_posix(),
        width=frame.width,
        height=frame.height,
        shot_id=f"{frame.video_id}_shot_{candidate.shot_id:06d}",
        event_id=f"{frame.video_id}_event_{candidate.event_id:06d}",
        is_anchor=candidate.protected,
        pts=frame.pts,
        time_base=frame.time_base,
        motion_score=frame.motion_score,
        shot_score=candidate.shot_score,
        event_score=candidate.event_score,
        selection_reasons=candidate.reasons,
    ).model_dump(mode="python")


def _recover_publication(final_dir: Path, partial_dir: Path) -> None:
    """Recover an interrupted directory publication and clear stale staging."""

    backup_dir = final_dir.parent / f".{final_dir.name}.backup"
    if backup_dir.exists():
        if final_dir.exists():
            shutil.rmtree(backup_dir)
        else:
            backup_dir.replace(final_dir)
    shutil.rmtree(partial_dir, ignore_errors=True)


def _publish_directory(partial_dir: Path, final_dir: Path) -> None:
    """Publish a completed image directory while preserving rollback data."""

    backup_dir = final_dir.parent / f".{final_dir.name}.backup"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if final_dir.exists():
        final_dir.replace(backup_dir)
    try:
        partial_dir.replace(final_dir)
    except Exception:
        if backup_dir.exists() and not final_dir.exists():
            backup_dir.replace(final_dir)
        raise
    shutil.rmtree(backup_dir, ignore_errors=True)


def _materialize(
    video_path: Path,
    candidates: list[CandidateFrame],
    config: PreprocessingConfig,
    encoder: Any,
) -> pd.DataFrame:
    """Write, deduplicate, and atomically publish one video's candidates."""
    group = video_path.stem.split("_", 1)[0]

    # 1. Định nghĩa đường dẫn chứa Frame đã xử lý
    final_dir = config.output_root / "images" / group / video_path.stem
    partial_dir = final_dir.parent / f".{video_path.stem}.partial"
    _recover_publication(final_dir, partial_dir)
    partial_dir.mkdir(parents=True)

    # 2. Lọc frame và lưu ảnh
    by_index = {item.frame.decode_index: item for item in candidates}
    images: dict[int, Path] = {}

    # Lặp qua từng frame của video
    for frame, source in iter_source_frames(video_path):
        candidate = by_index.get(frame.decode_index)

        if candidate is None:
            continue

        # Lưu ảnh vào thư mục tạm
        image_path = partial_dir / f"{frame.decode_index:09d}.jpg"
        source.to_image().convert("RGB").save(
            image_path, quality=config.image_quality
        )

        # Map lại index với đường dẫn
        images[frame.decode_index] = image_path

    # Kiểm tra và báo lỗi nếu có frame bị thiếu
    if len(images) != len(candidates):
        raise ValueError(f"Could not extract every candidate from {video_path.name}")

    # 3. Encoding và Deduplication
    paths = [images[item.frame.decode_index] for item in candidates]
    vectors = _encode_images(paths, encoder, config.dino_batch_size)
    retained = deduplicate(candidates, vectors, paths, config)
    retained = restore_maximum_gap(candidates, retained, config)
    retained_ids = {item.frame.decode_index for item in retained}

    # Xóa frame không được chọn
    for decode_index, image_path in images.items():
        if decode_index not in retained_ids:
            image_path.unlink()

    # 4. Publish
    _publish_directory(partial_dir, final_dir)
    image_root = Path("images") / group / video_path.stem
    rows = [
        _record(item, image_root / images[item.frame.decode_index].name)
        for item in retained
    ]
    return pd.DataFrame(rows, columns=list(FrameRecord.model_fields))


def _prepare_video(
    path: Path, config: PreprocessingConfig, shot_detector: Any,
    event_detector: Any, encoder: Any, resume: bool, config_hash: str,
    source_version: str | None = None,
) -> pd.DataFrame:
    """Prepare one video or reuse its checkpoint."""
    cached = _load_checkpoint(
        config, path, resume, config_hash, source_version
    )

    # 3. Tạo FramePreparationSession nếu chưa có
    if cached is not None:
        return cached

    # 4. Nếu chưa có checkpoint, tiến hành xử lý Video
    analysis = analyze_video(path, config, event_detector)

    # 5. Chọn Frame
    candidates = select_candidates(
        analysis.frames,
        shot_detector.score(path, analysis.shot_frames),
        analysis.event_scores,
        config,
    )

    # 6. Tạo DataFrame chứa các frame candidate đã chọn
    table = _materialize(path, candidates, config, encoder)
    stat = path.stat()

    # 7. Ghi Checkpoint
    checkpoint = table.assign(
        _config_hash=config_hash,
        _source_size=stat.st_size,
        _source_mtime_ns=stat.st_mtime_ns,
    )

    # 8. Gán Source Version nếu có
    if source_version is not None:
        checkpoint["_source_version"] = source_version

    # 9. Ghi checkpoint
    _write_parquet(checkpoint, _checkpoint_path(config, path.stem)) 
    return table


def _limit_config(
    config: PreprocessingConfig, limit: int | None,
) -> PreprocessingConfig:
    """Isolate smoke-test artifacts from the configured full-corpus output."""

    if limit is None:
        return config
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    return config.model_copy(update={
        "output_root": config.output_root.with_name(
            f"{config.output_root.name}.limit-{limit}"
        ),
    })


def _finalize_frame_store(
    config: PreprocessingConfig,
    tables: list[pd.DataFrame],
    *,
    config_hash: str,
    model_fingerprints: dict[str, str],
    limited_run: bool,
    resume_enabled: bool,
    source: dict[str, object] | None = None,
) -> Path:
    """Atomically publish canonical metadata after every video succeeds."""

    # 10. Nối các frame đã xử lý lại với nhau
    frames = pd.concat(tables, ignore_index=True).sort_values(
        ["video_id", "timestamp_ms", "frame_idx"], kind="stable"
    )
    
    # 11. Ghi file frames.parquet
    output = config.output_root / "frames.parquet"
    _write_parquet(frames.reset_index(drop=True), output)
    
    # 12. Tạo và ghi file manifest.json
    manifest: dict[str, object] = {
        "pipeline_version": _PIPELINE_VERSION,
        "config_hash": config_hash,
        "model_fingerprints": model_fingerprints,
        "video_count": len(set(frames["video_id"].astype(str))),
        "frame_count": len(frames),
        "limited_run": limited_run,
        "resume_enabled": resume_enabled,
    }

    if source is not None:
        manifest["source"] = source
    
    _write_json(manifest, config.output_root / "manifest.json")
    return output


def prepare_frame_store(
    config: PreprocessingConfig,
    *,
    shot_detector: Any | None = None,
    event_detector: Any | None = None,
    encoder: Any | None = None,
    resume: bool = True,
    limit: int | None = None,
) -> Path:
    """Build and return the canonical ``frames.parquet`` path."""
    if config.videos_root is None:
        raise ValueError("local preprocessing requires videos_root")

    # 1. Kiểm tra và tạo thư mục Output
    config = _limit_config(config, limit)
    config.output_root.mkdir(parents=True, exist_ok=True)

    # 2. Tính toán Hash của Config và Model
    model_fingerprints = _model_fingerprints(config)
    config_hash = _config_hash(config, model_fingerprints)

    # 3. Kiểm tra Resume
    checkpoint_resume = resume and config.dino_revision is not None

    # 4. Khởi tạo các Models
    shot_detector = shot_detector or TransNetDetector(config)
    event_detector = event_detector or EfficientGEBDDetector(config)
    encoder = encoder or DinoEncoder(config)
    
    # 5. Xử lý nhiều Video song song (tối ưu cho L40)
    video_paths = list(discover_videos(config, limit))
    tables = []
    
    # Sử dụng config.max_video_workers để tuỳ chỉnh linh hoạt
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_video_workers) as executor:
        future_to_path = {
            executor.submit(
                _prepare_video,
                path,
                config,
                shot_detector,
                event_detector,
                encoder,
                checkpoint_resume,
                config_hash,
            ): path
            for path in video_paths
        }
        for future in concurrent.futures.as_completed(future_to_path):
            tables.append(future.result())
    
    return _finalize_frame_store(
        config,
        tables,
        config_hash=config_hash,
        model_fingerprints=model_fingerprints,
        limited_run=limit is not None,
        resume_enabled=checkpoint_resume,
    )
