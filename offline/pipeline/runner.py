"""Local-first stage orchestration for V3C/VBS preparation."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from offline.config import VBSConfig
from .state import update_stage

STAGES = (
    "frames", "visual-index", "caption", "ocr", "objects", "asr",
    "context", "context-index", "asr-index", "indexes",
)


def _manifest_counts(config: VBSConfig) -> tuple[int, int]:
    path = config.paths.artifacts_root / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    return int(value["video_count"]), int(value["frame_count"])


def _index_command(config_path: Path, config: VBSConfig, stage: str) -> list[str]:
    video_count, frame_count = _manifest_counts(config)
    work = config.paths.work_root
    artifacts = config.paths.artifacts_root
    index_stage = {"visual-index": "visual", "context-index": "context", "asr-index": "asr", "indexes": "all"}[stage]
    return [
        sys.executable, "-m", "scripts.indexing.build_retrieval_indexes",
        "--config", str(config_path),
        "--model-config", str(config_path),
        "--stage", index_stage,
        "--version", "v3c-v1",
        "--source", "v3c_local_video",
        "--frame-store-id", "v3c-v1",
        "--data-root", str(work),
        "--frames", str(artifacts / "frames.parquet"),
        "--frame-store-output", str(artifacts),
        "--frame-manifest", str(artifacts / "manifest.json"),
        "--context", str(artifacts / "context" / "frame_context_v1.parquet"),
        "--transcripts", str(artifacts / "asr"),
        "--expected-video-count", str(video_count),
        "--expected-frame-count", str(frame_count),
        "--output-root", str(config.paths.indexes_root),
        "--inference-url", config.core.base_url,
    ]


def command_for(stage: str, config_path: Path, config: VBSConfig) -> list[str]:
    if stage == "frames":
        return [sys.executable, "-m", "scripts.corpus.prepare_vbs_frames", "--config", str(config_path)]
    if stage == "caption":
        return [
            sys.executable, "-m", "scripts.enrichment.generate_enrichment",
            "--config", str(config_path),
            "--execution-backend", "remote",
            "--image-workers", "8",
        ]
    if stage == "ocr":
        return [sys.executable, "-m", "scripts.enrichment.generate_ocr_enrichment", "--config", str(config_path)]
    if stage == "objects":
        return [sys.executable, "-m", "scripts.enrichment.prepare_vbs_objects", "--config", str(config_path)]
    if stage == "asr":
        return [sys.executable, "-m", "scripts.enrichment.prepare_vbs_transcripts", "--config", str(config_path)]
    if stage == "context":
        return [sys.executable, "-m", "scripts.enrichment.build_frame_context", "--config", str(config_path)]
    if stage in {"visual-index", "context-index", "asr-index", "indexes"}:
        return _index_command(config_path, config, stage)
    raise ValueError(f"Unknown preparation stage: {stage}")


def run_stage(stage: str, config_path: str | Path = "configs/vbs_prepare.yaml") -> None:
    path = Path(config_path).expanduser().resolve()
    config = VBSConfig.from_yaml(path)
    state_file = config.paths.state_root / "pipeline.json"
    command = command_for(stage, path, config)
    env = os.environ.copy()
    env.setdefault("VBS_CORE_API_URL", config.core.base_url)
    env.setdefault("VBS_GPU_API_URL", config.gpu.base_url)
    update_stage(state_file, stage, "running")
    try:
        subprocess.run(command, cwd=path.parent.parent, env=env, check=True)
    except Exception as error:
        update_stage(state_file, stage, "failed", detail=str(error))
        raise
    update_stage(state_file, stage, "done")


def run_sequence(stages: Sequence[str], config_path: str | Path = "configs/vbs_prepare.yaml") -> None:
    for stage in stages:
        run_stage(stage, config_path)
