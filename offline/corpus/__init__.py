"""Canonical local corpus preparation for V3C/VBS."""

from .frames import FrameBuildConfig, FrameBuildReport, build_frames, discover_videos, probe_video
from .paths import VBSDataPaths
from .models import FrameArtifact
from .keyframe_map import (
    load_btc_keyframe_map,
    join_btc_mapping,
    project_keyframe_paths,
)

__all__ = [
    "FrameBuildConfig",
    "FrameBuildReport",
    "FrameArtifact",
    "VBSDataPaths",
    "build_frames",
    "discover_videos",
    "probe_video",
    "load_btc_keyframe_map",
    "join_btc_mapping",
    "project_keyframe_paths",
]
