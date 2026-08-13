"""Canonical stage names shared by task pipelines and metrics."""

from __future__ import annotations

from enum import Enum


class PipelineStage(str, Enum):
    PARSE = "parse"
    EXPANSION = "expansion"
    ENCODE = "encode"
    SEARCH = "search"
    FUSION = "fusion"
    VIDEO_AGGREGATION = "video_aggregation"
    RERANK = "rerank"
    LOCALIZATION = "localization"
    ANSWER = "answer"
    MATERIALIZATION = "materialization"
