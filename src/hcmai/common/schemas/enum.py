"""Shared enumerations for HCMAI contracts.

Retrieval sources are additive evidence channels.  They are not an index
configuration policy; legacy specialist index handling remains explicit.
"""

from __future__ import annotations

from enum import Enum

class ProcessingStatus(str, Enum):
    """Status of an offline processing operation."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RetrievalSource(str, Enum):
    """Evidence channels used to retrieve a frame."""

    VISUAL = "visual"
    CONTEXT = "context"
    CAPTION = "caption"
    OCR = "ocr"
    ASR = "asr"


class QueryLanguage(str, Enum):
    """Languages represented in the development query set."""

    VIETNAMESE = "vi"
    ENGLISH = "en"
    MIXED = "mixed"


class TaskType(str, Enum):
    """Task type of each query."""

    KIS = "kis"
    TRAKE = "trake"


class QueryDifficulty(str, Enum):
    """Human-assigned difficulty of an evaluation query."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
