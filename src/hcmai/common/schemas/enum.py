from __future__ import annotations

from enum import Enum

from hcmai.common.schemas import NonEmptyString, ContractModel


class ProcessingStatus(str, Enum):
    """Status of an offline processing operation."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RetrievalSource(str, Enum):
    """Evidence channels used to retrieve a frame."""

    VISUAL = "visual"
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
    KISC = "kisc"
    VKIS = "vkis"
    VQA = "vqa"
    TRAKE = "trake"


class QueryDifficulty(str, Enum):
    """Human-assigned difficulty of an evaluation query."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
