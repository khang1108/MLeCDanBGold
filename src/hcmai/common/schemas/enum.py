from __future__ import annotations

from enum import Enum

from hcmai.common.schemas import NonEmptyString, ContractModel

class SearchMode(str, Enum):
    """Supported Search Profiles

    Args:
        FAST: We use this mode only when we are at the Finalist Round :33
        ACCURATE: Normal mode, set as Default
    """

    FAST = "fast"
    ACCURATE = "accuracte"


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
    VQA = "vqa"
    TRAKE = "trake"

class QueryDifficulty(str, Enum):
    """Human-assigned difficulty of an evaluation query."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"