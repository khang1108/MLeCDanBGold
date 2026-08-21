"""VQA query interpretation and answer normalization."""

from .normalization import normalize_answer
from .parser import parse_vqa_query

__all__ = ["normalize_answer", "parse_vqa_query"]
