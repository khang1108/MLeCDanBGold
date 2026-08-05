"""Provider-independent TRAKE query parser."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from pydantic import ValidationError

import re

from hcmai.common.schemas import TrakeParseResponse
from hcmai.common.utils.logging import get_logger

logger = get_logger(__name__)

StructuredCall = Callable[[dict[str, Any]], object]

# Events feed the SigLIP2 visual index, whose text tower is English-centric,
# so the parser translates like the KISC resolver already does.
_INSTRUCTION = """
Split one Vietnamese or English TRAKE video query into an ordered list of
atomic visual events. Keep the events in the exact chronological order the
query states and never reorder them. One event is one action or state a person
can recognize in a single keyframe: split consecutive actions into separate
events and never merge two actions that happen at different moments.

Drop ordering words when they only join two events: "đầu tiên", "sau đó",
"tiếp theo", "kế tiếp", "rồi", "cuối cùng", "lần lượt", "bước 1", "bước 2",
"first", "then", "next", "finally". Keep the same words when they belong to the
action itself, for example "cán đích đầu tiên" means finishing in first place.
Drop lead-ins with no visual content: "tìm video", "đoạn clip quay cảnh",
"cho tôi xem", "hãy trích xuất", "find the video", "show me".

Keep every detail that distinguishes a subject: color, clothing, gender,
object. Replace a pronoun such as "anh ấy", "người đó", "nó", "he", or "it"
with the concrete noun from the earlier event so each event reads standalone.
Never invent an object, action, setting, or event the query does not state.

Write every event in English as one short lowercase visual description with no
trailing punctuation. Translate faithfully and preserve names and numbers.

Return only JSON: {"events":["...","..."]}. No Markdown, no explanation.

Examples:

Input: "Tìm video quay cảnh đầu tiên một chiếc xe máy màu đỏ dừng trước vạch kẻ đường, sau đó người lái xe tháo mũ bảo hiểm ra, rồi quay sang nói chuyện với người đi bộ."
Output: {"events":["a red motorbike stops before the road marking line","the motorbike rider takes off the helmet","the motorbike rider turns to talk with a pedestrian"]}

Input: "1. Vận động viên chuẩn bị chạy | 2. Vận động viên bứt tốc trên đường chạy | 3. Vận động viên cán đích đầu tiên"
Output: {"events":["an athlete prepares to start running","the athlete sprints along the running track","the athlete crosses the finish line in first place"]}

Input: "Cảnh người nam mặc áo sơ mi trắng đi vào phòng làm việc, ngồi xuống ghế mở laptop lên gõ bàn phím."
Output: {"events":["a man in a white shirt walks into an office room","the man in a white shirt sits down on a chair","the man in a white shirt opens a laptop and types on the keyboard"]}
""".strip()

# The query pack separates events with "|"; nothing else is a confirmed
# delimiter, so no punctuation heuristic may split an event's own text.
_DELIMITER = re.compile(r"[|\n]")
_ORDINAL_PREFIX = re.compile(r"^(?:bước\s*)?\d+\s*[.)\-:]\s*", re.IGNORECASE)


class TrakeParserError(RuntimeError):
    """Bounded failure at the TRAKE parsing boundary."""


def split_delimited(query: str) -> list[str] | None:
    """Split an already-enumerated event list without calling a model."""
    parts = [
        _ORDINAL_PREFIX.sub("", part.strip()).strip()
        for part in _DELIMITER.split(query)
    ]
    parts = [part for part in parts if part]
    return parts if len(parts) > 1 else None


class TrakeQueryParser:
    """Split one TRAKE query into ordered English atomic events."""

    def __init__(self, structured_call: StructuredCall) -> None:
        self.structured_call = structured_call

    def parse(
        self, raw_query: str, event_count: int | None = None
    ) -> TrakeParseResponse:
        """Return the ordered events of one TRAKE query.

        Args:
            raw_query: One query as written in the query pack.
            event_count: Expected event count when the pack states it
                separately. Defaults to the count of "|"-delimited segments,
                which is authoritative for a batch run.

        Raises:
            TrakeParserError: If the query is empty or neither the provider nor
                the delimited split yields the expected event count. A row with
                the wrong count is invalid, so it never silently truncates.
        """
        query = raw_query.strip()
        if not query:
            raise TrakeParserError("raw_query must not be empty")
        if event_count is not None and event_count < 1:
            raise TrakeParserError("event_count must be at least 1")
        delimited = split_delimited(query)
        expected = event_count or (len(delimited) if delimited else None)
        try:
            events = self._from_model(query)
            if expected is not None and len(events) != expected:
                raise TrakeParserError(
                    f"model returned {len(events)} events, expected {expected}"
                )
        except TrakeParserError as error:
            # ponytail: the fallback keeps a batch row valid but leaves events
            # untranslated, so the English-only visual channel degrades. Remove
            # it once the provider has measured reliability.
            if delimited is None or (
                expected is not None and len(delimited) != expected
            ):
                raise
            logger.warning(
                "TRAKE parser fell back to delimiter split events=%d reason=%s",
                len(delimited),
                error,
            )
            events = delimited
        return TrakeParseResponse(events=events)

    def _from_model(self, query: str) -> list[str]:
        request = {"instruction": _INSTRUCTION, "raw_query": query}
        try:
            output = self.structured_call(request)
        except Exception as error:
            detail = " ".join(str(error).split())[:160]
            raise TrakeParserError(
                f"structured provider failed: {detail or type(error).__name__}"
            ) from error
        try:
            return TrakeParseResponse.model_validate(output).events
        except ValidationError as error:
            raise TrakeParserError(
                "structured output failed TrakeParseResponse validation"
            ) from error
