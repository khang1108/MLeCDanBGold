"""Stable EventTrail domain error codes."""


class EventTrailError(RuntimeError):
    """Domain error with stable error code for EventTrail operations."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
