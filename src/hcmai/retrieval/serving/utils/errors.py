"""Typed error taxonomy for HTTP-based retrieval transport and status mapping."""

from __future__ import annotations


class RetrievalClientError(Exception):
    """Base exception for all retrieval client failures."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code


class RetrievalUnavailableError(RetrievalClientError):
    """Retrieval service is unreachable, not ready, or timed out."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, category="unavailable", status_code=status_code)


class RetrievalInvalidRequestError(RetrievalClientError):
    """Request arguments are invalid or malformed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, category="invalid_argument", status_code=status_code)


class RetrievalNotFoundError(RetrievalClientError):
    """Requested resource (e.g. video_id) was not found."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, category="not_found", status_code=status_code)


class RetrievalTooLargeError(RetrievalClientError):
    """Uploaded payload or request exceeds configured size limits."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, category="too_large", status_code=status_code)


class RetrievalInternalError(RetrievalClientError):
    """Retrieval service encountered an internal runtime error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, category="internal", status_code=status_code)


class RetrievalProtocolError(RetrievalClientError):
    """Internal server error or invalid response protocol."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, category="protocol", status_code=status_code)


def map_http_error(status_code: int, message: str) -> RetrievalClientError:
    """Map HTTP status codes to typed RetrievalClientError subclasses."""
    if status_code in (404,):
        return RetrievalNotFoundError(message, status_code=status_code)
    if status_code in (400, 422):
        return RetrievalInvalidRequestError(message, status_code=status_code)
    if status_code in (413,):
        return RetrievalTooLargeError(message, status_code=status_code)
    if status_code in (502, 503, 504):
        return RetrievalUnavailableError(message, status_code=status_code)
    return RetrievalInternalError(message, status_code=status_code)
