"""Typed error taxonomy for retrieval gRPC transport and status mapping."""

from __future__ import annotations

import grpc


class RetrievalClientError(Exception):
    """Base exception for all retrieval client failures."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        status_code: grpc.StatusCode | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code


class RetrievalUnavailableError(RetrievalClientError):
    """Retrieval service is unreachable or RPC timed out."""

    def __init__(
        self,
        message: str,
        *,
        status_code: grpc.StatusCode | None = None,
    ) -> None:
        super().__init__(message, category="unavailable", status_code=status_code)


class RetrievalInvalidRequestError(RetrievalClientError):
    """Request arguments are invalid or malformed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: grpc.StatusCode | None = None,
    ) -> None:
        super().__init__(message, category="invalid_argument", status_code=status_code)


class RetrievalNotFoundError(RetrievalClientError):
    """Requested resource (e.g. video_id) was not found."""

    def __init__(
        self,
        message: str,
        *,
        status_code: grpc.StatusCode | None = None,
    ) -> None:
        super().__init__(message, category="not_found", status_code=status_code)


class RetrievalTooLargeError(RetrievalClientError):
    """Uploaded payload or request exceeds configured size limits."""

    def __init__(
        self,
        message: str,
        *,
        status_code: grpc.StatusCode | None = None,
    ) -> None:
        super().__init__(message, category="resource_exhausted", status_code=status_code)


class RetrievalProtocolError(RetrievalClientError):
    """Internal server error or invalid response protocol."""

    def __init__(
        self,
        message: str,
        *,
        status_code: grpc.StatusCode | None = None,
    ) -> None:
        super().__init__(message, category="protocol", status_code=status_code)


def map_rpc_error(error: grpc.RpcError) -> RetrievalClientError:
    """Map a grpc.RpcError to its corresponding typed client exception."""
    code = error.code() if hasattr(error, "code") else None
    raw_details = error.details() if hasattr(error, "details") else ""
    clean_details = str(raw_details or (code.name if code else "gRPC error"))

    if code in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
        return RetrievalUnavailableError(
            f"Retrieval service unavailable: {clean_details}",
            status_code=code,
        )
    if code is grpc.StatusCode.INVALID_ARGUMENT:
        return RetrievalInvalidRequestError(
            f"Invalid retrieval request: {clean_details}",
            status_code=code,
        )
    if code is grpc.StatusCode.NOT_FOUND:
        return RetrievalNotFoundError(
            f"Retrieval target not found: {clean_details}",
            status_code=code,
        )
    if code is grpc.StatusCode.RESOURCE_EXHAUSTED:
        return RetrievalTooLargeError(
            f"Retrieval payload too large: {clean_details}",
            status_code=code,
        )
    return RetrievalProtocolError(
        f"Retrieval protocol error: {clean_details}",
        status_code=code,
    )
