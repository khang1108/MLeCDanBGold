"""Shared HTTP transport for inference providers.

This module encapsulates low-level JSON POST requests using httpx and translates
transport/protocol failures into the typed inference error taxonomy.
"""

from __future__ import annotations

from typing import Any
import httpx

from hcmai.inference.errors import (
    InferenceAuthError,
    InferenceResponseError,
    InferenceUnavailableError,
)


class HttpTransport:
    """Synchronous JSON HTTP transport with typed error translation."""

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        """Send an HTTP POST request with JSON body and return the decoded JSON response.

        Args:
            url: Target endpoint URL.
            payload: JSON serializable request body.
            headers: HTTP headers including authentication and content type.
            timeout: Request timeout in seconds.

        Returns:
            Decoded JSON dictionary from the server.

        Raises:
            InferenceUnavailableError: On network connect, timeout, or unreachable endpoint.
            InferenceAuthError: On HTTP 401 or 403.
            InferenceResponseError: On non-2xx status codes or invalid JSON body.
        """
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload, headers=headers)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise InferenceUnavailableError(f"Inference endpoint unavailable at {url}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise InferenceUnavailableError(f"HTTP transport failed for {url}: {exc}") from exc

        if response.status_code in (401, 403):
            raise InferenceAuthError(
                f"Inference authentication failed ({response.status_code}) for {url}: {response.text[:200]}"
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise InferenceResponseError(
                f"Inference HTTP error {response.status_code} from {url}: {response.text[:200]}"
            ) from exc

        try:
            return response.json()
        except Exception as exc:
            raise InferenceResponseError(f"Inference endpoint returned invalid JSON: {exc}") from exc
