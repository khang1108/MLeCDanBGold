"""Shared HTTP transport for inference providers.

This module encapsulates low-level JSON POST requests using httpx, keeping wire
communication decoupled from inference capability business logic.
"""

from __future__ import annotations

from typing import Any
import httpx


class HttpTransport:
    """Synchronous JSON HTTP transport using httpx."""

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
            httpx.HTTPError: If network fails or server returns a non-2xx status code.
        """
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
