"""Content-addressed KIS query image asset store.

This module owns the persistent storage of user-uploaded query images.
Images are validated on intake (MIME type, byte limit, pixel limit, animated
rejection) and written atomically to disk keyed by their SHA-256 digest.

This module does NOT perform retrieval, embedding, or any LLM call.
It does not expose filesystem paths to callers — only opaque asset IDs.
"""

from __future__ import annotations

import hashlib
import io
import re
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, UnidentifiedImageError

from hcmai.kis.models import KISImageRef

if TYPE_CHECKING:
    pass

# Canonical ID pattern: sha256:<64 lowercase hex chars>
_ASSET_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

# Supported MIME types and their normalised file extensions
_MIME_TO_EXT: dict[str, str] = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}

# Reverse map: stored extension -> normalised content_type
_EXT_TO_MIME: dict[str, str] = {v: k for k, v in _MIME_TO_EXT.items()}


class InvalidImageError(ValueError):
    """The uploaded payload is not a supported, safely bounded static image."""


class KISImageAssetStore:
    """Content-addressed persistent store for user-uploaded KIS query images.

    Each accepted image is written exactly once under a SHA-256-keyed filename.
    Duplicate uploads are silently deduplicated — the same payload always yields
    the same ``KISImageRef``.

    ``asset_id`` format: ``sha256:<64 lowercase hex characters>``.

    Filesystem paths are never exposed to callers.
    """

    SUPPORTED_MEDIA_TYPES: frozenset[str] = frozenset(_MIME_TO_EXT.keys())

    def __init__(
        self,
        storage_dir: Path,
        *,
        max_upload_bytes: int,
        max_pixels: int,
    ) -> None:
        """Bind storage root and upload limits.

        Parameters
        ----------
        storage_dir:
            Directory where validated image files are stored. Created on first
            write if it does not exist yet.
        max_upload_bytes:
            Maximum raw byte length accepted before any decoding.
        max_pixels:
            Maximum decoded pixel count (width × height).
        """

        if max_upload_bytes <= 0 or max_pixels <= 0:
            raise ValueError("upload limits must be positive")

        self._dir = storage_dir
        self._max_upload_bytes = max_upload_bytes
        self._max_pixels = max_pixels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put(self, payload: bytes, content_type: str | None) -> KISImageRef:
        """Validate and persist one image, returning its stable ref.

        Duplicate payloads are deduplicated: identical bytes always return the
        same ``KISImageRef`` without writing a second file.

        Raises
        ------
        InvalidImageError
            When the payload fails MIME, size, pixel, or animated validation.
        """

        if not content_type or content_type not in self.SUPPORTED_MEDIA_TYPES:
            raise InvalidImageError(
                f"image must use JPEG, PNG, or WebP media type; got {content_type!r}"
            )
        if not payload:
            raise InvalidImageError("image payload must not be empty")
        if len(payload) > self._max_upload_bytes:
            raise InvalidImageError(
                f"image payload exceeds {self._max_upload_bytes} bytes"
            )

        # Validate the decoded image (pixel limit, animated rejection, etc.)
        self._decode_validate(payload)

        # Identify the actual format from decoded bytes (trust PIL, not caller)
        fmt = self._detect_format(payload)
        ext = _MIME_TO_EXT[fmt]
        normalised_ct = fmt  # e.g. "image/png"

        digest = hashlib.sha256(payload).hexdigest()
        asset_id = f"sha256:{digest}"
        target = self._target_path(digest, ext)

        if not target.exists():
            self._atomic_write(payload, target)

        return KISImageRef(asset_id=asset_id, content_type=normalised_ct)

    def ref(self, asset_id: str) -> KISImageRef:
        """Resolve a known asset_id to its ``KISImageRef``.

        Raises
        ------
        KeyError
            When the asset_id is unknown or malformed.
        """

        self._validate_id(asset_id)
        target = self._find_stored(asset_id)
        ext = target.suffix.lstrip(".")
        ct = _EXT_TO_MIME[ext]
        return KISImageRef(asset_id=asset_id, content_type=ct)

    def open(self, asset_id: str) -> Image.Image:
        """Return a detached RGB PIL image for a known asset.

        The returned image is detached from any file handle — the store's file
        is closed before this method returns.

        Raises
        ------
        KeyError
            When the asset_id is unknown or malformed.
        InvalidImageError
            When the stored file can no longer be decoded (should not happen in
            normal operation).
        """

        self._validate_id(asset_id)
        target = self._find_stored(asset_id)
        payload = target.read_bytes()
        return self._decode_validate(payload)

    def read(self, asset_id: str) -> tuple[bytes, str]:
        """Return the stored original bytes and normalised content type.

        Raises
        ------
        KeyError
            When the asset_id is unknown or malformed.
        """

        self._validate_id(asset_id)
        target = self._find_stored(asset_id)
        ext = target.suffix.lstrip(".")
        ct = _EXT_TO_MIME[ext]
        return target.read_bytes(), ct

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_id(self, asset_id: str) -> None:
        """Raise ValueError for malformed IDs (prevents path traversal)."""

        if not isinstance(asset_id, str) or not _ASSET_ID_RE.match(asset_id):
            raise ValueError(
                f"asset_id must match sha256:<64 hex chars>; got {asset_id!r}"
            )

    def _target_path(self, digest: str, ext: str) -> Path:
        return self._dir / f"{digest}.{ext}"

    def _find_stored(self, asset_id: str) -> Path:
        """Find the stored file for a validated asset_id.

        Raises
        ------
        KeyError
            When no file with the expected digest prefix is found.
        ValueError
            When the asset_id is malformed (already validated by callers, but
            defensive).
        """

        self._validate_id(asset_id)
        digest = asset_id[len("sha256:"):]
        for ext in _MIME_TO_EXT.values():
            candidate = self._target_path(digest, ext)
            if candidate.exists():
                return candidate
        raise KeyError(asset_id)

    def _decode_validate(self, payload: bytes) -> Image.Image:
        """Decode payload, enforce pixel + animated limits, return RGB image.

        Mirrors the validation in ``ImageSearchService._decode`` so both paths
        enforce the same safety invariants.
        """

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(payload)) as source:
                    width, height = source.size
                    if width <= 0 or height <= 0:
                        raise InvalidImageError("image dimensions must be positive")
                    if width * height > self._max_pixels:
                        raise InvalidImageError(
                            f"image exceeds {self._max_pixels} decoded pixels"
                        )
                    if getattr(source, "n_frames", 1) != 1:
                        raise InvalidImageError("animated images are not supported")
                    return source.convert("RGB")
        except InvalidImageError:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as err:
            raise InvalidImageError("image exceeds the safe decoded size") from err
        except UnidentifiedImageError as err:
            raise InvalidImageError("payload is not a recognised image format") from err

    def _detect_format(self, payload: bytes) -> str:
        """Return the normalised MIME type detected from the decoded image bytes.

        Uses PIL's format detection so the MIME used to name the stored file
        matches the actual encoding, not the caller's claim.
        """

        with Image.open(io.BytesIO(payload)) as img:
            fmt = (img.format or "").upper()

        mapping = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
        if fmt not in mapping:
            raise InvalidImageError(
                f"detected image format {fmt!r} is not supported"
            )
        return mapping[fmt]

    def _atomic_write(self, payload: bytes, target: Path) -> None:
        """Write bytes to target atomically using a temp file + rename."""

        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=target.parent, delete=False, suffix=".tmp"
        ) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)

        # Atomic replace — won't leave a partial file on crash
        tmp_path.replace(target)
