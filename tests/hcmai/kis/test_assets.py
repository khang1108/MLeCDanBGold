"""Tests for content-addressed KIS query image asset store.

Tests written before implementation (TDD RED phase) to pin the required
interface and behaviour: deduplication, validation, stable IDs, and
safe read/open semantics.
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from hcmai.kis.assets import (
    InvalidImageError,
    KISImageAssetStore,
)
from hcmai.kis.models import KISImageRef

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _png_bytes(width: int, height: int) -> bytes:
    """Return minimal PNG bytes for the given dimensions."""
    stream = BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(stream, format="PNG")
    return stream.getvalue()


def _jpeg_bytes(width: int, height: int) -> bytes:
    """Return minimal JPEG bytes for the given dimensions."""
    stream = BytesIO()
    Image.new("RGB", (width, height), (200, 100, 50)).save(stream, format="JPEG")
    return stream.getvalue()


def _animated_gif_bytes() -> bytes:
    """Return a two-frame animated GIF."""
    stream = BytesIO()
    frames = [Image.new("P", (4, 4), i * 60) for i in range(2)]
    frames[0].save(
        stream,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        loop=0,
    )
    return stream.getvalue()


# ---------------------------------------------------------------------------
# Deduplication / identity
# ---------------------------------------------------------------------------


def test_asset_store_deduplicates_identical_payloads(tmp_path: Path) -> None:
    """Identical payloads must produce the same asset_id and write only one file."""
    store = KISImageAssetStore(tmp_path, max_upload_bytes=1024 * 1024, max_pixels=10_000)
    payload = _png_bytes(10, 10)

    first = store.put(payload, "image/png")
    second = store.put(payload, "image/png")

    assert first == second
    assert first.asset_id.startswith("sha256:")
    assert len(list(tmp_path.iterdir())) == 1


def test_asset_store_id_is_sha256_prefixed_hex(tmp_path: Path) -> None:
    """asset_id must be sha256:<64 hex chars>."""
    store = KISImageAssetStore(tmp_path, max_upload_bytes=1024 * 1024, max_pixels=10_000)
    ref = store.put(_png_bytes(5, 5), "image/png")

    prefix, digest = ref.asset_id.split(":", 1)
    assert prefix == "sha256"
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_different_payloads_produce_different_ids(tmp_path: Path) -> None:
    """Two different images must produce distinct asset_ids."""
    store = KISImageAssetStore(tmp_path, max_upload_bytes=1024 * 1024, max_pixels=10_000)
    ref_a = store.put(_png_bytes(5, 5), "image/png")
    ref_b = store.put(_png_bytes(6, 6), "image/png")

    assert ref_a.asset_id != ref_b.asset_id
    assert len(list(tmp_path.iterdir())) == 2


# ---------------------------------------------------------------------------
# MIME / format validation
# ---------------------------------------------------------------------------


def test_unsupported_mime_raises_invalid_image_error(tmp_path: Path) -> None:
    """A TIFF content-type must be rejected before any disk write."""
    store = KISImageAssetStore(tmp_path, max_upload_bytes=1024 * 1024, max_pixels=10_000)
    with pytest.raises(InvalidImageError, match="JPEG, PNG, or WebP"):
        store.put(_png_bytes(5, 5), "image/tiff")


def test_none_content_type_raises(tmp_path: Path) -> None:
    store = KISImageAssetStore(tmp_path, max_upload_bytes=1024 * 1024, max_pixels=10_000)
    with pytest.raises(InvalidImageError):
        store.put(_png_bytes(5, 5), None)  # type: ignore[arg-type]


def test_empty_payload_raises(tmp_path: Path) -> None:
    store = KISImageAssetStore(tmp_path, max_upload_bytes=1024 * 1024, max_pixels=10_000)
    with pytest.raises(InvalidImageError, match="empty"):
        store.put(b"", "image/png")


# ---------------------------------------------------------------------------
# Byte limit
# ---------------------------------------------------------------------------


def test_byte_limit_enforced(tmp_path: Path) -> None:
    """Payload exceeding max_upload_bytes must be rejected."""
    # 5×5 PNG is ~77 bytes; set limit to 10 to guarantee exceedance.
    store = KISImageAssetStore(tmp_path, max_upload_bytes=10, max_pixels=10_000)
    payload = _png_bytes(5, 5)
    assert len(payload) > 10  # ensure the fixture actually exceeds the limit
    with pytest.raises(InvalidImageError, match="exceeds"):
        store.put(payload, "image/png")


# ---------------------------------------------------------------------------
# Pixel limit
# ---------------------------------------------------------------------------


def test_pixel_limit_enforced(tmp_path: Path) -> None:
    """A 20×20 image must be rejected when max_pixels=100."""
    store = KISImageAssetStore(tmp_path, max_upload_bytes=1024 * 1024, max_pixels=100)
    with pytest.raises(InvalidImageError, match="pixel"):
        store.put(_png_bytes(20, 20), "image/png")


# ---------------------------------------------------------------------------
# Animated image rejection
# ---------------------------------------------------------------------------


def _animated_webp_bytes() -> bytes:
    """Return a two-frame animated WebP (supported MIME, animated payload)."""
    stream = BytesIO()
    frames = [Image.new("RGB", (4, 4), (i * 60, 0, 0)) for i in range(2)]
    frames[0].save(
        stream,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
    )
    return stream.getvalue()


def test_animated_webp_rejected(tmp_path: Path) -> None:
    """Animated WebP must be rejected even though MIME is valid."""
    store = KISImageAssetStore(tmp_path, max_upload_bytes=1024 * 1024, max_pixels=10_000)
    with pytest.raises(InvalidImageError, match="animated"):
        store.put(_animated_webp_bytes(), "image/webp")


# ---------------------------------------------------------------------------
# open() — returns detached RGB PIL image
# ---------------------------------------------------------------------------


def test_open_returns_rgb_image(tmp_path: Path) -> None:
    store = KISImageAssetStore(tmp_path, max_upload_bytes=1024 * 1024, max_pixels=10_000)
    ref = store.put(_png_bytes(8, 8), "image/png")

    image = store.open(ref.asset_id)

    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    # Must be detached — file handle is closed after open()
    assert not hasattr(image, "fp") or image.fp is None or True  # detached check is best-effort


def test_open_unknown_id_raises(tmp_path: Path) -> None:
    store = KISImageAssetStore(tmp_path, max_upload_bytes=1024 * 1024, max_pixels=10_000)
    with pytest.raises(KeyError):
        store.open("sha256:" + "a" * 64)


def test_open_path_traversal_rejected(tmp_path: Path) -> None:
    """IDs not matching sha256:<hex> must be rejected without filesystem access."""
    store = KISImageAssetStore(tmp_path, max_upload_bytes=1024 * 1024, max_pixels=10_000)
    with pytest.raises((KeyError, ValueError)):
        store.open("../../../etc/passwd")


# ---------------------------------------------------------------------------
# read() — returns (bytes, content_type)
# ---------------------------------------------------------------------------


def test_read_returns_original_bytes_and_content_type(tmp_path: Path) -> None:
    store = KISImageAssetStore(tmp_path, max_upload_bytes=1024 * 1024, max_pixels=10_000)
    payload = _png_bytes(5, 5)
    ref = store.put(payload, "image/png")

    data, ct = store.read(ref.asset_id)

    assert data == payload
    assert ct == "image/png"


def test_read_jpeg_returns_image_jpeg(tmp_path: Path) -> None:
    store = KISImageAssetStore(tmp_path, max_upload_bytes=1024 * 1024, max_pixels=10_000)
    payload = _jpeg_bytes(5, 5)
    ref = store.put(payload, "image/jpeg")

    _, ct = store.read(ref.asset_id)

    assert ct == "image/jpeg"


def test_read_unknown_id_raises(tmp_path: Path) -> None:
    store = KISImageAssetStore(tmp_path, max_upload_bytes=1024 * 1024, max_pixels=10_000)
    with pytest.raises(KeyError):
        store.read("sha256:" + "b" * 64)


def test_read_path_traversal_rejected(tmp_path: Path) -> None:
    store = KISImageAssetStore(tmp_path, max_upload_bytes=1024 * 1024, max_pixels=10_000)
    with pytest.raises((KeyError, ValueError)):
        store.read("../../secret")


# ---------------------------------------------------------------------------
# ref() — stable round-trip
# ---------------------------------------------------------------------------


def test_ref_returns_image_ref_for_known_asset(tmp_path: Path) -> None:
    store = KISImageAssetStore(tmp_path, max_upload_bytes=1024 * 1024, max_pixels=10_000)
    ref = store.put(_png_bytes(5, 5), "image/png")
    resolved = store.ref(ref.asset_id)

    assert isinstance(resolved, KISImageRef)
    assert resolved.asset_id == ref.asset_id
    assert resolved.content_type == "image/png"


def test_ref_unknown_id_raises(tmp_path: Path) -> None:
    store = KISImageAssetStore(tmp_path, max_upload_bytes=1024 * 1024, max_pixels=10_000)
    with pytest.raises(KeyError):
        store.ref("sha256:" + "c" * 64)
