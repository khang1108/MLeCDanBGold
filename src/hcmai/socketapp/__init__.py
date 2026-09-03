"""Catalog and player support for local HCMAI source-video playback.

The package preserves canonical ``video_id`` values while mapping them to
validated files below one configured root. It does not download, transcode, or
infer metadata from videos, and it does not own Cloudflare credentials.
"""

from hcmai.socketapp.catalog import CatalogError, VideoCatalog, VideoEntry

__all__ = [
    "CatalogError",
    "VideoCatalog",
    "VideoEntry",
]
