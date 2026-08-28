"""Local HTTP video origin used by the HCMAI frame inspector.

The package intentionally owns only local cataloging and byte-range delivery.
It does not download media, run model inference, manage Cloudflare
credentials, or expose the HCMAI retrieval database.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
