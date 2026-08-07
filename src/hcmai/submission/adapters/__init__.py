"""Remote submission provider adapters."""

from hcmai.submission.adapters.dres import DRESClient, DRESClientError

__all__ = ["DRESClient", "DRESClientError"]
