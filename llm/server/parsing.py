"""Bounded multipart decoding for inference HTTP endpoints.

The helpers validate transport shapes and resource limits. They do not invoke
models or assign canonical HCMAI identities.
"""

from __future__ import annotations

import io
import json

from fastapi import HTTPException, UploadFile
import numpy as np
from PIL import Image


def decode_images(
    item_ids: str,
    uploads: list[UploadFile],
    *,
    maximum: int,
) -> tuple[list[str], list[Image.Image]]:
    """Decode an aligned, unique, size-bounded multipart image batch."""

    try:
        identifiers = json.loads(item_ids)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail="item_ids must be JSON") from error
    if not isinstance(identifiers, list) or len(identifiers) != len(uploads):
        raise HTTPException(status_code=400, detail="item/image count mismatch")
    if not identifiers or len(identifiers) > maximum:
        raise HTTPException(
            status_code=400,
            detail=f"image batch must contain 1..{maximum}",
        )
    identifiers = [str(value).strip() for value in identifiers]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise HTTPException(
            status_code=400,
            detail="item_ids must be unique strings",
        )
    try:
        payloads = [upload.file.read(5_000_001) for upload in uploads]
        if any(len(value) > 5_000_000 for value in payloads):
            raise ValueError("candidate image exceeds 5 MB")
        decoded = [Image.open(io.BytesIO(value)).convert("RGB") for value in payloads]
    except Exception as error:
        raise HTTPException(status_code=400, detail="invalid candidate image") from error
    return identifiers, decoded


def decode_tensor(upload: UploadFile) -> np.ndarray:
    """Decode one bounded NPY frame tensor with shape ``[T,H,W,3]``."""

    try:
        payload = upload.file.read(64 * 1024 * 1024 + 1)
        if len(payload) > 64 * 1024 * 1024:
            raise ValueError("tensor exceeds 64 MiB")
        value = np.load(io.BytesIO(payload), allow_pickle=False)
        if value.ndim != 4 or value.shape[0] == 0 or value.shape[-1] != 3:
            raise ValueError("tensor must have shape [T,H,W,3]")
        if value.dtype not in {np.dtype("uint8"), np.dtype("float32")}:
            raise ValueError("tensor must use uint8 or float32")
        if value.dtype == np.float32 and not np.all(np.isfinite(value)):
            raise ValueError("tensor contains non-finite values")
        return np.ascontiguousarray(value)
    except Exception as error:
        raise HTTPException(status_code=400, detail="invalid frame tensor") from error
