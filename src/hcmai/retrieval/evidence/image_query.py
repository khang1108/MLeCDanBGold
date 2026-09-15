"""Score visual frame evidence using query image exemplars.

This module encodes user-provided query images with the image encoder and scores
them against the canonical visual index. Multiple image exemplars attached to
the same semantic event are aggregated using element-wise maximum pooling.
Events with no image exemplars produce a zero row.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import numpy as np

from hcmai.kis.models import KISImageRef
from hcmai.retrieval.evidence.components import TemporalScoreComponent


class ImageQueryTemporalScorer:
    """Score full-corpus visual frames using user-provided query images."""

    def __init__(
        self,
        *,
        visual_index: Any,
        image_encoder: Any,
        asset_store: Any,
        chunk_size: int = 65_536,
    ) -> None:
        """Bind visual index, image encoder, and asset store.

        Args:
            visual_index: Canonical visual index exposing frame_ids and score_subset().
            image_encoder: Model encoder exposing encode_images().
            asset_store: Image asset store exposing open(asset_id).
            chunk_size: Processing batch size for index scoring.
        """
        self.visual_index = visual_index
        self.image_encoder = image_encoder
        self.asset_store = asset_store
        self.chunk_size = chunk_size

    def score_events(
        self,
        event_images: Sequence[Sequence[KISImageRef]],
    ) -> TemporalScoreComponent:
        """Score canonical frames against query images for each event.

        Args:
            event_images: Sequence of image reference sequences, one per event.

        Returns:
            TemporalScoreComponent named "visual_image" of shape [event_count, frame_count].
        """
        event_count = len(event_images)
        frame_count = len(self.visual_index.frame_ids)
        raw_scores = np.zeros((event_count, frame_count), dtype=np.float32)

        # Map flat images to events
        all_images: list[Any] = []
        event_slice_map: list[tuple[int, int]] = []

        for refs in event_images:
            start_idx = len(all_images)
            for ref in refs:
                img = self.asset_store.open(ref.asset_id)
                all_images.append(img)
            end_idx = len(all_images)
            event_slice_map.append((start_idx, end_idx))

        if not all_images:
            return TemporalScoreComponent(name="visual_image", raw_scores=raw_scores)

        # Batch encode all images
        vectors = np.asarray(self.image_encoder.encode_images(all_images), dtype=np.float32)
        positions = np.arange(frame_count, dtype=np.int64)

        # Batch score against visual index
        scores = self.visual_index.score_subset(vectors, positions, self.chunk_size)

        # Max-pool across exemplars per event
        for event_idx, (start_idx, end_idx) in enumerate(event_slice_map):
            if start_idx < end_idx:
                raw_scores[event_idx] = np.max(scores[start_idx:end_idx], axis=0)

        return TemporalScoreComponent(name="visual_image", raw_scores=raw_scores)
