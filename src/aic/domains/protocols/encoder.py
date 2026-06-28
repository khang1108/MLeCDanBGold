"""
A Protocol for Encoder.
"""

from __future__ import annotations

from typing import Protocol


class Encoder(Protocol):
    """
    Unified Protocol for Encoder in this project.
    """

    def encode(self, data: list[str]) -> list[list[float]]:
        """Encode the data to a vector embedding. The data can be a list of ``link_of_image``
            or ``text``.

        Args:
            data (List[str]): A list of links_to_img or text.

        Returns:
            A vector embedding
        """
        ...
