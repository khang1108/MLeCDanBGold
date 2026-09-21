from __future__ import annotations
from typing import Self
from pydantic import Field, JsonValue, model_validator
from .common import HTTPContract, NonEmptyString

class CaptionItem(HTTPContract):
    item_id: NonEmptyString
    caption: NonEmptyString

class CaptionResponse(HTTPContract):
    model: NonEmptyString
    revision: str | None = None
    items: list[CaptionItem]
    latency_ms: float = Field(ge=0)

class OCRRegionItem(HTTPContract):
    text: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)
    @model_validator(mode="after")
    def validate_box(self) -> Self:
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("OCR region maximums must not precede minimums")
        return self

class OCRItem(HTTPContract):
    item_id: NonEmptyString
    text: str
    raw_output: JsonValue | None = None
    regions: list[OCRRegionItem] = Field(default_factory=list)

class OCRResponse(HTTPContract):
    model: NonEmptyString
    revision: str | None = None
    items: list[OCRItem]
    latency_ms: float = Field(ge=0)

class ObjectItem(HTTPContract):
    item_id: NonEmptyString
    labels: list[NonEmptyString] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    boxes: list[list[float]] = Field(default_factory=list)
    @model_validator(mode="after")
    def validate_detections(self) -> Self:
        if len({len(self.labels), len(self.scores), len(self.boxes)}) != 1:
            raise ValueError("object detection arrays must have identical length")
        for score in self.scores:
            if not 0.0 <= score <= 1.0:
                raise ValueError("object detection scores must be in [0, 1]")
        for box in self.boxes:
            if len(box) != 4:
                raise ValueError("object boxes must contain four values")
            ymin, xmin, ymax, xmax = box
            if any(value < 0.0 or value > 1.0 for value in box):
                raise ValueError("object boxes must be normalized to [0, 1]")
            if ymin > ymax or xmin > xmax:
                raise ValueError("object box minimum exceeds maximum")
        return self

class ObjectResponse(HTTPContract):
    model: NonEmptyString
    revision: str | None = None
    items: list[ObjectItem]
    latency_ms: float = Field(ge=0)
