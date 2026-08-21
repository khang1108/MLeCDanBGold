"""Các Adapters cho mô hình OCR.

Nơi chứa mã nguồn giao tiếp với các mô hình nhận diện chữ viết khác nhau."""

from .remote import RemoteOCRAdapter

__all__ = ["RemoteOCRAdapter"]
