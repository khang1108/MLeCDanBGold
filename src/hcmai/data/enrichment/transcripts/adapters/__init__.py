"""Các Adapters cho Transcript.

Chứa các lớp giao tiếp trực tiếp với các mô hình AI cụ thể (ASR, Diarization)."""

from .remote import RemoteASRAdapter, RemoteDiarizationAdapter

__all__ = ["RemoteASRAdapter", "RemoteDiarizationAdapter"]
