"""Scoped S3 audio references for remote ASR and diarization."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Callable

from hcmai.common.schemas import AudioReferenceRequest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_flac(video: Path, output: Path, sample_rate: int) -> None:
    """Trích xuất âm thanh từ video sang định dạng lossless FLAC đơn kênh (mono).
    Sử dụng FFMPEG và ghi ra file tạm trước khi đổi tên để đảm bảo tính nguyên vẹn (atomic write),
    tránh file bị lỗi nếu quá trình bị ngắt giữa chừng.
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(f"{output.suffix}.partial")
    partial.unlink(missing_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-c:a",
                "flac",
                str(partial),
            ],
            check=True,
            capture_output=True,
        )
        if not partial.is_file() or partial.stat().st_size == 0:
            raise RuntimeError("ffmpeg did not produce an audio artifact")
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)


class S3AudioReferenceProvider:
    """Cung cấp URL tạm thời (Presigned URL) cho file Audio trên S3.
    
    Flow hoạt động:
    1. Trích xuất âm thanh FLAC từ file video local (nếu chưa có sẵn).
    2. Upload file FLAC này lên một thư mục tạm trên S3.
    3. Tạo Presigned URL giới hạn thời gian (mặc định 1 giờ) gửi cho worker.
    
    Việc này giúp các remote inference worker lấy được âm thanh trực tiếp
    mà không cần tải lại toàn bộ file MP4 gốc cồng kềnh.
    """

    def __init__(
        self,
        client: Any,
        *,
        bucket: str,
        prefix: str,
        work_root: Path,
        expires_seconds: int = 3600,
        extractor: Callable[[Path, Path, int], None] = extract_flac,
    ) -> None:
        if expires_seconds < 60 or expires_seconds > 86_400:
            raise ValueError("audio URL expiry must be within 60..86400 seconds")
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.work_root = work_root
        self.expires_seconds = expires_seconds
        self.extractor = extractor
        self._objects: dict[tuple[str, int], tuple[str, str]] = {}

    def reference(
        self, video_path: Path, video_id: str, sample_rate: int
    ) -> AudioReferenceRequest:
        """
        Tạo một AudioReferenceRequest cho video_path.

        Args:
            video_path (Path): Đường dẫn đến file video.
            video_id (str): ID của video.
            sample_rate (int): Tần số lấy mẫu âm thanh.

        Returns:
            AudioReferenceRequest: AudioReferenceRequest cho video_path.

        Raises:
            FileNotFoundError: Nếu audio source does not match canonical video_id.
        """
        video = video_path.expanduser().resolve()
        if not video.is_file() or video.stem != video_id:
            raise FileNotFoundError("audio source does not match canonical video_id")

        cache_key = (str(video), sample_rate)
        value = self._objects.get(cache_key)

        if value is None:
            source = _sha256(video)
            audio = self.work_root / "audio" / f"{source}-{sample_rate}.flac"

            if not audio.is_file():
                self.extractor(video, audio, sample_rate)
            
            digest = _sha256(audio)
            key = f"{self.prefix}/temporary-audio/{digest}.flac"
            
            self.client.upload_file(str(audio), self.bucket, key)
            
            remote = self.client.head_object(Bucket=self.bucket, Key=key)
            
            if int(remote["ContentLength"]) != audio.stat().st_size:
                raise OSError("uploaded temporary audio size mismatch")
            
            value = (key, digest)
            self._objects[cache_key] = value
        
        key, digest = value
        url = self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=self.expires_seconds,
        )
        return AudioReferenceRequest(
            request_id=f"audio-{digest}",
            video_id=video_id,
            audio_url=url,
            audio_sha256=digest,
            sample_rate=sample_rate,
        )

    def cleanup(self) -> None:
        """Dọn dẹp (xóa) toàn bộ các file âm thanh tạm trên S3 do object này tạo ra."""

        for key, _ in sorted(set(self._objects.values())):
            self.client.delete_object(Bucket=self.bucket, Key=key)
        self._objects.clear()
