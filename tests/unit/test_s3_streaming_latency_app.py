from __future__ import annotations

from io import BytesIO

import pytest

from scripts.s3_streaming_latency_app import (
    browser_player_html,
    build_presigned_video_url,
    probe_range,
)


class FakeBody(BytesIO):
    pass


class FakeS3Client:
    def __init__(self, payload: bytes = b"video") -> None:
        self.payload = payload
        self.get_kwargs = None
        self.presign_kwargs = None

    def head_object(self, **kwargs):
        return {"ContentLength": 123_456}

    def get_object(self, **kwargs):
        self.get_kwargs = kwargs
        return {
            "Body": FakeBody(self.payload),
            "ContentRange": f"bytes 0-{len(self.payload) - 1}/123456",
            "ContentType": "binary/octet-stream",
        }

    def generate_presigned_url(self, *args, **kwargs):
        self.presign_kwargs = (args, kwargs)
        return "https://example.invalid/video?signature=secret"


def test_probe_range_reads_payload_and_requests_expected_range() -> None:
    client = FakeS3Client(b"0123456789")

    result = probe_range(client, bucket="bucket", key="video.mp4", range_bytes=10, chunk_bytes=3)

    assert client.get_kwargs == {"Bucket": "bucket", "Key": "video.mp4", "Range": "bytes=0-9"}
    assert result.bytes_read == 10
    assert result.object_size_bytes == 123_456
    assert result.content_range == "bytes 0-9/123456"
    assert result.total_ms >= result.time_to_first_byte_ms >= result.response_headers_ms >= 0
    assert result.throughput_mbps > 0


@pytest.mark.parametrize("range_bytes,chunk_bytes", [(0, 1), (1, 0), (-1, 1)])
def test_probe_range_rejects_invalid_sizes(range_bytes: int, chunk_bytes: int) -> None:
    with pytest.raises(ValueError):
        probe_range(
            FakeS3Client(),
            bucket="bucket",
            key="video.mp4",
            range_bytes=range_bytes,
            chunk_bytes=chunk_bytes,
        )


def test_presigned_url_overrides_binary_content_type_for_browser() -> None:
    client = FakeS3Client()

    url = build_presigned_video_url(
        client,
        bucket="bucket",
        key="video.mp4",
        expires_seconds=900,
    )

    assert url.startswith("https://")
    assert client.presign_kwargs == (
        ("get_object",),
        (
            {
                "Params": {
                    "Bucket": "bucket",
                    "Key": "video.mp4",
                    "ResponseContentType": "video/mp4",
                },
                "ExpiresIn": 900,
            }
        ),
    )


def test_browser_html_escapes_url_and_includes_latency_events() -> None:
    markup = browser_player_html('https://example.invalid/video?a=1&x="bad"')

    assert 'data-url="https://example.invalid/video?a=1&amp;x=&quot;bad&quot;"' in markup
    assert '<video id="video" controls muted playsinline preload="none"></video>' in markup
    assert 'preload="none"' in markup
    assert "video.src = startButton.dataset.url" in markup
    assert "startButton.addEventListener" in markup
    assert "loadeddata" in markup
    assert "playing" in markup
    assert "requestVideoFrameCallback" in markup
    assert "rebufferCount" in markup
    assert "video.buffered" in markup
