"""Least-busy failover pool for semantically identical inference workers."""

from __future__ import annotations

from threading import Lock
from typing import Any, Callable, Protocol, Sequence

from hcmai.common.config import InferenceConfig
from hcmai.common.schemas import InferenceReadiness
from hcmai.llm.adapters.http import InferenceClient


class EndpointPoolConfig(Protocol):
    """Structural config contract that keeps the transport layer data-agnostic."""

    urls: Sequence[str]
    timeout_seconds: float
    connect_timeout_seconds: float
    read_timeout_seconds: float
    write_timeout_seconds: float
    pool_timeout_seconds: float
    max_attempts: int
    max_concurrency: int


class InferenceClientPool:
    """Quản lý các endpoint của remote GPU worker chạy cùng một mô hình.
    Sử dụng thuật toán "Least-busy" (chọn worker đang rảnh nhất) và hỗ trợ tự động failover (chuyển
    sang worker khác nếu bị lỗi) nhằm tối ưu hóa throughput trong hệ thống phân tán.
    """

    def __init__(self, clients: list[InferenceClient]) -> None:
        if not clients:
            raise ValueError("inference client pool must not be empty")
            
        self.clients = tuple(clients)
        # Mảng đếm số lượng request đang được xử lý (inflight) tại mỗi worker
        self._inflight = [0] * len(clients)
        # Khóa Lock để đảm bảo tính an toàn (thread-safe) khi cập nhật bộ đếm
        self._lock = Lock()

    @classmethod
    def from_config(cls, config: EndpointPoolConfig) -> InferenceClientPool:
        """Khởi tạo pool dựa trên cấu hình RemoteEndpointPoolConfig."""
        inference = InferenceConfig(
            enabled=True,
            timeout_seconds=config.timeout_seconds,
            connect_timeout_seconds=config.connect_timeout_seconds,
            read_timeout_seconds=config.read_timeout_seconds,
            write_timeout_seconds=config.write_timeout_seconds,
            pool_timeout_seconds=config.pool_timeout_seconds,
            max_attempts=config.max_attempts,
            max_concurrency=config.max_concurrency,
        )
        return cls([InferenceClient(url, inference) for url in config.urls])

    def readiness(self) -> InferenceReadiness:
        """Kiểm tra trạng thái sẵn sàng (health-check) của pool.
        Pool được xem là sẵn sàng nếu có ÍT NHẤT MỘT worker trả về trạng thái ready.
        """
        errors: list[Exception] = []
        for client in self.clients:
            try:
                value = client.readiness()
                if value.ready:
                    return value
            except Exception as error:  # endpoint failover boundary
                errors.append(error)
                
        raise RuntimeError("no inference endpoint is ready") from (
            errors[-1] if errors else None
        )

    def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Bộ định tuyến cốt lõi (Core router) cho mọi lời gọi API.
        
        Logic:
        1. Tìm worker rảnh nhất (ít request inflight nhất) và chưa được thử.
        2. Tăng bộ đếm inflight và thực thi gọi API.
        3. Nếu thành công -> Trả về kết quả và giảm bộ đếm.
        4. Nếu lỗi -> Lưu lỗi, giảm bộ đếm và lặp lại bước 1 với các worker còn lại.
        5. Nếu tất cả đều lỗi -> Báo lỗi cuối cùng (Fail-fast).
        """
        attempted: set[int] = set()
        last_error: Exception | None = None
        
        while len(attempted) < len(self.clients):
            # 1. Tìm worker rảnh nhất (least-busy)
            with self._lock:
                candidates = [
                    index for index in range(len(self.clients))
                    if index not in attempted
                ]
                index = min(candidates, key=self._inflight.__getitem__)
                self._inflight[index] += 1
                
            attempted.add(index)
            
            # 2. Thực thi gọi hàm tương ứng trên worker đã chọn
            try:
                method: Callable[..., Any] = getattr(self.clients[index], name)
                return method(*args, **kwargs)
                
            # 3. Lỗi (Network, Timeout, Server Error) -> Chuyển sang worker khác (Failover)
            except Exception as error:
                last_error = error
                
            # 4. Giải phóng bộ đếm sau khi hoàn tất (dù thành công hay thất bại)
            finally:
                with self._lock:
                    self._inflight[index] -= 1
                    
        # Nếu thoát khỏi vòng lặp tức là toàn bộ worker đều fail
        assert last_error is not None
        raise last_error

    # Các hàm Wrapper bên dưới sẽ tự động gọi _call() 
    # để truyền lời gọi qua thuật toán least-busy.

    def caption(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("caption", *args, **kwargs)

    def ocr(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("ocr", *args, **kwargs)

    def embed_text(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("embed_text", *args, **kwargs)

    def embed_images(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("embed_images", *args, **kwargs)

    def boundary_scores(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("boundary_scores", *args, **kwargs)

    def transcribe_audio_reference(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("transcribe_audio_reference", *args, **kwargs)

    def diarize_audio_reference(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("diarize_audio_reference", *args, **kwargs)

    def close(self) -> None:
        """Đóng tất cả kết nối TCP/HTTP đang mở."""
        for client in self.clients:
            client.close()
