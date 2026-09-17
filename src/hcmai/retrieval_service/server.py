"""gRPC server process and CLI entry point for retrieval service.

This module starts one long-lived retrieval server on 127.0.0.1:8002, registers
the standard health service, and hosts RetrievalServicer without Uvicorn.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import logging
import signal
import sys
import time
import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from hcmai.common.environment import load_repository_environment
from hcmai.common.utils.logging import configure_logging, get_logger
from hcmai.retrieval_service.proto import retrieval_pb2_grpc
from hcmai.retrieval_service.runtime import RetrievalRuntime
from hcmai.retrieval_service.servicer import RetrievalServicer

logger = get_logger(__name__)

SERVICE_NAME = "hcmai.retrieval.v1.RetrievalService"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8002
DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def create_server(
    runtime: RetrievalRuntime | None,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    startup_messages: Sequence[str] = (),
) -> tuple[grpc.Server, health.HealthServicer, int]:
    """Create and configure one gRPC server with health and retrieval services."""
    options = (
        ("grpc.max_send_message_length", max_message_bytes),
        ("grpc.max_receive_message_length", max_message_bytes),
    )
    server = grpc.server(ThreadPoolExecutor(max_workers=4), options=options)
    servicer = RetrievalServicer(runtime, startup_messages=startup_messages)
    retrieval_pb2_grpc.add_RetrievalServiceServicer_to_server(servicer, server)

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    status = (
        health_pb2.HealthCheckResponse.SERVING
        if runtime is not None
        else health_pb2.HealthCheckResponse.NOT_SERVING
    )
    health_servicer.set("", status)
    health_servicer.set(SERVICE_NAME, status)

    bound_port = server.add_insecure_port(f"{host}:{port}")
    if bound_port == 0:
        raise RuntimeError(f"Could not bind gRPC server to {host}:{port}")

    # Attach servicer to server for testing/inspection
    server._servicer = servicer  # type: ignore[attr-defined]
    return server, health_servicer, bound_port


def main() -> None:
    """CLI entry point to launch the standalone retrieval server."""
    parser = argparse.ArgumentParser(
        description="Standalone gRPC retrieval service for HCMAI",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Bind host (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Bind port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--max-message-mib",
        type=int,
        default=64,
        help="Maximum message size in MiB (default: 64)",
    )
    args = parser.parse_args()

    configure_logging(level=logging.INFO)
    load_repository_environment()

    max_message_bytes = args.max_message_mib * 1024 * 1024
    startup_messages: list[str] = []

    logger.info("Starting standalone retrieval service initialization...")
    runtime: RetrievalRuntime | None = None
    try:
        runtime = RetrievalRuntime.load(startup_messages)
    except Exception as error:
        logger.exception("Unexpected error during RetrievalRuntime loading: %s", error)
        startup_messages.append(f"Fatal startup exception: {type(error).__name__}: {error}")

    if runtime is None:
        logger.warning(
            "Retrieval runtime is NOT ready. Server will start in NOT_SERVING state. "
            "Diagnostics: %s",
            startup_messages,
        )
    else:
        logger.info(
            "Retrieval runtime ready. Scoring revision: %s, Modalities: %s",
            runtime.scoring_revision,
            runtime.active_modalities,
        )

    server, health_servicer, bound_port = create_server(
        runtime,
        host=args.host,
        port=args.port,
        max_message_bytes=max_message_bytes,
        startup_messages=startup_messages,
    )

    server.start()
    logger.info("Retrieval service listening on %s:%d", args.host, bound_port)

    stop_event = False

    def handle_signal(signum: int, frame: object) -> None:
        nonlocal stop_event
        logger.info("Received signal %d, shutting down...", signum)
        health_servicer.set("", health_pb2.HealthCheckResponse.NOT_SERVING)
        health_servicer.set(SERVICE_NAME, health_pb2.HealthCheckResponse.NOT_SERVING)
        server.stop(grace=5)
        stop_event = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while not stop_event:
            time.sleep(0.5)
    except KeyboardInterrupt:
        handle_signal(signal.SIGINT, None)

    logger.info("Retrieval service stopped.")


if __name__ == "__main__":
    main()
