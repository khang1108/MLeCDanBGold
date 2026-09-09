"""Run reproducible remote KIS/TRAKE benchmark requests.

The script reads competition-style query text files, calls the configured
public API, and stores raw responses plus wall-clock timings.  It does not
assign relevance; evaluation is intentionally a separate step so that
ground-truth and manual/weak-label judgments remain distinguishable.

Transient transport and server failures are retried with a short exponential
backoff.  The raw output records the number of attempts so a report can
distinguish a clean request from one recovered after a transient failure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import httpx


def build_items(root: Path, split: str) -> list[tuple[Path, dict[str, object], str]]:
    """Load KIS and TRAKE query payloads from one artifact split."""

    items: list[tuple[Path, dict[str, object], str]] = []
    for path in sorted((root / split).glob("*.txt")):
        if path.name.endswith("-kis.txt") or path.name.endswith("-qa.txt"):
            payload: dict[str, object] = {
                "query": path.read_text(encoding="utf-8").strip(),
                "top_k": 100,
                "use_dense": True,
                "use_bm25": True,
            }
            kind = "qa" if path.name.endswith("-qa.txt") else "kis"
            items.append((path, payload, kind))
        elif path.name.endswith("-trake.txt"):
            events = [
                match.group(1).strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if (match := re.match(r"^(?:E\d+|Cảnh\s*\d+)\s*:?\s*(.+)$", line.strip(), re.IGNORECASE))
            ]
            payload = {
                "events": events,
                "top_k": 100,
                "use_dense": True,
                "use_bm25": True,
            }
            items.append((path, payload, "trake"))
    return items


async def run(args: argparse.Namespace) -> None:
    """Execute all selected requests with bounded concurrency."""

    root = Path(args.query_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    allowed_kinds = set(args.kinds)
    items = [
        item
        for split in args.splits
        for item in build_items(root, split)
        if item[2] in allowed_kinds
    ]
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout, connect=20.0)
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        timeout=timeout,
        limits=httpx.Limits(
            max_connections=args.concurrency,
            max_keepalive_connections=args.concurrency,
        ),
    ) as client:

        async def one(path: Path, payload: dict[str, object], kind: str) -> dict[str, object]:
            async with semaphore:
                started = time.perf_counter()
                endpoint = "/api/v1/trake" if kind == "trake" else "/api/v1/search"
                errors: list[str] = []
                attempts = 0
                while attempts <= args.retries:
                    attempts += 1
                    try:
                        response = await client.post(endpoint, json=payload)
                        try:
                            body: object = response.json()
                        except Exception:
                            body = {"raw": response.text}
                        retryable = response.status_code in (408, 429) or response.status_code >= 500
                        if not retryable or attempts > args.retries:
                            elapsed = time.perf_counter() - started
                            print(path.name, response.status_code, f"{elapsed:.1f}s", f"attempts={attempts}", flush=True)
                            return {
                                "query_file": str(path),
                                "type": kind,
                                "payload": payload,
                                "status_code": response.status_code,
                                "elapsed_wall_s": elapsed,
                                "attempts": attempts,
                                "retry_errors": errors,
                                "response": body,
                            }
                        errors.append(f"HTTP {response.status_code}")
                    except Exception as error:
                        errors.append(repr(error))
                        if attempts > args.retries:
                            elapsed = time.perf_counter() - started
                            print(path.name, "ERROR", repr(error), f"attempts={attempts}", flush=True)
                            return {
                                "query_file": str(path),
                                "type": kind,
                                "payload": payload,
                                "status_code": 0,
                                "elapsed_wall_s": elapsed,
                                "attempts": attempts,
                                "retry_errors": errors,
                                "error": repr(error),
                            }
                    await asyncio.sleep(min(2.0 ** (attempts - 1), 8.0))

        results = await asyncio.gather(*(one(*item) for item in items))
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(results)} responses to {output}")


def main() -> None:
    """Parse CLI options and run the asynchronous benchmark."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://backend.iamphuckhang.dev")
    parser.add_argument("--query-root", default="artifacts/query")
    parser.add_argument("--splits", nargs="+", default=["002"])
    parser.add_argument("--output", default="artifacts/benchmark/2026-09-06/split002_results.json")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries for transport/408/429/5xx failures (default: 2).",
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=("kis", "qa", "trake"),
        default=("kis", "qa", "trake"),
        help="Query families to execute; QA uses the KIS retrieval endpoint.",
    )
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
