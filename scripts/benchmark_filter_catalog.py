"""Benchmark Filter catalog latency and process RSS with reproducible queries."""

from __future__ import annotations

import argparse
import json
import math
import time

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError

from hcmai.api.contracts import FilterRequest
from hcmai.filtering.catalog import FilterCatalog
from hcmai.filtering.service import FilterService


BENCHMARK_SCHEMA_VERSION = "filter-benchmark-v1"


def _positive_integer(value: str) -> int:
    """Parse one strictly positive benchmark resource setting."""

    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    """Define a benchmark that never mutates production runtime settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--concurrency", type=_positive_integer, default=1)
    parser.add_argument("--samples", type=_positive_integer, default=20)
    parser.add_argument("--warmups", type=_positive_integer, default=1)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run configured cases and emit one stable JSON measurement document."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    cases = _load_cases(arguments.queries, parser)
    rss_before = _rss_kib()
    catalog = FilterCatalog.open(
        arguments.catalog,
        pool_size=min(arguments.concurrency, 4),
    )
    service = FilterService(catalog)
    try:
        measured_cases = [
            _measure_case(
                service,
                name=name,
                request=request,
                concurrency=arguments.concurrency,
                samples=arguments.samples,
                warmups=arguments.warmups,
            )
            for name, request in cases
        ]
        rss_after = _rss_kib()
        payload = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "catalog_version": catalog.info.catalog_version,
            "catalog_frame_count": catalog.info.frame_count,
            "catalog_size_bytes": arguments.catalog.stat().st_size,
            "concurrency": arguments.concurrency,
            "samples_per_case": arguments.samples,
            "warmups_per_case": arguments.warmups,
            "rss_before_kib": rss_before,
            "rss_after_kib": rss_after,
            "rss_delta_kib": max(0, rss_after - rss_before),
            "cases": measured_cases,
        }
    finally:
        service.close()

    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


def _load_cases(
    path: Path,
    parser: argparse.ArgumentParser,
) -> list[tuple[str, FilterRequest]]:
    """Validate named public requests before allocating catalog resources."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"could not read query fixture: {error}")
    if not isinstance(raw, list) or not raw:
        parser.error("query fixture must contain a non-empty array")

    cases: list[tuple[str, FilterRequest]] = []
    names: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            parser.error(f"query case {index} must be an object")
        name = item.get("name")
        request = item.get("request")
        if not isinstance(name, str) or not name.strip():
            parser.error(f"query case {index} requires a nonblank name")
        normalized_name = name.strip()
        if normalized_name in names:
            parser.error(f"duplicate query case name: {normalized_name}")
        if not isinstance(request, dict):
            parser.error(f"query case {normalized_name} requires a request object")
        try:
            parsed_request = FilterRequest.model_validate(request)
        except ValidationError as error:
            parser.error(f"query case {normalized_name} is invalid: {error}")
        names.add(normalized_name)
        cases.append((normalized_name, parsed_request))
    return cases


def _measure_case(
    service: FilterService,
    *,
    name: str,
    request: FilterRequest,
    concurrency: int,
    samples: int,
    warmups: int,
) -> dict[str, Any]:
    """Warm one request, then record successful latency and error counts."""

    for _ in range(warmups):
        service.filter(request)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        measurements = list(
            executor.map(
                lambda _: _measure_once(service, request),
                range(samples),
            )
        )
    latencies = sorted(
        latency for latency, error in measurements if error is None
    )
    errors = [error for _, error in measurements if error is not None]
    return {
        "name": name,
        "sample_count": samples,
        "success_count": len(latencies),
        "error_count": len(errors),
        "p50_ms": _percentile(latencies, 0.50),
        "p95_ms": _percentile(latencies, 0.95),
        "max_ms": max(latencies, default=0.0),
        "error_types": sorted({type(error).__name__ for error in errors}),
    }


def _measure_once(
    service: FilterService,
    request: FilterRequest,
) -> tuple[float, Exception | None]:
    """Measure one complete count-plus-page service call in milliseconds."""

    started = time.perf_counter_ns()
    try:
        service.filter(request)
    except Exception as error:
        return (time.perf_counter_ns() - started) / 1_000_000, error
    return (time.perf_counter_ns() - started) / 1_000_000, None


def _percentile(values: list[float], fraction: float) -> float:
    """Return a deterministic nearest-rank percentile for small sample sets."""

    if not values:
        return 0.0
    index = max(0, math.ceil(fraction * len(values)) - 1)
    return values[index]


def _rss_kib() -> int:
    """Read current Linux process resident memory in KiB."""

    status = Path("/proc/self/status").read_text(encoding="utf-8")
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    raise RuntimeError("VmRSS is not available in /proc/self/status")


if __name__ == "__main__":
    raise SystemExit(main())
