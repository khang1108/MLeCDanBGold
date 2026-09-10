"""Shared command-line entry point for method-specific baseline runners."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys
from typing import Any, cast

from baseline.contracts import BASELINE_METHODS, MethodName, MethodOptions, QUERY_KINDS
from baseline.query_io import load_query_cases
from baseline.runner import RunOptions, run_cases, write_run
from hcmai.orchestration.pipeline import SearchService


_LATTICE_METHODS = {"unary_candidate_lattice", "unary_bounded_order"}
_TEMPORAL_METHODS = {"dante_style_unary_dp", *_LATTICE_METHODS}


def build_parser(method: MethodName) -> argparse.ArgumentParser:
    """Build a parser exposing only controls relevant to one fixed method."""

    if method not in BASELINE_METHODS:
        raise ValueError(f"unsupported baseline method: {method}")
    parser = argparse.ArgumentParser(
        prog=f"python -m baseline.{_module_name(method)}",
        description=f"Run the {method} full-corpus retrieval control.",
    )
    parser.add_argument("--query-root", type=Path, default=Path("artifacts/query"))
    parser.add_argument("--splits", nargs="+", default=["002"])
    parser.add_argument("--kinds", nargs="+", choices=QUERY_KINDS, default=list(QUERY_KINDS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=_positive_int, default=100)
    parser.add_argument("--max-queries", type=_positive_int)
    parser.add_argument(
        "--use-dense",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-bm25",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    if method in _TEMPORAL_METHODS:
        parser.add_argument("--lambda-gap", type=_non_negative_float)
        parser.add_argument("--event-power", type=_positive_float)
    if method == "dante_style_unary_dp":
        parser.add_argument("--cluster-delta", type=_non_negative_float)
    if method in _LATTICE_METHODS:
        parser.add_argument("--candidates-per-event", type=_positive_int, default=32)
        parser.add_argument(
            "--candidate-min-separation-ms",
            type=_non_negative_int,
            default=0,
        )
    if method == "unary_bounded_order":
        parser.add_argument("--max-adjacent-swaps", type=_non_negative_int, default=1)
        parser.add_argument("--max-order-candidates", type=_positive_int, default=64)
        parser.add_argument("--swap-penalty", type=_non_negative_float, default=0.0)
    return parser


def parse_args(
    method: MethodName,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse and cross-validate one method's CLI arguments."""

    parser = build_parser(method)
    args = parser.parse_args(argv)
    if not args.use_dense and not args.use_bm25:
        parser.error("at least one of --use-dense or --use-bm25 must be enabled")
    return args


def main(method: MethodName, argv: Sequence[str] | None = None) -> int:
    """Load configured artifacts once, execute selected queries, and write JSON."""

    args = parse_args(method, argv)
    cases = load_query_cases(args.query_root, args.splits, args.kinds, args.max_queries)
    if not cases:
        build_parser(method).error("the selected splits and kinds contain no query files")

    messages: list[str] = []
    service = _load_service(messages)
    try:
        for message in messages:
            print(f"startup: {message}", file=sys.stderr)
        options = RunOptions(
            method=method,
            top_k=args.top_k,
            use_dense=args.use_dense,
            use_bm25=args.use_bm25,
            method_options=_method_options(args, service),
        )
        run = run_cases(service, cases, options)
        write_run(args.output, run)
    finally:
        service.close()
    responses = cast(list[dict[str, object]], run["responses"])
    failure_count = sum(item.get("status_code") != 200 for item in responses)
    print(f"saved {len(cases)} baseline responses to {args.output}")
    if failure_count:
        print(
            f"baseline run completed with {failure_count} failed queries",
            file=sys.stderr,
        )
        return 1
    return 0


def _load_service(messages: list[str]) -> SearchService:
    return SearchService.load(messages)


def _method_options(
    args: argparse.Namespace,
    service: Any,
) -> MethodOptions:
    alignment = service.config.alignment
    lambda_gap = getattr(args, "lambda_gap", None)
    event_power = getattr(args, "event_power", None)
    cluster_delta = getattr(args, "cluster_delta", None)
    return MethodOptions(
        lambda_gap=(alignment.lambda_gap if lambda_gap is None else lambda_gap),
        event_power=(alignment.event_power if event_power is None else event_power),
        cluster_delta=(alignment.cluster_delta if cluster_delta is None else cluster_delta),
        candidates_per_event=getattr(args, "candidates_per_event", 32),
        candidate_min_separation_ms=getattr(args, "candidate_min_separation_ms", 0),
        max_adjacent_swaps=getattr(args, "max_adjacent_swaps", 1),
        max_order_candidates=getattr(args, "max_order_candidates", 64),
        swap_penalty=getattr(args, "swap_penalty", 0.0),
    )


def _module_name(method: MethodName) -> str:
    return {
        "global_query": "run_global_query",
        "independent_events": "run_independent_events",
        "dante_style_unary_dp": "run_dante_style",
        "unary_candidate_lattice": "run_unary_lattice",
        "unary_bounded_order": "run_unary_bounded_order",
    }[method]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed
