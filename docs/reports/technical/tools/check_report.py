#!/usr/bin/env python3
"""Validate the technical report's publication constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

BODY_PAGE_LIMIT = 4
REQUIRED_SCREENSHOTS = (
    "figures/screenshots/kis-results.png",
    "figures/screenshots/trake-results.png",
    "figures/screenshots/video-inspector.png",
    "figures/screenshots/collaborative-workspace.png",
)
REQUIRED_GENERATED_INPUTS = (
    "generated/artifact-statistics.json",
    "generated/artifact-statistics.tex",
)
ARTIFACT_STATISTICS_LABEL = "fig:artifact-statistics"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--aux", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    return parser.parse_args()


def body_end_page(aux_text: str) -> int:
    match = re.search(r"\\newlabel\{body:last\}\{\{[^}]*\}\{(\d+)\}", aux_text)
    if match is None:
        raise ValueError("body:last label is absent from the auxiliary file")
    return int(match.group(1))


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        signature = stream.read(24)
    if len(signature) < 24 or signature[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    return struct.unpack(">II", signature[16:24])


def main() -> int:
    args = arguments()
    root = args.root.resolve()
    failures: list[str] = []

    for path in (args.aux, args.log, args.pdf):
        if not path.is_file():
            failures.append(f"missing build output: {path}")
    if failures:
        return report(failures)

    aux_text = args.aux.read_text(encoding="utf-8", errors="replace")
    try:
        page = body_end_page(aux_text)
        if page > BODY_PAGE_LIMIT:
            failures.append(
                f"title-through-body ends on page {page}; limit is {BODY_PAGE_LIMIT}"
            )
    except ValueError as error:
        failures.append(str(error))

    body_source = (root / "sections/main-body.tex").read_text(encoding="utf-8")
    if re.search(r"\\begin\{(?:figure|table)\*?\}", body_source):
        failures.append("a figure/table float appears in sections/main-body.tex")

    generated_paths = [root / relative for relative in REQUIRED_GENERATED_INPUTS]
    for relative, path in zip(REQUIRED_GENERATED_INPUTS, generated_paths, strict=True):
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing generated report input: {relative}")

    statistics_path = generated_paths[0]
    if statistics_path.is_file():
        try:
            statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
            method = statistics["method"]
            if method.get("reader") != "pyarrow.parquet.ParquetFile.iter_batches":
                failures.append("artifact statistics do not record the bounded-batch reader")
            batch_size = method.get("batch_size")
            if not isinstance(batch_size, int) or not 1 <= batch_size <= 8_192:
                failures.append("artifact-statistics batch size is absent or exceeds 8,192")
            if method.get("use_threads") is not False:
                failures.append("artifact statistics must record use_threads=false")
        except (AttributeError, KeyError, TypeError, json.JSONDecodeError, OSError) as error:
            failures.append(f"invalid generated artifact statistics JSON: {error}")

    appendix_path = root / "appendices/appendix.tex"
    if not appendix_path.is_file():
        failures.append("missing appendix source: appendices/appendix.tex")
    else:
        appendix_source = appendix_path.read_text(encoding="utf-8")
        required_fragments = (
            r"\input{generated/artifact-statistics}",
            rf"\label{{{ARTIFACT_STATISTICS_LABEL}}}",
            rf"\ref{{{ARTIFACT_STATISTICS_LABEL}}}",
        )
        for fragment in required_fragments:
            if fragment not in appendix_source:
                failures.append(f"appendix is missing artifact-chart fragment: {fragment}")
    if rf"\newlabel{{{ARTIFACT_STATISTICS_LABEL}}}" not in aux_text:
        failures.append("compiled auxiliary file is missing the artifact-statistics chart label")

    log_text = args.log.read_text(encoding="utf-8", errors="replace")
    warning_patterns = (
        r"LaTeX Warning: There were undefined references",
        r"LaTeX Warning: Citation .* undefined",
        r"Package natbib Warning: Citation .* undefined",
        r"multiply defined",
    )
    for pattern in warning_patterns:
        if re.search(pattern, log_text, flags=re.IGNORECASE):
            failures.append(f"build log matches: {pattern}")

    try:
        output = subprocess.run(
            ["pdfinfo", str(args.pdf)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        pages = re.search(r"^Pages:\s+(\d+)$", output, flags=re.MULTILINE)
        if pages is None:
            failures.append("pdfinfo did not report a page count")
        elif int(pages.group(1)) <= BODY_PAGE_LIMIT:
            failures.append("compiled PDF has no references/appendix pages")
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        failures.append(f"could not inspect PDF page count: {error}")

    for relative in REQUIRED_SCREENSHOTS:
        path = root / relative
        if not path.is_file():
            failures.append(f"missing required live screenshot: {relative}")
            continue
        try:
            width, height = png_dimensions(path)
        except (OSError, ValueError) as error:
            failures.append(f"invalid screenshot {relative}: {error}")
            continue
        if width < 1200 or height < 700:
            failures.append(
                f"screenshot {relative} is only {width}x{height}; minimum is 1200x700"
            )

    return report(failures)


def report(failures: list[str]) -> int:
    if failures:
        print("Report validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("Report validation passed: body <= 4 pages, appendix floats only, references resolved, and live screenshots present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
