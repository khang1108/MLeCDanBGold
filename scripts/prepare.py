"""Single entry point for local V3C/VBS preparation stages.

For a single A6000, run GPU-heavy stages one by one and restart the GPU server
with the matching HCMAI_GPU_TASK between caption/ocr/objects/asr.  ``paper-fast``
only needs the core embedding API plus the caption GPU worker.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from offline.pipeline import STAGES, run_sequence, run_stage

PAPER_FAST = ("frames", "visual-index", "caption")

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*STAGES, "paper-fast"))
    parser.add_argument("--config", type=Path, default=Path("configs/vbs_prepare.yaml"))
    return parser.parse_args(argv)

def main(argv=None) -> int:
    args = parse_args(argv)
    if args.stage == "paper-fast":
        run_sequence(PAPER_FAST, args.config)
    else:
        run_stage(args.stage, args.config)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
