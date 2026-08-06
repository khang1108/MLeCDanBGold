"""Fast read-only diagnostics for local HCMAI runtime artifacts."""

from __future__ import annotations

import argparse
import json

from hcmai.orchestration.diagnostics import diagnose


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--check-remote", action="store_true")
    args = parser.parse_args()
    report = diagnose(
        args.config,
        sample_size=args.sample_size,
        check_remote=args.check_remote,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
