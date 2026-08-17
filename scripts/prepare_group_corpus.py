"""Script (entrypoint) để chạy Data Preparation Pipeline cho riêng MỘT Group (phân tán).
Nhận đầu vào là thư mục chứa video (local) và file inventory (danh sách video của group).
Sau khi chạy xong sẽ tự động upload (publish/commit) kết quả lên S3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hcmai.data.corpus_build import (
    GroupPreparationService,
    S3CorpusPreparationConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/preparation.s3.yaml"))
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--enrichment-config", type=Path, default=Path("configs/enrichment.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("llm/config.yaml"))
    parser.add_argument("--retrieval-config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--cleanup-raw", action="store_true")
    parser.add_argument("--cleanup-artifacts", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = S3CorpusPreparationConfig.from_yaml(args.config)
    result = GroupPreparationService(
        config,
        args.videos,
        args.inventory,
        resume=not args.no_resume,
        cleanup_raw=args.cleanup_raw,
        cleanup_artifacts=args.cleanup_artifacts,
        enrichment_config=args.enrichment_config,
        model_config=args.model_config,
        retrieval_config=args.retrieval_config,
    ).run()
    print(f"Group run: {result.run_id}")
    print(f"Sources: {result.source_count}")
    print(f"Completed: {','.join(result.completed_stages) or '-'}")
    print(f"Resumed: {','.join(result.skipped_stages) or '-'}")
    if result.publication is not None:
        print(
            "Committed: "
            f"s3://{result.publication.bucket}/{result.publication.latest_key}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

