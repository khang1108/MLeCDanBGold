"""Chia nhỏ S3 bucket thành N nhóm `GroupSourceInventory` để phân tán task.

Chạy script này ở local để lấy 5 file JSON. Mỗi file JSON sẽ được nạp
vào một Kaggle Notebook riêng biệt để chạy `prepare_group_corpus.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hcmai.data.corpus_build.config import S3CorpusPreparationConfig
from hcmai.data.corpus_build.group import GroupSourceInventory, GroupSourceObject
from hcmai.data.s3 import create_s3_client, list_video_objects

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/preparation.s3.yaml",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=5,
        help="Số lượng Kaggle notebooks (mặc định: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kaggle_groups"),
        help="Thư mục chứa các file JSON được sinh ra",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    print(f"Đọc cấu hình S3 từ {args.config}...")
    config = S3CorpusPreparationConfig.from_yaml(args.config)
    s3_config = config.preprocessing.s3
    if not s3_config:
        raise ValueError("Config không có phần S3 preprocessing")

    client = create_s3_client(s3_config)
    
    print(f"Đang fetch danh sách objects từ s3://{s3_config.bucket}/{s3_config.videos_prefix} ...")
    objects = list_video_objects(client, s3_config)
    total_videos = len(objects)
    print(f"Tổng cộng tìm thấy: {total_videos} videos.")
    
    if total_videos == 0:
        print("Không có video nào để chia!")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Chia danh sách objects thành num_groups phần
    # Sắp xếp để đảm bảo tính nhất quán
    objects.sort(key=lambda o: o.key)
    
    chunk_size = (total_videos + args.num_groups - 1) // args.num_groups
    chunks = [objects[i:i + chunk_size] for i in range(0, total_videos, chunk_size)]
    
    for i, chunk in enumerate(chunks, start=1):
        group_id = f"group-{i:02d}"
        
        inventory_objects = [
            GroupSourceObject(
                key=obj.key,
                size=obj.size,
                etag=obj.etag,
                last_modified_ns=obj.last_modified_ns,
            )
            for obj in chunk
        ]
        
        inventory = GroupSourceInventory(
            group_id=group_id,
            bucket=s3_config.bucket,
            prefix=s3_config.videos_prefix,
            objects=inventory_objects,
        )
        
        output_path = args.output_dir / f"{group_id}.json"
        with output_path.open("w", encoding="utf-8") as f:
            f.write(inventory.model_dump_json(indent=2))
            
        print(f"Đã lưu {output_path} ({len(inventory_objects)} videos)")

    print("\nHoàn tất! Hãy upload các file JSON này lên Kaggle hoặc dùng nội dung của chúng.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
