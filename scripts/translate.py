"""Dịch caption tiếng Anh sang tiếng Việt bằng Qwen3-VL chạy cục bộ qua vLLM.

Bỏ qua file đã có sẵn ở thư mục output, nên chạy lại là tiếp tục dở dang.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
ARTIFACT_VERSION = "caption_qwen_vl_vi_v1"
PROMPT = (
    "Dịch mô tả khung hình sau sang tiếng Việt. Giữ nguyên mọi chi tiết "
    "(màu sắc, số lượng, vật thể, vị trí). Chỉ trả về bản dịch.\n\n"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse input captions, output root, and A6000 engine settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser.parse_args(argv)


def pending(table: pa.Table) -> list[int]:
    """Chỉ số các dòng caption hoàn chỉnh, bỏ dòng lỗi và dòng rỗng."""

    statuses = table.column("status").to_pylist()
    texts = table.column("text").to_pylist()
    return [
        index
        for index, (status, text) in enumerate(zip(statuses, texts))
        if status == "completed" and text
    ]


def merge(texts: Sequence[str], picked: Sequence[int], values: Sequence[str]) -> list[str]:
    """Đặt bản dịch vào đúng vị trí, giữ nguyên các dòng không dịch."""

    if len(picked) != len(values):
        raise ValueError("số bản dịch không khớp số dòng đã gửi")
    merged = list(texts)
    for index, value in zip(picked, values):
        merged[index] = value
    return merged


def rewrite(table: pa.Table, translated: Sequence[str]) -> pa.Table:
    """Thay cột text và đánh dấu artifact_version của bản dịch."""

    table = table.set_column(
        table.schema.get_field_index("text"),
        "text",
        pa.array(list(translated), type=table.schema.field("text").type),
    )
    index = table.schema.get_field_index("artifact_version")
    return table.set_column(
        index,
        "artifact_version",
        pa.array([ARTIFACT_VERSION] * table.num_rows, type=table.schema.field(index).type),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Dịch từng file caption chưa có kết quả rồi ghi ra thư mục output."""

    args = parse_args(argv)
    root = args.captions
    files = sorted(root.rglob("caption.parquet")) if root.is_dir() else [root]
    todo = [
        (
            source,
            args.output / (source.relative_to(root) if root.is_dir() else source.name),
        )
        for source in files
    ]
    todo = [(source, target) for source, target in todo if not target.exists()]
    print(f"{len(files)} file, còn {len(todo)} file cần dịch")
    if not todo:
        return 0

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    for source, target in tqdm(todo, unit="file"):
        table = pq.read_table(source)
        texts = table.column("text").to_pylist()
        picked = pending(table)
        outputs = llm.chat(
            [
                [{"role": "user", "content": PROMPT + texts[index]}]
                for index in picked
            ],
            sampling,
            use_tqdm=False,
        )
        values = [output.outputs[0].text.strip() for output in outputs]
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(rewrite(table, merge(texts, picked, values)), target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
