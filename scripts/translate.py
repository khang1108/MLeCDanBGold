"""Translate English captions to Vietnamese with Qwen3 served locally by vLLM.

Caption files already present in the output directory are skipped, so a rerun resumes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

MODEL = "Qwen/Qwen3-8B"
ARTIFACT_VERSION = "caption_vi_v1"
SYSTEM = """Dịch mô tả khung hình video từ tiếng Anh sang tiếng Việt.

Quy tắc:
- Dịch trọn vẹn, không để sót bất kỳ từ tiếng Anh nào.
- Giữ nguyên tên riêng, tên thương hiệu, và chữ trên biển hiệu hoặc logo.
- Giữ đủ mọi chi tiết: màu sắc, số lượng, chất liệu, vật thể, vị trí, hành động.
- Không thêm, không bớt, không giải thích, không mở đầu.
- Trả lời bằng đúng một đoạn văn là bản dịch."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse input captions, output root, and A6000 engine settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser.parse_args(argv)


def pending(table: pa.Table) -> list[int]:
    """Return the row indexes of completed, non-empty captions."""

    statuses = table.column("status").to_pylist()
    texts = table.column("text").to_pylist()
    return [
        index
        for index, (status, text) in enumerate(zip(statuses, texts))
        if status == "completed" and text
    ]


def merge(texts: Sequence[str], picked: Sequence[int], values: Sequence[str]) -> list[str]:
    """Place each translation at its own row, leaving untranslated rows unchanged."""

    if len(picked) != len(values):
        raise ValueError("số bản dịch không khớp số dòng đã gửi")
    merged = list(texts)
    for index, value in zip(picked, values):
        merged[index] = value
    return merged


def rewrite(table: pa.Table, translated: Sequence[str], revision: str | None) -> pa.Table:
    """Replace the text column and rewrite lineage to the translating model."""

    replacements = {
        "text": list(translated),
        "artifact_version": [ARTIFACT_VERSION] * table.num_rows,
        "model_name": [MODEL] * table.num_rows,
        "model_revision": [revision] * table.num_rows,
    }
    for column, values in replacements.items():
        index = table.schema.get_field_index(column)
        table = table.set_column(
            index, column, pa.array(values, type=table.schema.field(index).type)
        )
    return table


def main(argv: Sequence[str] | None = None) -> int:
    """Translate every caption file without a result and write it to the output tree."""

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
        revision=args.revision,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    for source, target in tqdm(todo, unit="file"):
        table = pq.read_table(source)
        texts = table.column("text").to_pylist()
        picked = pending(table)
        outputs = llm.chat(
            [
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": texts[index]},
                ]
                for index in picked
            ],
            sampling,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        values = [output.outputs[0].text.strip() for output in outputs]
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            rewrite(table, merge(texts, picked, values), args.revision), target
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
