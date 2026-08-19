"""Run resumable, chunked remote ASR directly from a GPU worker.

The worker downloads only one bounded chunk of S3 videos, extracts audio
locally, sends signed temporary-audio references to the ASR service running on
``127.0.0.1:8101``, uploads each transcript pair back to S3, and then removes
the chunk.  It intentionally does not build FrameStore, Caption, OCR, Object,
or index artifacts.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from tqdm import tqdm

from hcmai.common.config import InferenceConfig, TranscriptJobConfig
from hcmai.data.corpus_build.audio import S3AudioReferenceProvider
from hcmai.data.corpus_build.config import S3CorpusPreparationConfig
from hcmai.data.enrichment.transcripts.adapters.remote import RemoteASRAdapter
from hcmai.data.enrichment.transcripts.pipeline import TranscriptService
from hcmai.data.s3 import S3VideoObject, create_s3_client, list_video_objects
from hcmai.llm.adapters.http import InferenceClient


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the bounded S3 ASR worker interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/preparation.s3.yaml"))
    parser.add_argument("--enrichment-config", type=Path, default=Path("configs/enrichment.yaml"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8101")
    parser.add_argument("--chunk-gib", type=float, default=40.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--output-prefix")
    return parser.parse_args(argv)


def _chunks(
    sources: Sequence[S3VideoObject], max_bytes: int
) -> Iterator[list[S3VideoObject]]:
    """Group source videos without exceeding the configured staging budget."""

    chunk: list[S3VideoObject] = []
    total = 0
    for source in sources:
        if chunk and total + source.size > max_bytes:
            yield chunk
            chunk, total = [], 0
        chunk.append(source)
        total += source.size
    if chunk:
        yield chunk


def _head_exists(client: Any, bucket: str, key: str) -> bool:
    """Return whether one uploaded S3 artifact exists."""

    try:
        client.head_object(Bucket=bucket, Key=key)
    except Exception:
        return False
    return True


def _artifact_keys(prefix: str, video_id: str) -> tuple[str, str]:
    """Return stable S3 keys for one transcript and its manifest."""

    group = video_id.split("_", maxsplit=1)[0]
    stem = f"{prefix.strip('/')}/{group}/{video_id}"
    return f"{stem}.parquet", f"{stem}.manifest.json"


@contextmanager
def _download_chunk(
    client: Any,
    storage: Any,
    sources: Sequence[S3VideoObject],
    root: Path,
) -> Iterator[dict[str, Path]]:
    """Download one bounded chunk and remove it when the context exits."""

    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    try:
        for source in sources:
            path = root / Path(source.key).name
            partial = path.with_suffix(f"{path.suffix}.partial")
            partial.unlink(missing_ok=True)
            client.download_file(storage.bucket, source.key, str(partial))
            if not partial.is_file() or partial.stat().st_size != source.size:
                raise OSError(f"downloaded size mismatch: {source.key}")
            partial.replace(path)
            paths[source.video_id] = path
        yield paths
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run ASR in bounded S3 chunks and publish per-video transcripts."""

    args = parse_args(argv)
    if args.chunk_gib <= 0:
        raise ValueError("--chunk-gib must be positive")
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")

    preparation = S3CorpusPreparationConfig.from_yaml(args.config)
    storage = preparation.preprocessing.s3
    if storage is None:
        raise ValueError("preparation config must define preprocessing.s3")
    if storage.staging_root is None:
        raise ValueError("preparation config must define s3.staging_root")
    enrichment = TranscriptJobConfig.from_yaml(args.enrichment_config)
    client = create_s3_client(storage)
    sources = list_video_objects(client, storage)
    sources = sources[args.offset :]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        sources = sources[: args.limit]
    if not sources:
        raise ValueError("no S3 videos selected")

    output_prefix = args.output_prefix or f"{storage.artifacts_prefix}/transcripts"
    endpoint = InferenceClient(
        args.endpoint,
        InferenceConfig(
            enabled=True,
            timeout_seconds=300,
            connect_timeout_seconds=10,
            read_timeout_seconds=300,
            write_timeout_seconds=60,
            pool_timeout_seconds=30,
            max_attempts=3,
            max_concurrency=1,
        ),
    )
    references: S3AudioReferenceProvider | None = None
    processed = skipped = failed = 0
    max_bytes = int(args.chunk_gib * 1024**3)
    try:
        for chunk_index, chunk in enumerate(_chunks(sources, max_bytes), start=1):
            chunk_size = sum(item.size for item in chunk)
            print(
                f"Chunk {chunk_index}: {len(chunk)} videos, "
                f"{chunk_size / 1024**3:.2f} GiB"
            )
            with tempfile.TemporaryDirectory(
                prefix=f"hcmai-asr-{chunk_index:05d}-",
                dir=storage.staging_root,
            ) as temporary:
                chunk_root = Path(temporary)
                references = S3AudioReferenceProvider(
                    client,
                    bucket=storage.bucket,
                    prefix=f"{output_prefix}/temporary-audio",
                    work_root=chunk_root / "state",
                )
                service = TranscriptService(
                    RemoteASRAdapter(endpoint, enrichment.asr, references)
                )
                with _download_chunk(
                    client, storage, chunk, chunk_root / "videos"
                ) as paths:
                    for source in tqdm(chunk, desc=f"ASR chunk {chunk_index}", unit="video"):
                        parquet_key, manifest_key = _artifact_keys(
                            output_prefix, source.video_id
                        )
                        if (
                            not args.no_resume
                            and _head_exists(client, storage.bucket, parquet_key)
                            and _head_exists(client, storage.bucket, manifest_key)
                        ):
                            skipped += 1
                            continue
                        output_root = chunk_root / "transcripts"
                        try:
                            output, _ = service.prepare_video(
                                paths[source.video_id],
                                output_root,
                                resume=False,
                                schema_version=enrichment.schema_version,
                                pipeline_version=enrichment.pipeline_version,
                            )
                            manifest = output.with_suffix(".manifest.json")
                            client.upload_file(str(output), storage.bucket, parquet_key)
                            client.upload_file(str(manifest), storage.bucket, manifest_key)
                            processed += 1
                        except Exception as error:  # per-video failure boundary
                            failed += 1
                            print(
                                f"ASR FAILED video={source.video_id} "
                                f"error={type(error).__name__}: {error}"
                            )
                references.cleanup()
    finally:
        endpoint.close()
        if references is not None:
            references.cleanup()

    summary = {
        "selected": len(sources),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "output_prefix": output_prefix,
    }
    print(json.dumps(summary, indent=2))
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
