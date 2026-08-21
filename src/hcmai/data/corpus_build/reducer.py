"""Deterministic global index reduction from committed group bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from hcmai.common.schemas import RetrievalSource
from hcmai.retrieval.retriever.dense.index import DenseIndex


@dataclass(frozen=True, slots=True)
class CommittedGroup:
    """Đại diện cho một Group đã được xử lý xong và commit (upload đầy đủ) lên S3."""
    group_id: str
    run_id: str
    version_prefix: str


def _body_bytes(response: dict[str, Any]) -> bytes:
    body = response["Body"]
    value = body.read() if hasattr(body, "read") else body
    return bytes(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class S3GroupIndexReducer:
    """Tải tất cả các tệp vector (embeddings) và mapping rời rạc từ các Group trên S3,
    sau đó gộp (reduce) chúng lại thành một Global Dense Index duy nhất.
    Bao gồm kiểm tra tính nhất quán (checksum, tính duy nhất của frame_id, L2-norm).
    """

    def __init__(self, client: Any, *, bucket: str, work_root: Path) -> None:
        self.client = client
        self.bucket = bucket
        self.work_root = work_root

    def reduce(
        self,
        groups: Iterable[CommittedGroup],
        *,
        source: RetrievalSource,
        output_dir: Path,
        dataset_version: str,
        model_name: str,
    ) -> Path:
        ordered = sorted(groups, key=lambda item: item.group_id)
        if not ordered or len({item.group_id for item in ordered}) != len(ordered):
            raise ValueError("committed group set must be non-empty and unique")

        vectors: list[np.ndarray] = []
        mappings: list[pd.DataFrame] = []
        for group in ordered:
            manifest = self._verified_manifest(group)
            
            vector_item, mapping_item = _embedding_items(manifest, source)
            
            group_root = self.work_root / group.group_id / group.run_id / source.value
            vector_path = self._download(group, vector_item, group_root)
            mapping_path = self._download(group, mapping_item, group_root)
            
            value = np.load(vector_path, allow_pickle=False)
            
            mapping = pd.read_parquet(mapping_path)
            
            if value.ndim != 2 or len(value) != len(mapping):
                raise ValueError("group vector/mapping cardinality mismatch")
            
            positions = mapping["embedding_index"].to_numpy(dtype=np.int64)
            if sorted(positions.tolist()) != list(range(len(mapping))):
                raise ValueError("group embedding_index is not contiguous")
            
            order = np.argsort(positions, kind="stable")
            
            vectors.append(np.asarray(value[order], dtype=np.float32))
            mappings.append(mapping.iloc[order.tolist()].reset_index(drop=True))
        
        dimensions = {value.shape[1] for value in vectors}
        if len(dimensions) != 1:
            raise ValueError("committed groups use different embedding dimensions")
        
        combined_vectors = np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
        
        norms = np.linalg.norm(combined_vectors, axis=1)
        
        if not np.all(np.isfinite(combined_vectors)) or not np.allclose(
            norms, 1.0, atol=1e-4
        ):
            raise ValueError("global reducer requires finite L2-normalized vectors")
        
        combined_mapping = pd.concat(mappings, ignore_index=True)
        if combined_mapping["frame_id"].duplicated().any():
            raise ValueError("committed groups contain duplicate canonical frame IDs")
        
        combined_mapping["embedding_index"] = pd.Series(
            np.arange(len(combined_mapping), dtype=np.int64).tolist()
        )
        
        index = DenseIndex.build(
            combined_vectors,
            combined_mapping,
            dataset_version=dataset_version,
            model_name=model_name,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        index.save(output_dir)
        return output_dir / "dense.index"

    def _verified_manifest(self, group: CommittedGroup) -> dict[str, Any]:
        commit_key = f"{group.version_prefix.rstrip('/')}/COMMITTED.json"
        commit = json.loads(_body_bytes(self.client.get_object(
            Bucket=self.bucket, Key=commit_key
        )))
        if commit.get("group_id") != group.group_id or commit.get("run_id") != group.run_id:
            raise ValueError("group commit identity mismatch")
        manifest_key = str(commit["manifest_key"])
        manifest_bytes = _body_bytes(self.client.get_object(
            Bucket=self.bucket, Key=manifest_key
        ))
        if hashlib.sha256(manifest_bytes).hexdigest() != commit.get("manifest_sha256"):
            raise ValueError("group manifest checksum mismatch")
        manifest = json.loads(manifest_bytes)
        if manifest.get("group_id") != group.group_id or manifest.get("run_id") != group.run_id:
            raise ValueError("group manifest identity mismatch")
        return manifest

    def _download(
        self,
        group: CommittedGroup,
        item: dict[str, Any],
        root: Path,
    ) -> Path:
        relative = str(item["path"])
        path = root / Path(relative).name
        path.parent.mkdir(parents=True, exist_ok=True)
        key = f"{group.version_prefix.rstrip('/')}/{relative}"
        self.client.download_file(self.bucket, key, str(path))
        if path.stat().st_size != int(item["size"]) or _sha256(path) != item["sha256"]:
            raise ValueError("downloaded group artifact checksum mismatch")
        return path


def _embedding_items(
    manifest: dict[str, Any], source: RetrievalSource
) -> tuple[dict[str, Any], dict[str, Any]]:
    files = list(manifest.get("files", []))
    if source is RetrievalSource.VISUAL:
        vectors = [item for item in files if item["path"].endswith("embeddings/visual_embeddings.npy")]
        mappings = [item for item in files if item["path"].endswith("embeddings/frame_mapping.parquet")]
    else:
        prefix = f"embeddings/{source.value}/"
        vectors = [
            item for item in files
            if prefix in item["path"] and item["path"].endswith("_embeddings.npy")
        ]
        mappings = [
            item for item in files
            if prefix in item["path"] and item["path"].endswith("frame_mapping.parquet")
        ]
    if len(vectors) != 1 or len(mappings) != 1:
        raise ValueError(f"committed group lacks one {source.value} vector/mapping pair")
    return vectors[0], mappings[0]
