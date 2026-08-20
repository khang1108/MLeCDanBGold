"""Composition tests for the explicit fast-track retrieval factory.

The fixtures use tiny exact indexes so these tests exercise only online
composition.  They do not build corpus artifacts or load production models.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest

from hcmai.common.config import (
    EncoderConfig,
    FusionConfig,
    RetrievalCacheConfig,
)
from hcmai.common.schemas import RetrievalSource
from hcmai.data.stores.frame import FrameStore
from hcmai.retrieval.retriever.dense.index import DenseIndex
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex


class FakeEncoder:
    """Return deterministic normalized query vectors and record batch calls."""

    embedding_dim = 2
    resolved_revision = "a" * 40

    def __init__(
        self,
        model_name: str,
        backend: Literal["siglip", "bge_m3"],
    ) -> None:
        self.config = EncoderConfig(
            backend=backend,
            model_name=model_name,
            revision=self.resolved_revision,
        )
        self.calls: list[list[str]] = []

    def encode_text(self, texts, stats=None) -> np.ndarray:
        """Map cable-car queries to the first axis and all others to the second."""

        del stats
        self.calls.append(list(texts))
        return np.asarray(
            [
                [1.0, 0.0] if "cable" in text.lower() else [0.0, 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )


def _frame_mapping() -> pd.DataFrame:
    """Return two canonical frame rows shared by the frame-native indexes."""

    return pd.DataFrame(
        [
            {
                "embedding_index": 0,
                "frame_id": "f1",
                "video_id": "v1",
                "frame_idx": 101,
                "timestamp_ms": 1_500,
            },
            {
                "embedding_index": 1,
                "frame_id": "f2",
                "video_id": "v1",
                "frame_idx": 202,
                "timestamp_ms": 5_000,
            },
        ]
    )


def _frame_store(tmp_path: Path) -> FrameStore:
    """Persist the canonical coordinates used for ASR segment projection."""

    pytest.importorskip("pyarrow")
    path = tmp_path / "frames.parquet"
    pd.DataFrame(
        [
            {
                "frame_id": "f1",
                "video_id": "v1",
                "frame_idx": 101,
                "timestamp_ms": 1_500,
                "image_path": "frames/f1.jpg",
                "width": 640,
                "height": 360,
            },
            {
                "frame_id": "f2",
                "video_id": "v1",
                "frame_idx": 202,
                "timestamp_ms": 5_000,
                "image_path": "frames/f2.jpg",
                "width": 640,
                "height": 360,
            },
        ]
    ).to_parquet(path, index=False)
    return FrameStore(path)


def _indexes() -> tuple[DenseIndex, DenseIndex, SegmentDenseIndex]:
    """Build visual, Context, and timeline ASR indexes with aligned fixtures."""

    pytest.importorskip("faiss")
    visual = DenseIndex.build(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        _frame_mapping(),
        dataset_version="test-v1",
        model_name="fake/siglip",
        show_progress=False,
    )
    context = DenseIndex.build(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        _frame_mapping(),
        dataset_version="test-v1",
        model_name="fake/bge",
        show_progress=False,
    )
    segments = SegmentDenseIndex.build(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        pd.DataFrame(
            [
                {
                    "embedding_index": 0,
                    "segment_id": "v1:strong",
                    "video_id": "v1",
                    "segment_index": 0,
                    "start_ms": 1_000,
                    "end_ms": 2_000,
                },
                {
                    "embedding_index": 1,
                    "segment_id": "v1:second",
                    "video_id": "v1",
                    "segment_index": 1,
                    "start_ms": 4_500,
                    "end_ms": 5_500,
                },
            ]
        ),
        dataset_version="test-v1",
        model_name="fake/bge",
    )
    return visual, context, segments


def _encoders() -> tuple[FakeEncoder, FakeEncoder]:
    """Return distinct visual-query and generic evidence-query encoders."""

    return (
        FakeEncoder("fake/siglip", "siglip"),
        FakeEncoder("fake/bge", "bge_m3"),
    )


def _service(
    tmp_path: Path,
    *,
    include_context: bool = True,
    include_asr: bool = True,
    cache_config: RetrievalCacheConfig | None = None,
) -> tuple[RetrievalService, FakeEncoder, FakeEncoder, DenseIndex]:
    """Compose the fast-track service with selected optional evidence indexes."""

    visual, context, segments = _indexes()
    visual_encoder, text_encoder = _encoders()
    service = RetrievalService.from_fast_track_indexes(
        visual_index=visual,
        visual_encoder=visual_encoder,
        context_index=context if include_context else None,
        asr_segment_index=segments if include_asr else None,
        text_encoder=text_encoder if include_context or include_asr else None,
        frame_store=_frame_store(tmp_path),
        fusion=FusionConfig(required_sources={RetrievalSource.VISUAL}),
        cache_config=cache_config,
        max_projection_gap_ms=1_000,
    )
    return service, visual_encoder, text_encoder, visual


def test_fast_track_service_returns_frame_native_fused_candidates(
    tmp_path: Path,
) -> None:
    """Fuse all modalities by canonical frame ID after ASR projection."""

    service, _, text_encoder, _ = _service(tmp_path)

    result = service.search("red cable car", top_k=10)

    assert service.active_sources == (
        RetrievalSource.VISUAL,
        RetrievalSource.CONTEXT,
        RetrievalSource.ASR,
    )
    assert tuple(
        type(retriever).__name__ for retriever in service._retriever.retrievers  # type: ignore[attr-defined]
    ) == ("DenseRetriever", "ContextRetriever", "ASRSegmentRetriever")
    assert [candidate.frame_id for candidate in result.candidates] == ["f1", "f2"]
    first = result.candidates[0]
    assert set(first.source_ranks) == {
        RetrievalSource.VISUAL,
        RetrievalSource.CONTEXT,
        RetrievalSource.ASR,
    }
    assert first.metadata["frame"] == {
        "frame_id": "f1",
        "video_id": "v1",
        "frame_idx": 101,
        "timestamp_ms": 1_500,
    }
    assert first.metadata["asr_segment"]["segment_id"] == "v1:strong"
    assert text_encoder.calls == [["red cable car"]]


@pytest.mark.parametrize(
    ("include_context", "include_asr", "expected_sources"),
    [
        (False, True, (RetrievalSource.VISUAL, RetrievalSource.ASR)),
        (True, False, (RetrievalSource.VISUAL, RetrievalSource.CONTEXT)),
        (False, False, (RetrievalSource.VISUAL,)),
    ],
)
def test_fast_track_optional_indexes_are_independent(
    tmp_path: Path,
    include_context: bool,
    include_asr: bool,
    expected_sources: tuple[RetrievalSource, ...],
) -> None:
    """Keep Visual required while either or both optional indexes are absent."""

    service, _, _, _ = _service(
        tmp_path,
        include_context=include_context,
        include_asr=include_asr,
    )

    assert service.active_sources == expected_sources
    assert service.search("red cable car", top_k=2).candidates
    if not include_context and not include_asr:
        assert not hasattr(service._retriever, "retrievers")


@pytest.mark.parametrize("include_context,include_asr", [(True, False), (False, True)])
def test_fast_track_requires_text_encoder_for_any_text_index(
    tmp_path: Path, include_context: bool, include_asr: bool
) -> None:
    """Reject an optional text index when no compatible query encoder exists."""

    visual, context, segments = _indexes()
    visual_encoder, _ = _encoders()

    with pytest.raises(ValueError, match="text_encoder"):
        RetrievalService.from_fast_track_indexes(
            visual_index=visual,
            visual_encoder=visual_encoder,
            context_index=context if include_context else None,
            asr_segment_index=segments if include_asr else None,
            text_encoder=None,
            frame_store=_frame_store(tmp_path),
            fusion=FusionConfig(required_sources={RetrievalSource.VISUAL}),
        )


def test_fast_track_retrievers_share_one_enabled_cache(tmp_path: Path) -> None:
    """Reuse one cache and prompt version across all composed retrievers."""

    cache_config = RetrievalCacheConfig(enabled=True, prompt_version="fast-track-v1")
    service, _, _, _ = _service(tmp_path, cache_config=cache_config)
    retrievers = service._retriever.retrievers  # type: ignore[attr-defined]

    caches = [retriever.embedding_cache for retriever in retrievers]
    assert caches[0] is not None
    assert all(cache is caches[0] for cache in caches)
    assert {retriever.prompt_version for retriever in retrievers} == {"fast-track-v1"}


def test_fast_track_visual_scoring_uses_visual_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Protect TRAKE's visual-only video scoring path in multimodal services."""

    service, visual_encoder, _, visual_index = _service(tmp_path)
    captured: dict[str, Any] = {}

    def fake_score(index, vectors, *args):
        captured.update(index=index, vectors=vectors, args=args)
        return []

    monkeypatch.setattr(
        "hcmai.retrieval.retriever.pipeline.score_videos", fake_score
    )

    assert service.score_visual_videos(["red cable car"]) == []
    assert captured["index"] is visual_index
    assert visual_encoder.calls == [["red cable car"]]


def test_offline_index_cli_all_runs_strict_sequential_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Build all corpora in GPU-safe order and validate only after publication."""

    from scripts import build_retrieval_indexes as workflow

    events: list[str] = []
    projected_frames = tmp_path / "projected.parquet"
    projected_frames.write_bytes(b"preflight-projection")
    config = SimpleNamespace(projected_frames_path=projected_frames)
    models = object()
    text_encoder = object()

    monkeypatch.setattr(workflow, "load_offline_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(workflow, "load_model_config", lambda *args, **kwargs: models)
    monkeypatch.setattr(
        workflow,
        "run_preflight",
        lambda received: events.append("preflight") or projected_frames,
    )
    monkeypatch.setattr(
        workflow,
        "build_visual",
        lambda received, received_models, frames: events.append("visual"),
    )
    monkeypatch.setattr(
        workflow,
        "create_text_encoder",
        lambda received_models: text_encoder,
    )
    monkeypatch.setattr(
        workflow,
        "build_context",
        lambda received, received_models, frames, encoder=None: events.append(
            "context"
        ),
    )
    monkeypatch.setattr(
        workflow,
        "build_asr",
        lambda received, received_models, encoder=None: events.append("asr"),
    )
    monkeypatch.setattr(
        workflow,
        "run_validate",
        lambda received, received_models: events.append("validate"),
    )
    monkeypatch.setattr(workflow, "release_gpu_memory", lambda: None)

    workflow.run(workflow.parse_args(["--stage", "all"]))

    assert events == ["preflight", "visual", "context", "asr", "validate"]


def test_offline_index_cli_context_does_not_build_other_modalities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Permit one text-index rebuild without loading Visual or ASR builders."""

    from scripts import build_retrieval_indexes as workflow

    events: list[str] = []
    projected_frames = tmp_path / "projected.parquet"
    projected_frames.write_bytes(b"preflight-projection")
    config = SimpleNamespace(projected_frames_path=projected_frames)

    monkeypatch.setattr(workflow, "load_offline_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(workflow, "load_model_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(workflow, "build_visual", lambda *args, **kwargs: events.append("visual"))
    monkeypatch.setattr(workflow, "build_context", lambda *args, **kwargs: events.append("context"))
    monkeypatch.setattr(workflow, "build_asr", lambda *args, **kwargs: events.append("asr"))

    workflow.run(workflow.parse_args(["--stage", "context"]))

    assert events == ["context"]


def test_offline_projection_resolves_relative_keyframe_root_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep a config-relative keyframe path directly readable by the builder."""

    from scripts import build_retrieval_indexes as workflow
    from hcmai.retrieval.embedding.artifacts import EmbeddingArtifactBuilder

    monkeypatch.chdir(tmp_path)
    keyframes_root = Path("data/keyframes")
    video_root = keyframes_root / "v1"
    video_root.mkdir(parents=True)
    image = video_root / "000001.jpg"
    image.write_bytes(b"fixture-image")
    frames = pd.DataFrame(
        [
            {
                "frame_id": "f1",
                "video_id": "v1",
                "frame_idx": 10,
                "timestamp_ms": 400,
                "keyframe_order": 1,
                "image_path": "machine-specific/old.jpg",
            }
        ]
    )

    projected = workflow.project_staged_keyframes(frames, keyframes_root)
    builder = EmbeddingArtifactBuilder(
        frames_path=tmp_path / "unused.parquet",
        dataset_root=keyframes_root,
        output_dir=tmp_path / "output",
        encoder_config=EncoderConfig(),
        encoder=object(),  # type: ignore[arg-type]
    )

    assert Path(projected.iloc[0]["image_path"]).is_absolute()
    assert builder._resolve_image(projected.iloc[0]["image_path"]) == image.resolve()


def test_offline_model_config_rejects_mutable_revisions(tmp_path: Path) -> None:
    """Refuse branch aliases that cannot reproduce a published index."""

    from scripts import build_retrieval_indexes as workflow

    models = tmp_path / "models.yaml"
    models.write_text(
        """
visual_embedding:
  backend: siglip
  model_name: fake/visual
  revision: main
evidence_embedding:
  backend: bge_m3
  model_name: fake/text
  revision: 5617a9f61b028005a4858fdac845db406aefb181
""".strip()
    )

    with pytest.raises(ValueError, match="40-character hexadecimal"):
        workflow.load_model_config(models)


def test_offline_preflight_rejects_evidence_with_no_usable_corpus() -> None:
    """Stop before model loading when Context and transcript builders would fail."""

    from scripts import build_retrieval_indexes as workflow
    from hcmai.common.schemas import ProcessingStatus

    context = SimpleNamespace(frame_id="f1")
    data = SimpleNamespace(
        iter_frame_contexts=lambda: iter((context,)),
        get_frame_context_text=lambda frame_id: None,
    )
    with pytest.raises(ValueError, match="no usable context_text"):
        workflow._require_usable_context_ids(data)

    failed_segment = SimpleNamespace(
        status=ProcessingStatus.FAILED,
        text="recognizable words",
    )
    with pytest.raises(ValueError, match="no usable completed segments"):
        workflow._require_usable_completed_segments((failed_segment,))


def test_index_pull_validates_then_atomically_promotes_all_bundles(
    tmp_path: Path,
) -> None:
    """Keep live indexes intact until report and all staged transfers succeed."""

    repository = Path(__file__).resolve().parents[2]
    script = repository / "scripts/sync_thundercompute_indexes.sh"
    local_root = tmp_path / "local"
    live_root = local_root / "artifacts/indexes"
    remote_root = tmp_path / "remote"
    remote_indexes = remote_root / "artifacts/indexes"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (local_root / "src").parent.mkdir(parents=True)
    (local_root / "src").symlink_to(repository / "src", target_is_directory=True)

    for name in ("visual", "context", "asr_segments"):
        (live_root / name).mkdir(parents=True)
        (live_root / name / "old.txt").write_text(f"old-{name}")
    (live_root / "build_report.json").write_text('{"status":"old"}')

    revision = "a" * 40
    dataset_version = "test-v1"
    frame_mapping = pd.DataFrame(
        [
            {
                "embedding_index": 0,
                "frame_id": "f1",
                "video_id": "v1",
                "frame_idx": 10,
                "timestamp_ms": 400,
            }
        ]
    )
    segment_mapping = pd.DataFrame(
        [
            {
                "embedding_index": 0,
                "segment_id": "s1",
                "video_id": "v1",
                "segment_index": 0,
                "start_ms": 100,
                "end_ms": 500,
            }
        ]
    )
    vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)

    def publish_remote_bundles(
        *, source_fingerprint: str = "source-v1"
    ) -> None:
        visual = DenseIndex.build(
            vectors,
            frame_mapping,
            dataset_version=dataset_version,
            model_name="fake/visual",
        )
        visual.metadata.retrieval_source = "visual"
        visual.metadata.model_revision = revision
        visual.metadata.source_fingerprint = source_fingerprint
        visual.metadata.config_fingerprint = "config-v1"
        visual.save(remote_indexes / "visual")

        context = DenseIndex.build(
            vectors,
            frame_mapping,
            dataset_version=dataset_version,
            model_name="fake/text",
        )
        context.metadata.retrieval_source = "context"
        context.metadata.model_revision = revision
        context.metadata.source_fingerprint = source_fingerprint
        context.metadata.config_fingerprint = "config-v1"
        context.save(remote_indexes / "context")

        asr = SegmentDenseIndex.build(
            vectors,
            segment_mapping,
            dataset_version=dataset_version,
            model_name="fake/text",
        )
        asr.metadata.model_revision = revision
        asr.metadata.source_fingerprint = source_fingerprint
        asr.metadata.config_fingerprint = "config-v1"
        asr.save(remote_indexes / "asr_segments")

    publish_remote_bundles()

    fake_rsync = fake_bin / "rsync"
    fake_rsync.write_text(
        """#!/usr/bin/env python3
import os
from pathlib import Path
import shutil
import sys

source, destination = sys.argv[-2:]
failure = os.environ.get("HCMAI_FAKE_RSYNC_FAIL")
if failure and failure in source:
    raise SystemExit(23)
remote_path = Path(source.split(":", 1)[1])
requested_root = Path(os.environ["HCMAI_THUNDER_ROOT"])
actual = Path(os.environ["HCMAI_FAKE_REMOTE_ROOT"]) / remote_path.relative_to(requested_root)
target = Path(destination)
if actual.is_dir():
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(actual, target, dirs_exist_ok=True)
else:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(actual, target)
"""
    )
    fake_rsync.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "HCMAI_PYTHON": sys.executable,
        "HCMAI_LOCAL_ROOT": str(local_root),
        "HCMAI_THUNDER_HOST": "fake-host",
        "HCMAI_THUNDER_ROOT": "/remote/hcmai",
        "HCMAI_FAKE_REMOTE_ROOT": str(remote_root),
    }

    def bundle_size(name: str) -> int:
        return sum(
            path.stat().st_size
            for path in (remote_indexes / name).rglob("*")
            if path.is_file()
        )

    def index_report(name: str, model_name: str) -> dict[str, object]:
        index = (
            SegmentDenseIndex.load(remote_indexes / name)
            if name == "asr_segments"
            else DenseIndex.load(remote_indexes / name)
        )
        metadata = index.metadata
        return {
            "path": f"/remote/only/{name}",
            "vector_count": 1,
            "model_name": model_name,
            "model_revision": revision,
            "embedding_dim": 2,
            "normalization": "l2",
            "size_bytes": bundle_size(name),
            "schema_version": metadata.schema_version,
            "entity_kind": metadata.entity_kind,
            "retrieval_source": metadata.retrieval_source,
            "source_fingerprint": metadata.source_fingerprint,
            "config_fingerprint": metadata.config_fingerprint,
            "checksums": metadata.checksums,
        }

    def write_report(status: str) -> None:
        (remote_indexes / "build_report.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "dataset_version": dataset_version,
                    "indexes": {
                        "visual": index_report("visual", "fake/visual"),
                        "context": index_report("context", "fake/text"),
                        "asr_segments": index_report("asr_segments", "fake/text"),
                    },
                }
            )
        )

    def pull(**extra_env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(script), "pull-indexes"],
            env={**env, **extra_env},
            capture_output=True,
            text=True,
            check=False,
        )

    def assert_live_is_old() -> None:
        for name in ("visual", "context", "asr_segments"):
            assert (live_root / name / "old.txt").read_text() == f"old-{name}"
        assert json.loads((live_root / "build_report.json").read_text()) == {
            "status": "old"
        }

    assert pull().returncode != 0
    assert_live_is_old()

    write_report("failed")
    assert pull().returncode != 0
    assert_live_is_old()

    (remote_indexes / "build_report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "indexes": {"visual": {}, "context": {}},
            }
        )
    )
    assert pull().returncode != 0
    assert_live_is_old()

    write_report("passed")
    assert pull(HCMAI_FAKE_RSYNC_FAIL="context/").returncode != 0
    assert_live_is_old()

    # A complete valid index from different source provenance is still stale
    # relative to the fetched report and must not replace the live bundle.
    visual_metadata = remote_indexes / "visual/metadata.json"
    visual_metadata.write_text(
        visual_metadata.read_text().replace("source-v1", "source-v2")
    )
    DenseIndex.load(remote_indexes / "visual")
    assert pull().returncode != 0
    assert_live_is_old()

    publish_remote_bundles()
    write_report("passed")
    # A status=passed report from before the mutation must not authorize a
    # mixed bundle whose checksum/byte-size contracts no longer match.
    with (remote_indexes / "visual/vectors.npy").open("ab") as handle:
        handle.write(b"tamper")
    assert pull().returncode != 0
    assert_live_is_old()

    publish_remote_bundles()
    write_report("passed")
    metadata_path = remote_indexes / "visual/metadata.json"
    legacy_metadata = json.loads(metadata_path.read_text())
    legacy_metadata["schema_version"] = "dense-index-v1"
    legacy_metadata["checksums"] = None
    metadata_path.write_text(json.dumps(legacy_metadata))
    assert pull().returncode != 0
    assert_live_is_old()

    publish_remote_bundles()
    write_report("passed")
    assert pull().returncode == 0
    DenseIndex.load(live_root / "visual")
    DenseIndex.load(live_root / "context")
    SegmentDenseIndex.load(live_root / "asr_segments")
    for name in ("visual", "context", "asr_segments"):
        assert not (live_root / name / "old.txt").exists()
    assert json.loads((live_root / "build_report.json").read_text())["status"] == "passed"
