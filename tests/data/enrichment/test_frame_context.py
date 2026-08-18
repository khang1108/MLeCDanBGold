"""Tests for deterministic, dependency-aware FrameContext V1 artifacts."""

from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from hcmai.common.schemas import CaptionEvidence, OCREvidence
from hcmai.data.enrichment.context import (
    FrameContextConfig,
    build_frame_context,
    serialize_frame_context,
)
from hcmai.data.enrichment.pipeline import EnrichmentService


def test_context_section_order_and_missing_omission() -> None:
    """Serialize usable specialist text in the frozen V1 section order."""

    text = serialize_frame_context(
        caption="A man speaks.",
        ocr="AIC 2026",
        objects="person x2; microphone x1",
        config=FrameContextConfig(),
    )

    assert text == (
        "[CAPTION]\nA man speaks.\n\n"
        "[VISIBLE_TEXT]\nAIC 2026\n\n"
        "[OBJECTS]\nperson x2; microphone x1"
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"caption": " caption only ", "ocr": None, "objects": None},
            "[CAPTION]\ncaption only",
        ),
        (
            {"caption": None, "ocr": " OCR only ", "objects": None},
            "[VISIBLE_TEXT]\nOCR only",
        ),
        (
            {"caption": None, "ocr": None, "objects": " object only "},
            "[OBJECTS]\nobject only",
        ),
        (
            {"caption": "  ", "ocr": "", "objects": None},
            None,
        ),
    ],
)
def test_context_independent_sections(
    kwargs: dict[str, str | None], expected: str | None
) -> None:
    """Omit absent sections without emitting placeholders."""

    assert serialize_frame_context(config=FrameContextConfig(), **kwargs) == expected


def test_context_exact_whitespace_token_budgets_and_bytes() -> None:
    """Apply exact 80/80/40 budgets with byte-identical output."""

    caption = "  \n".join(f"c{index}" for index in range(81))
    ocr = "\t".join(f"o{index}" for index in range(81))
    objects = "  ".join(f"x{index}" for index in range(41))
    kwargs = {"caption": caption, "ocr": ocr, "objects": objects}

    first = serialize_frame_context(config=FrameContextConfig(), **kwargs)
    second = serialize_frame_context(config=FrameContextConfig(), **kwargs)

    assert first is not None and second is not None
    assert first.encode("utf-8") == second.encode("utf-8")
    assert "c79" in first and "c80" not in first
    assert "o79" in first and "o80" not in first
    assert "x39" in first and "x40" not in first
    assert "SPEECH" not in first and "ASR" not in first


@pytest.mark.parametrize(
    "updates",
    [
        {"caption_token_budget": -1},
        {"ocr_token_budget": -1},
        {"object_token_budget": -1},
        {"min_ocr_quality": -0.01},
        {"min_ocr_quality": 1.01},
    ],
)
def test_context_config_rejects_invalid_policy(updates: dict[str, object]) -> None:
    """Reject invalid token budgets and OCR quality thresholds."""

    with pytest.raises(ValueError):
        FrameContextConfig(**updates)


def test_context_config_canonicalizes_padded_version() -> None:
    """Store one canonical version string in both rows and dependency identity."""

    config = FrameContextConfig(context_version="  frame-context-v2  ")

    assert config.context_version == "frame-context-v2"


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _artifacts(root: Path, *, count: int = 2) -> tuple[Path, Path, Path, Path]:
    frames = root / "frames.parquet"
    pd.DataFrame(
        [
            {
                "frame_id": f"f{index}",
                "video_id": "v1",
                "frame_idx": index,
                "timestamp_ms": index * 1000,
                "image_path": f"f{index}.jpg",
                "width": 10,
                "height": 10,
            }
            for index in range(count)
        ]
    ).to_parquet(frames, index=False)
    _write_json(root / "manifest.json", {"frame_store_id": "btc-v1"})

    caption = root / "caption/captions.parquet"
    caption.parent.mkdir()
    pd.DataFrame(
        [
            CaptionEvidence(
                frame_id=f"f{index}",
                video_id="v1",
                frame_idx=index,
                text=f"caption {index}",
                frame_store_id="btc-v1",
                artifact_version="caption-v1",
                model_name="captioner",
            ).model_dump(mode="json")
            for index in range(count)
        ]
    ).to_parquet(caption, index=False)
    _write_json(
        caption.parent / "manifest.json",
        {"artifact_version": "caption-v1", "frame_store_id": "btc-v1"},
    )

    ocr = root / "ocr/frames.parquet"
    ocr.parent.mkdir()
    pd.DataFrame(
        [
            OCREvidence(
                frame_id=f"f{index}",
                video_id="v1",
                frame_idx=index,
                normalized_text=f"ocr {index}",
                quality_score=0.75,
                frame_store_id="btc-v1",
                artifact_version="ocr-v1",
                model_name="ocr",
            ).model_dump(mode="json")
            for index in range(count)
        ]
    ).to_parquet(ocr, index=False)
    _write_json(
        ocr.parent / "manifest.json",
        {"artifact_version": "ocr-v1", "frame_store_id": "btc-v1"},
    )

    objects = root / "objects/frames.parquet"
    objects.parent.mkdir()
    pd.DataFrame(
        [
            {
                "frame_id": f"f{index}",
                "video_id": "v1",
                "frame_idx": index,
                "counts_json": json.dumps({"person": index + 1}),
                "summary": f"person x{index + 1}",
                "detection_count": index + 1,
                "frame_store_id": "btc-v1",
                "artifact_version": "object-v1",
                "status": "completed",
                "error_code": None,
                "error_message": None,
            }
            for index in range(count)
        ]
    ).to_parquet(objects, index=False)
    _write_json(
        objects.parent / "manifest.json",
        {"artifact_version": "object-v1", "frame_store_id": "btc-v1"},
    )
    return frames, caption, ocr, objects


def _build(root: Path, *, config: FrameContextConfig | None = None) -> Path:
    frames, caption, ocr, objects = _artifacts(root)
    return build_frame_context(
        frames,
        caption,
        ocr,
        objects,
        root / "context",
        config or FrameContextConfig(),
        frame_store_id="btc-v1",
    )


def test_builder_writes_canonical_rows_and_lineage(tmp_path: Path) -> None:
    """Build one context row per canonical frame with upstream versions."""

    path = _build(tmp_path)
    rows = pd.read_parquet(path)
    manifest = json.loads((path.parent / "manifest.json").read_text())

    assert rows.frame_id.tolist() == ["f0", "f1"]
    assert rows.caption_version.tolist() == ["caption-v1", "caption-v1"]
    assert rows.ocr_version.tolist() == ["ocr-v1", "ocr-v1"]
    assert rows.object_version.tolist() == ["object-v1", "object-v1"]
    assert rows.object_count.tolist() == [1, 2]
    assert manifest == {
        "context_version": "frame-context-v1",
        "caption_version": "caption-v1",
        "ocr_version": "ocr-v1",
        "object_version": "object-v1",
        "frame_store_id": "btc-v1",
        "serializer_config": {
            "caption_token_budget": 80,
            "ocr_token_budget": 80,
            "object_token_budget": 40,
            "min_ocr_quality": 0.5,
        },
    }
    assert "asr" not in json.dumps(manifest).casefold()


def test_missing_failed_and_low_quality_evidence_is_omitted(tmp_path: Path) -> None:
    """Contain specialist failures and low-quality OCR at frame granularity."""

    frames, caption, ocr, objects = _artifacts(tmp_path)
    caption_rows = pd.read_parquet(caption).iloc[:1]
    caption_rows.to_parquet(caption, index=False)
    ocr_rows = pd.read_parquet(ocr)
    ocr_rows.loc[0, "quality_score"] = 0.49
    ocr_rows.to_parquet(ocr, index=False)
    object_rows = pd.read_parquet(objects)
    object_rows.loc[1, "status"] = "failed"
    object_rows.loc[1, "error_code"] = "Missing"
    object_rows.loc[1, "error_message"] = "missing"
    object_rows.to_parquet(objects, index=False)

    path = build_frame_context(
        frames, caption, ocr, objects, tmp_path / "context", FrameContextConfig()
    )
    rows = pd.read_parquet(path).set_index("frame_id")

    assert "[VISIBLE_TEXT]" not in rows.loc["f0", "context_text"]
    assert "[CAPTION]" not in rows.loc["f1", "context_text"]
    assert "[OBJECTS]" not in rows.loc["f1", "context_text"]


def test_completed_ocr_and_objects_keep_text_with_diagnostic_metadata(
    tmp_path: Path,
) -> None:
    """Use completed OCR/Object text even when rows retain diagnostics."""

    frames, caption, ocr, objects = _artifacts(tmp_path)
    for path in (ocr, objects):
        rows = pd.read_parquet(path)
        rows.loc[0, "error_code"] = "Diagnostic"
        rows.loc[0, "error_message"] = "non-fatal diagnostic"
        rows.to_parquet(path, index=False)

    path = build_frame_context(
        frames, caption, ocr, objects, tmp_path / "context", FrameContextConfig()
    )
    context = pd.read_parquet(path).set_index("frame_id").loc["f0"]

    assert "[VISIBLE_TEXT]\nocr 0" in context["context_text"]
    assert "[OBJECTS]\nperson x1" in context["context_text"]


def test_matching_identity_reuses_and_policy_or_upstream_version_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reuse only exact valid bundles and invalidate context dependencies."""

    path = _build(tmp_path)
    before = path.read_bytes()

    def unexpected_write(*args: object, **kwargs: object) -> None:
        raise AssertionError("matching bundle should not be republished")

    monkeypatch.setattr(
        "hcmai.data.enrichment.context.builder._publish_staged_bundle",
        unexpected_write,
    )
    frames = tmp_path / "frames.parquet"
    for asr_version in ("asr-v1", "asr-v2"):
        _write_json(
            tmp_path / "transcripts/manifest.json",
            {"artifact_version": asr_version},
        )
        build_frame_context(
            frames,
            tmp_path / "caption/captions.parquet",
            tmp_path / "ocr/frames.parquet",
            tmp_path / "objects/frames.parquet",
            tmp_path / "context",
            FrameContextConfig(),
        )
    assert path.read_bytes() == before

    monkeypatch.undo()
    changed = replace(FrameContextConfig(), object_token_budget=1)
    build_frame_context(
        frames,
        tmp_path / "caption/captions.parquet",
        tmp_path / "ocr/frames.parquet",
        tmp_path / "objects/frames.parquet",
        tmp_path / "context",
        changed,
    )
    manifest = json.loads((tmp_path / "context/manifest.json").read_text())
    assert manifest["serializer_config"]["object_token_budget"] == 1

    caption_manifest = tmp_path / "caption/manifest.json"
    _write_json(
        caption_manifest,
        {"artifact_version": "caption-v2", "frame_store_id": "btc-v1"},
    )
    captions = pd.read_parquet(tmp_path / "caption/captions.parquet")
    captions["artifact_version"] = "caption-v2"
    captions.to_parquet(tmp_path / "caption/captions.parquet", index=False)
    build_frame_context(
        frames,
        tmp_path / "caption/captions.parquet",
        tmp_path / "ocr/frames.parquet",
        tmp_path / "objects/frames.parquet",
        tmp_path / "context",
        changed,
    )
    manifest = json.loads((tmp_path / "context/manifest.json").read_text())
    assert manifest["caption_version"] == "caption-v2"


@pytest.mark.parametrize(
    ("field", "corrupt_value"),
    [
        ("caption_text", "corrupt caption"),
        ("ocr_text", "corrupt OCR"),
        ("object_summary", "corrupt objects"),
        ("context_text", "corrupt context"),
        ("caption_available", False),
        ("ocr_quality", 0.1),
        ("object_count", 99),
    ],
)
def test_schema_valid_context_corruption_forces_rebuild(
    tmp_path: Path, field: str, corrupt_value: object
) -> None:
    """Compare every derived context field before accepting a resumed bundle."""

    path = _build(tmp_path)
    expected = pd.read_parquet(path)
    corrupted = expected.copy()
    corrupted.loc[0, field] = corrupt_value
    corrupted.to_parquet(path, index=False)

    build_frame_context(
        tmp_path / "frames.parquet",
        tmp_path / "caption/captions.parquet",
        tmp_path / "ocr/frames.parquet",
        tmp_path / "objects/frames.parquet",
        path.parent,
        FrameContextConfig(),
        frame_store_id="btc-v1",
    )

    pd.testing.assert_frame_equal(pd.read_parquet(path), expected)


@pytest.mark.parametrize("source", ["canonical", "caption", "ocr", "object"])
def test_duplicate_or_foreign_identity_is_rejected(
    tmp_path: Path, source: str
) -> None:
    """Reject ambiguous rows and specialist frames outside the canonical store."""

    frames, caption, ocr, objects = _artifacts(tmp_path)
    paths = {"canonical": frames, "caption": caption, "ocr": ocr, "object": objects}
    path = paths[source]
    table = pd.read_parquet(path)
    if source == "canonical":
        table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    else:
        table.loc[0, "frame_id"] = "foreign"
    table.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="(?i)(duplicate|foreign|canonical)"):
        build_frame_context(
            frames,
            caption,
            ocr,
            objects,
            tmp_path / "context",
            FrameContextConfig(),
        )


@pytest.mark.parametrize("source", ["caption", "ocr", "object"])
def test_duplicate_specialist_rows_are_rejected(
    tmp_path: Path, source: str
) -> None:
    """Reject duplicate specialist rows before joining canonical evidence."""

    frames, caption, ocr, objects = _artifacts(tmp_path)
    paths = {"caption": caption, "ocr": ocr, "object": objects}
    path = paths[source]
    rows = pd.read_parquet(path)
    pd.concat([rows, rows.iloc[[0]]], ignore_index=True).to_parquet(
        path, index=False
    )

    with pytest.raises(ValueError, match=f"(?i){source}.*duplicate"):
        build_frame_context(
            frames,
            caption,
            ocr,
            objects,
            tmp_path / "context",
            FrameContextConfig(),
        )


def test_lineage_mismatch_is_rejected_before_existing_bundle_is_touched(
    tmp_path: Path,
) -> None:
    """Preserve a prior context bundle when prerequisite lineage is invalid."""

    path = _build(tmp_path)
    before = {
        name: (path.parent / name).read_bytes()
        for name in ("frame_context_v1.parquet", "manifest.json")
    }
    captions = pd.read_parquet(tmp_path / "caption/captions.parquet")
    captions.loc[0, "frame_store_id"] = "other-store"
    captions.to_parquet(tmp_path / "caption/captions.parquet", index=False)

    with pytest.raises(ValueError, match="(?i)lineage"):
        build_frame_context(
            tmp_path / "frames.parquet",
            tmp_path / "caption/captions.parquet",
            tmp_path / "ocr/frames.parquet",
            tmp_path / "objects/frames.parquet",
            path.parent,
            FrameContextConfig(),
        )

    assert {
        name: (path.parent / name).read_bytes()
        for name in ("frame_context_v1.parquet", "manifest.json")
    } == before


@pytest.mark.parametrize("source", ["caption", "ocr", "object"])
def test_null_specialist_row_lineage_conflicts_with_resolved_store(
    tmp_path: Path, source: str
) -> None:
    """Reject null row lineage once the canonical store identity is known."""

    frames, caption, ocr, objects = _artifacts(tmp_path)
    paths = {"caption": caption, "ocr": ocr, "object": objects}
    path = paths[source]
    rows = pd.read_parquet(path)
    rows.loc[0, "frame_store_id"] = None
    rows.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="(?i)lineage"):
        build_frame_context(
            frames,
            caption,
            ocr,
            objects,
            tmp_path / "context",
            FrameContextConfig(),
        )


@pytest.mark.parametrize("source", ["caption", "ocr", "object"])
def test_null_specialist_manifest_lineage_conflicts_with_other_manifests(
    tmp_path: Path, source: str
) -> None:
    """Reject mixed null/non-null specialist manifest lineage deterministically."""

    frames, caption, ocr, objects = _artifacts(tmp_path)
    versions = {
        "caption": "caption-v1",
        "ocr": "ocr-v1",
        "object": "object-v1",
    }
    paths = {"caption": caption, "ocr": ocr, "object": objects}
    _write_json(
        paths[source].parent / "manifest.json",
        {"artifact_version": versions[source], "frame_store_id": None},
    )

    with pytest.raises(ValueError, match="(?i)lineage"):
        build_frame_context(
            frames,
            caption,
            ocr,
            objects,
            tmp_path / "context",
            FrameContextConfig(),
        )


@pytest.mark.parametrize("target", ["detection_count", "counts_json"])
@pytest.mark.parametrize("invalid", [1.5, True, "1"])
def test_object_counts_require_strict_integral_values(
    tmp_path: Path, target: str, invalid: object
) -> None:
    """Reject fractional, Boolean, and string object count representations."""

    frames, caption, ocr, objects = _artifacts(tmp_path)
    rows = pd.read_parquet(objects)
    if target == "detection_count":
        rows["counts_json"] = "{}"
        rows[target] = [invalid] * len(rows)
    else:
        rows["detection_count"] = 3
        rows[target] = [json.dumps({"person": invalid})] * len(rows)
    rows.to_parquet(objects, index=False)

    with pytest.raises(ValueError, match="(?i)integer"):
        build_frame_context(
            frames,
            caption,
            ocr,
            objects,
            tmp_path / "context",
            FrameContextConfig(),
        )


def test_publication_failure_restores_prior_complete_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restore data and manifest if either staged publication step fails."""

    path = _build(tmp_path)
    names = ("frame_context_v1.parquet", "manifest.json")
    before = {name: (path.parent / name).read_bytes() for name in names}
    original_replace = Path.replace
    injected = False

    def fail_manifest_once(source: Path, target: Path) -> Path:
        nonlocal injected
        if source.name == ".manifest.json.staged" and not injected:
            injected = True
            raise OSError("injected context publication failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_manifest_once)
    with pytest.raises(OSError, match="injected"):
        build_frame_context(
            tmp_path / "frames.parquet",
            tmp_path / "caption/captions.parquet",
            tmp_path / "ocr/frames.parquet",
            tmp_path / "objects/frames.parquet",
            path.parent,
            replace(FrameContextConfig(), object_token_budget=1),
        )

    assert injected
    assert {name: (path.parent / name).read_bytes() for name in names} == before


def test_malformed_prerequisite_fails_before_existing_bundle_is_touched(
    tmp_path: Path,
) -> None:
    """Reject a corrupt artifact manifest before staging context output."""

    path = _build(tmp_path)
    names = ("frame_context_v1.parquet", "manifest.json")
    before = {name: (path.parent / name).read_bytes() for name in names}
    (tmp_path / "ocr/manifest.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        build_frame_context(
            tmp_path / "frames.parquet",
            tmp_path / "caption/captions.parquet",
            tmp_path / "ocr/frames.parquet",
            tmp_path / "objects/frames.parquet",
            path.parent,
            FrameContextConfig(),
        )

    assert {name: (path.parent / name).read_bytes() for name in names} == before


def test_empty_canonical_store_is_rejected_before_output(tmp_path: Path) -> None:
    """Fail clearly on an empty canonical store without creating context output."""

    frames, caption, ocr, objects = _artifacts(tmp_path)
    pd.read_parquet(frames).iloc[0:0].to_parquet(frames, index=False)

    with pytest.raises(ValueError, match="at least one frame"):
        build_frame_context(
            frames,
            caption,
            ocr,
            objects,
            tmp_path / "context",
            FrameContextConfig(),
        )

    assert not (tmp_path / "context").exists()


def test_builder_does_not_modify_specialist_source_artifacts(tmp_path: Path) -> None:
    """Keep specialist evidence byte-for-byte unchanged after context build."""

    frames, caption, ocr, objects = _artifacts(tmp_path)
    sources = (
        caption,
        caption.parent / "manifest.json",
        ocr,
        ocr.parent / "manifest.json",
        objects,
        objects.parent / "manifest.json",
    )
    before = {path: path.read_bytes() for path in sources}

    build_frame_context(
        frames,
        caption,
        ocr,
        objects,
        tmp_path / "context",
        FrameContextConfig(),
    )

    assert {path: path.read_bytes() for path in sources} == before


def test_service_exposes_the_frozen_builder_interface() -> None:
    """Keep the orchestration facade exact and free of generation parameters."""

    assert list(inspect.signature(EnrichmentService.build_frame_context).parameters) == [
        "frames_path",
        "caption_path",
        "ocr_frames_path",
        "object_frames_path",
        "output_dir",
        "config",
        "frame_store_id",
    ]


def test_context_modules_have_no_speech_or_model_dependencies() -> None:
    """Protect the cheap derived artifact from inference and timeline imports."""

    package = (
        Path(__file__).resolve().parents[3]
        / "src/hcmai/data/enrichment/context"
    )
    forbidden = ("asr", "speech", "transcript", "transformer", "llm", "vlm")
    imports: list[str] = []
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")

    assert not any(
        fragment in name.casefold()
        for name in imports
        for fragment in forbidden
    )
