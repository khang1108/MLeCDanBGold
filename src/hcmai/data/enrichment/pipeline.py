"""Expose thin service boundaries for independent offline enrichment stages.

The service delegates Caption, OCR, BTC Object, and deterministic FrameContext
materialization. Specialist generation and context serialization remain owned
by their respective packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hcmai.common.utils.io import read_yaml
from hcmai.data.enrichment.caption.config import CaptionConfig, PROJECT_ROOT
from hcmai.data.enrichment.caption.adapters.qwen_vl import QwenVLCaptionAdapter
from hcmai.data.enrichment.caption.generator import generate_captions
from hcmai.data.enrichment.caption.models.contracts import CaptionAdapter
from hcmai.data.enrichment.context.builder import build_frame_context
from hcmai.data.enrichment.context.config import FrameContextConfig
from hcmai.data.enrichment.ocr.config import OCRConfig
from hcmai.data.enrichment.ocr.generator import generate_ocr
from hcmai.data.enrichment.ocr.models.contracts import OCRAdapter
from hcmai.data.enrichment.objects.config import ObjectConfig
from hcmai.data.enrichment.objects.importer import import_objects


@dataclass(frozen=True)
class EnrichmentJobConfig:
    """Validated paths and policies for independently runnable V1 stages."""

    dataset_version: str
    source: str
    btc_root: Path
    data_root: Path
    frame_store_id: str
    frames_path: Path
    frame_store_output: Path
    objects_root: Path
    caption_output_dir: Path
    caption: CaptionConfig
    ocr_output_dir: Path
    ocr: OCRConfig
    object_output_dir: Path
    objects: ObjectConfig
    transcript_output_dir: Path
    context_output_dir: Path
    context: FrameContextConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> EnrichmentJobConfig:
        """Load the complete Enrichment V1 contract from one YAML file."""

        raw = read_yaml(Path(path).expanduser().resolve())
        if not isinstance(raw, dict):
            raise ValueError("Enrichment YAML requires a mapping")

        sections: dict[str, dict[str, Any]] = {}
        for name in ("dataset", "caption", "ocr", "objects", "context"):
            value = raw.get(name)
            if not isinstance(value, dict):
                raise ValueError(f"Enrichment YAML requires a {name} mapping")
            sections[name] = dict(value)
        transcript = raw.get("transcript", {})
        if not isinstance(transcript, dict):
            raise ValueError("Enrichment transcript section must be a mapping")
        sections["transcript"] = dict(transcript)

        dataset = sections["dataset"]
        required_dataset = {
            "version",
            "source",
            "btc_root",
            "data_root",
            "frame_store_id",
            "frames_path",
            "frame_store_output",
            "objects_root",
        }
        missing = sorted(required_dataset - set(dataset))
        if missing:
            raise ValueError(
                "Missing dataset enrichment configuration: " + ", ".join(missing)
            )
        if dataset["source"] != "btc_keyframes":
            raise ValueError("Enrichment V1 dataset source must be 'btc_keyframes'")

        caption_values = sections["caption"]
        caption_output = _required_output(caption_values, "caption")
        caption_values["dataset_version"] = dataset["version"]

        ocr_values = sections["ocr"]
        ocr_output = _required_output(ocr_values, "ocr")
        ocr_values["dataset_version"] = dataset["version"]

        object_values = sections["objects"]
        object_output = _required_output(object_values, "objects")
        object_config = ObjectConfig(
            objects_root=_project_path(dataset["objects_root"]),
            output_dir=object_output,
            **object_values,
        )

        transcript_output = Path(
            sections["transcript"].get(
                "output_dir", "artifacts/enrichment/transcripts"
            )
        )
        context_values = sections["context"]
        context_output = _required_output(context_values, "context")

        return cls(
            dataset_version=str(dataset["version"]),
            source=str(dataset["source"]),
            btc_root=_project_path(dataset["btc_root"]),
            data_root=_project_path(dataset["data_root"]),
            frame_store_id=str(dataset["frame_store_id"]),
            frames_path=_project_path(dataset["frames_path"]),
            frame_store_output=_project_path(dataset["frame_store_output"]),
            objects_root=_project_path(dataset["objects_root"]),
            caption_output_dir=caption_output,
            caption=CaptionConfig.from_dict(caption_values),
            ocr_output_dir=ocr_output,
            ocr=OCRConfig(**ocr_values),
            object_output_dir=object_output,
            objects=object_config,
            transcript_output_dir=_project_path(transcript_output),
            context_output_dir=_project_path(context_output),
            context=FrameContextConfig(**context_values),
        )


def _required_output(values: dict[str, Any], section: str) -> Path:
    """Remove and return one required stage output path."""

    output = values.pop("output_dir", None)
    if output is None:
        raise ValueError(f"Missing enrichment configuration: {section}.output_dir")
    return _project_path(output)


def _project_path(value: str | Path) -> Path:
    """Resolve configured paths from the repository, never the caller's CWD."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


class EnrichmentService:
    """Run independent enrichment stages through explicit boundaries."""

    @staticmethod
    def generate_captions(
        frames_path: str | Path,
        output_dir: str | Path,
        config: CaptionConfig,
        adapter: CaptionAdapter | None = None,
        *,
        dataset_root: str | Path = ".",
        frame_store_id: str | None = None,
    ) -> dict[str, Any]:
        return generate_captions(
            frames_path,
            output_dir,
            config,
            adapter,
            dataset_root=dataset_root,
            frame_store_id=frame_store_id,
        )

    @staticmethod
    def generate_ocr(
        frames_path: str | Path,
        output_dir: str | Path,
        config: OCRConfig,
        adapter: OCRAdapter | None = None,
        *,
        dataset_root: str | Path = ".",
        frame_store_id: str | None = None,
    ) -> dict[str, Any]:
        return generate_ocr(
            frames_path,
            output_dir,
            config,
            adapter,
            dataset_root=dataset_root,
            frame_store_id=frame_store_id,
        )

    @staticmethod
    def import_objects(
        frames_path: str | Path,
        objects_root: str | Path,
        output_dir: str | Path,
        config: ObjectConfig,
        *,
        frame_store_id: str | None = None,
    ) -> dict[str, Any]:
        """Import BTC-provided object JSON without running detection."""

        return import_objects(
            frames_path,
            objects_root,
            output_dir,
            config,
            frame_store_id=frame_store_id,
        )

    @staticmethod
    def build_frame_context(
        frames_path: str | Path,
        caption_path: str | Path,
        ocr_frames_path: str | Path,
        object_frames_path: str | Path,
        output_dir: str | Path,
        config: FrameContextConfig,
        *,
        frame_store_id: str | None = None,
    ) -> Path:
        """Build deterministic context from existing specialist artifacts."""

        return build_frame_context(
            frames_path,
            caption_path,
            ocr_frames_path,
            object_frames_path,
            output_dir,
            config,
            frame_store_id=frame_store_id,
        )

    @staticmethod
    def build_frame_context_from_job(job: EnrichmentJobConfig) -> Path:
        """Build context from existing job artifacts without generating sources."""

        return EnrichmentService.build_frame_context(
            job.frames_path,
            job.caption_output_dir / "captions.parquet",
            job.ocr_output_dir / "frames.parquet",
            job.object_output_dir / "frames.parquet",
            job.context_output_dir,
            job.context,
            frame_store_id=job.frame_store_id,
        )

    @staticmethod
    def run_caption_cli() -> int:
        """Run the legacy caption CLI entry point."""

        from hcmai.data.enrichment.caption.generator import main

        return main()

    @staticmethod
    def create_caption_adapter(config: CaptionConfig) -> CaptionAdapter:
        """Create the configured local caption adapter."""

        return QwenVLCaptionAdapter(config)

    @staticmethod
    def create_ocr_adapter(config: OCRConfig) -> OCRAdapter:
        """Create the configured local OCR adapter."""

        if config.backend == "remote":
            raise NotImplementedError("Remote OCR adapter is not implemented.")
        from hcmai.data.enrichment.ocr.adapters.florence import FlorenceAdapter
        return FlorenceAdapter(config)
