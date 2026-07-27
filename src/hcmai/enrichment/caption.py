"""Batch and resume frame caption enrichment."""
from __future__ import annotations
import argparse
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence
import pandas as pd
from hcmai.common.schemas import FrameEnrichment, ProcessingStatus
from hcmai.common.utils.image import load_image
from hcmai.common.utils.io import read_json, read_yaml, write_json, write_parquet

@dataclass
class CaptionConfig:
    """Settings identifying one reproducible caption enrichment."""
    model_checkpoint: str = "microsoft/Florence-2-base-ft"; revision: str | None = None
    prompt: str = "<CAPTION>"
    decoding: dict[str, Any] = field(default_factory=lambda: {"max_new_tokens": 64, "num_beams": 3, "do_sample": False})
    device: str = "cpu"; precision: str = "fp32"; dtype: str = "float32"
    image_size: int = 768; batch_size: int = 4
    enrichment_version: str = "caption_v1"; write_interval: int = 25; dataset_version: str = "unknown"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaptionConfig:
        """Load known settings and accept the conventional ``name`` key."""
        values = dict(data); values["model_checkpoint"] = values.pop("name", values.get("model_checkpoint", cls.model_checkpoint))
        if "precision" in values and "dtype" not in values: values["dtype"] = {"fp16": "float16", "bf16": "bfloat16"}.get(values["precision"], "float32")
        config = cls(**{key: value for key, value in values.items() if key in cls.__dataclass_fields__})
        if min(config.batch_size, config.image_size, config.write_interval) < 1: raise ValueError("batch_size, image_size, and write_interval must be positive")
        return config
class FrameCaptioner:
    """Lazy, single-instance caption model boundary."""
    def __init__(self, config: CaptionConfig, model: Any = None, processor: Any = None,
                 batch_fn: Callable[[Sequence[Any]], Sequence[Any]] | None = None):
        self.config, self.model, self.processor = config, model, processor
        self.batch_fn, self.resolved_revision, self._dtype = batch_fn, None, None
    def _load(self) -> None:
        if self.model is None or self.processor is None:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            revision = {"revision": self.config.revision} if self.config.revision else {}
            types = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}; self._dtype = types.get(self.config.dtype, torch.float32)
            self.processor = self.processor or AutoProcessor.from_pretrained(self.config.model_checkpoint, **revision)
            self.model = self.model or AutoModelForImageTextToText.from_pretrained(self.config.model_checkpoint, torch_dtype=self._dtype, **revision).to(self.config.device)
            self.model.eval()
        self.resolved_revision = getattr(getattr(self.model, "config", None), "_commit_hash", None) or self.config.revision
    def resolve_revision(self) -> str:
        """Resolve the immutable model revision before writing reusable rows."""
        if self.resolved_revision:
            return self.resolved_revision
        if self.batch_fn is not None:
            self.resolved_revision = self.config.revision
        else:
            self._load()
        if not self.resolved_revision:
            raise ValueError("Cannot create resumable captions without a resolved model revision")
        return self.resolved_revision
    def caption_batch(self, images: Sequence[Any]) -> list[Any]:
        """Return captions or per-image exceptions for one batch."""
        if self.batch_fn is not None: return list(self.batch_fn(images))
        try:
            self._load()
        except Exception as error:
            self.batch_fn = lambda items, failure=error: [failure] * len(items)
            raise
        inputs = self.processor(text=[self.config.prompt] * len(images), images=list(images), return_tensors="pt", padding=True)
        for key, value in inputs.items():
            value = value.to(self.config.device)
            inputs[key] = value.to(self._dtype) if self._dtype is not None and value.is_floating_point() else value
        generated = self.model.generate(**inputs, **self.config.decoding); decoded = self.processor.batch_decode(generated, skip_special_tokens=False)
        return [self.processor.post_process_generation(text, task=self.config.prompt, image_size=image.size).get(self.config.prompt, "") for text, image in zip(decoded, images)]
def _valid(data: dict[str, Any], version: str) -> FrameEnrichment | None:
    try:
        values, objects = dict(data), data.get("objects")
        to_list = getattr(objects, "tolist", None)
        values["objects"] = to_list() if callable(to_list) else objects or []
        nulls = ("caption", "detailed_caption", "ocr_text", "asr_text", "enrichment_version", "error_message")
        values.update({key: None for key in nulls if pd.isna(values.get(key))})
        row = FrameEnrichment.model_validate(values)
    except Exception:
        return None
    ok = row.enrichment_version == version and row.status == ProcessingStatus.COMPLETED
    ok = ok and bool(row.caption and row.caption.strip()) and row.error_message is None
    return row if ok else None
def _atomic(path: Path, writer: Callable[[Path], None]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        writer(temporary); temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
def _resume_guard(path: Path, old: dict[str, Any], config: CaptionConfig, root: Path,
                  resolved_revision: str | None = None) -> None:
    if not path.exists(): return
    if not old: raise ValueError("Cannot safely resume: manifest.json is missing")
    if old.get("enrichment_version") != config.enrichment_version: return
    previous, current = old.get("effective_configuration"), asdict(config)
    if not isinstance(previous, dict): raise ValueError("Cannot safely resume: effective configuration is missing")
    changed = [key for key, value in current.items() if previous.get(key) != value]
    if old.get("dataset_root") != str(root): changed.append("dataset_root")
    if resolved_revision is not None and old.get("resolved_model_revision") != resolved_revision:
        changed.append("resolved_model_revision")
    if changed:
        raise ValueError(f"Cannot resume {config.enrichment_version!r}: changed {', '.join(sorted(set(changed)))}; use a new enrichment_version or output directory")
def _resume(frames: list[dict[str, Any]], path: Path, config: CaptionConfig):
    groups: dict[str, list[dict[str, Any]]] = {}
    if path.exists():
        try:
            prior = pd.read_parquet(path).to_dict(orient="records")
        except Exception as error:
            message = str(error).strip()[:200] or type(error).__name__; raise RuntimeError(f"Cannot resume corrupted Parquet {path}: {message}") from error
        for data in prior:
            if data.get("enrichment_version") == config.enrichment_version: groups.setdefault(str(data.get("frame_id")), []).append(data)
    rows, todo, skipped, retried = {}, [], 0, 0
    for frame in frames:
        frame_id, old = frame["frame_id"], groups.get(frame["frame_id"], [])
        row = _valid(old[0], config.enrichment_version) if len(old) == 1 else None
        if row:
            rows[frame_id], skipped = row, skipped + 1; continue
        retried += bool(old)
        rows[frame_id] = FrameEnrichment(frame_id=frame_id, model_name=config.model_checkpoint, enrichment_version=config.enrichment_version, status=ProcessingStatus.PENDING); todo.append(frame)
    return rows, todo, skipped, retried
def _failure(frame_id: str, config: CaptionConfig, stage: str, error: Exception):
    message = str(error).strip()[:300] or type(error).__name__
    row = FrameEnrichment(frame_id=frame_id, model_name=config.model_checkpoint, enrichment_version=config.enrichment_version, status=ProcessingStatus.FAILED, error_message=message)
    detail = {"frame_id": frame_id, "enrichment_version": config.enrichment_version,
              "processing_stage": stage, "exception_category": type(error).__name__, "error_message": message}
    return row, detail
def _write(output: Path, order: list[str], rows: dict[str, FrameEnrichment], failures: dict[str, dict[str, str]]) -> None:
    data = pd.DataFrame([rows[key].model_dump(mode="json") for key in order])
    _atomic(output / "frame_enrichment.parquet", lambda path: write_parquet(data, path, index=False))
    _atomic(output / "failures.json", lambda path: write_json([failures[key] for key in order if key in failures], path))
def _run(todo: list[dict[str, Any]], order: list[str], rows: dict[str, Any], failures: dict[str, Any],
         captioner: FrameCaptioner, config: CaptionConfig, output: Path, root: Path) -> list[float]:
    latencies, since_write = [], 0
    for start in range(0, len(todo), config.batch_size):
        chunk, valid = todo[start:start + config.batch_size], []
        for frame in chunk:
            try:
                path = Path(str(frame["image_path"])).expanduser(); image = load_image(path if path.is_absolute() else root / path, mode="RGB")
                image.thumbnail((config.image_size, config.image_size)); valid.append((frame["frame_id"], image))
            except Exception as error:
                rows[frame["frame_id"]], failures[frame["frame_id"]] = _failure(frame["frame_id"], config, "image_load", error)
        if valid:
            began = perf_counter()
            try:
                results = captioner.caption_batch([image for _, image in valid])
                if len(results) != len(valid): raise ValueError("caption backend returned the wrong result count")
            except Exception as error:
                results = [error] * len(valid)
            latencies.append((perf_counter() - began) * 1000)
            for (frame_id, _), result in zip(valid, results):
                if isinstance(result, Exception) or result is None or not str(result).strip():
                    error = result if isinstance(result, Exception) else ValueError("empty caption"); rows[frame_id], failures[frame_id] = _failure(frame_id, config, "model", error)
                else:
                    rows[frame_id] = FrameEnrichment(frame_id=frame_id, caption=str(result).strip(), model_name=config.model_checkpoint, enrichment_version=config.enrichment_version)
        since_write += len(chunk)
        if since_write >= config.write_interval:
            _write(output, order, rows, failures); since_write = 0
    return latencies
def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None
def _manifest(config: CaptionConfig, frames_path: Path, root: Path, rows: dict[str, Any], captioner: FrameCaptioner,
              old: dict[str, Any], started: datetime, elapsed: float, latencies: list[float], skipped: int, retried: int):
    complete, ordered = sum(row.status == ProcessingStatus.COMPLETED for row in rows.values()), sorted(latencies)
    pick = lambda part: ordered[round((len(ordered) - 1) * part)] if ordered else 0.0
    return {"artifact_version": "frame_enrichment.v1", "enrichment_version": config.enrichment_version,
            "dataset_version": config.dataset_version, "input_parquet_path": str(frames_path), "dataset_root": str(root),
            "model_checkpoint": config.model_checkpoint, "resolved_model_revision": captioner.resolved_revision,
            "prompt": config.prompt, "decoding": config.decoding, "device": config.device,
            "precision": config.precision, "dtype": config.dtype, "image_size": config.image_size,
            "batch_size": config.batch_size, "input_frame_count": len(rows), "completed_count": complete,
            "failed_count": len(rows) - complete, "skipped_count": skipped, "retried_count": retried,
            "start_time": started.isoformat(), "end_time": datetime.now(timezone.utc).isoformat(),
            "elapsed_time_sec": elapsed, "throughput_images_per_sec": (complete - skipped) / elapsed if elapsed else 0.0,
            "batch_latency_ms_p50": pick(.5), "batch_latency_ms_p95": pick(.95),
            "effective_configuration": asdict(config), "git_commit": _git_commit()}
def generate_captions(frames_path: str | Path, output_dir: str | Path, config: CaptionConfig,
                      captioner: FrameCaptioner | None = None, *, dataset_root: str | Path = ".") -> dict[str, Any]:
    """Generate or resume one deterministic caption enrichment artifact."""
    started, began, frames_path = datetime.now(timezone.utc), perf_counter(), Path(frames_path)
    root = Path(dataset_root).expanduser().resolve()
    frames = pd.read_parquet(frames_path).to_dict(orient="records"); order = [frame["frame_id"] for frame in frames]
    if len(order) != len(set(order)): raise ValueError("input frames contain duplicate frame_id values")
    output, captioner = Path(output_dir), captioner or FrameCaptioner(config); output.mkdir(parents=True, exist_ok=True)
    manifest_path, parquet_path = output / "manifest.json", output / "frame_enrichment.parquet"
    old = read_json(manifest_path) if manifest_path.exists() else {}
    _resume_guard(parquet_path, old, config, root)
    rows, todo, skipped, retried = _resume(frames, parquet_path, config)
    resolved_revision = captioner.resolve_revision()
    _resume_guard(parquet_path, old, config, root, resolved_revision)
    provisional = {**old, "enrichment_version": config.enrichment_version,
                   "effective_configuration": asdict(config), "dataset_root": str(root),
                   "resolved_model_revision": resolved_revision}
    _atomic(manifest_path, lambda path: write_json(provisional, path))
    failures: dict[str, dict[str, str]] = {}
    latencies = _run(todo, order, rows, failures, captioner, config, output, root)
    _write(output, order, rows, failures)
    manifest = _manifest(config, frames_path, root, rows, captioner, old, started, perf_counter() - began, latencies, skipped, retried)
    _atomic(manifest_path, lambda path: write_json(manifest, path))
    return manifest
def main() -> int:
    """Run caption enrichment from YAML."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--frames", required=True)
    parser.add_argument("--dataset-root"); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raw = read_yaml(args.config) or {}; dataset = raw.get("dataset", {})
    values = raw.get("caption", raw.get("models", {}).get("caption", {}))
    values = {**values, "dataset_version": dataset.get("version", values.get("dataset_version", "unknown"))}
    manifest = generate_captions(args.frames, args.output, CaptionConfig.from_dict(values),
                                 dataset_root=args.dataset_root or dataset.get("root", "."))
    keys = "completed_count", "failed_count", "skipped_count", "retried_count"
    print({key: manifest[key] for key in keys})
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
