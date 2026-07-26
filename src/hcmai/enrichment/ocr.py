"""Independent resumable OCR enrichment."""
from __future__ import annotations
import math, unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import pandas as pd
from PIL import Image
from hcmai.common.schemas import FrameEnrichment, ProcessingStatus
from hcmai.common.utils.image import load_image
from hcmai.common.utils.io import read_json, write_json, write_parquet
@dataclass
class OCRConfig:
    """Settings identifying one reproducible OCR enrichment."""
    enabled: bool = True
    backend: str = "florence2"
    checkpoint: str | None = "florence-community/Florence-2-base-ft"
    revision: str | None = None
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 1
    image_size: int | None = 768
    enrichment_version: str = "florence2_ocr_v1"
    dataset_version: str = "unknown"
    @property
    def model_name(self) -> str:
        """Return the canonical backend identity."""
        return self.checkpoint or self.backend
@dataclass
class OCRResult:
    """One ordered backend OCR response."""
    text: str
    raw_output: object | None = None
    confidence: float | None = None
class OCREngine:
    """Lazy native Florence-2 OCR backend."""
    def __init__(self, config: OCRConfig) -> None:
        self.config, self.model, self.processor, self.resolved_revision, self._failure = config, None, None, config.revision, None
    def _load(self) -> None:
        if self._failure is not None:
            raise RuntimeError("OCR backend initialization failed") from self._failure
        if self.model is not None and self.processor is not None: return
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            dtype = {"bfloat16": torch.bfloat16}.get(self.config.dtype, torch.float32)
            options = {"revision": self.config.revision, "trust_remote_code": False}
            self.processor = AutoProcessor.from_pretrained(self.config.model_name, **options)
            self.model = AutoModelForImageTextToText.from_pretrained(self.config.model_name, dtype=dtype, **options)
            self.model.to(self.config.device).eval()
            self.resolved_revision = getattr(self.model.config, "_commit_hash", None) or self.config.revision
        except Exception as error:
            self._failure = error
            raise
    def recognize_batch(self, images: Sequence[Image.Image]) -> list[OCRResult]:
        """Return one OCR result per image in input order."""
        self._load()
        import torch
        inputs = self.processor(text=["<OCR>"] * len(images), images=list(images), return_tensors="pt", padding=True)
        dtype = {"bfloat16": torch.bfloat16}.get(self.config.dtype, torch.float32)
        inputs = {key: value.to(self.config.device, dtype=dtype)
                  if value.is_floating_point() else value.to(self.config.device)
                  for key, value in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=256, num_beams=3, do_sample=False)
        decoded = self.processor.batch_decode(generated, skip_special_tokens=False)
        results = []
        for text, image in zip(decoded, images):
            raw = self.processor.post_process_generation(text, task="<OCR>", image_size=image.size).get("<OCR>", "")
            value = "" if str(raw).strip().casefold() == "unanswerable" else raw
            results.append(OCRResult(text=value, raw_output=raw))
        return results
def normalize_text(text: str) -> str:
    """Normalize text without discarding Vietnamese characters."""
    return " ".join(unicodedata.normalize("NFC", text).split())
def _valid(data: dict[str, object], config: OCRConfig) -> FrameEnrichment | None:
    try:
        values, objects = dict(data), data.get("objects")
        values["objects"] = objects.tolist() if hasattr(objects, "tolist") else objects or []
        for name in ("caption", "detailed_caption", "ocr_text", "asr_text", "error_message"):
            if pd.isna(values.get(name)): values[name] = None
        row = FrameEnrichment.model_validate(values)
    except Exception:
        return None
    valid = (row.enrichment_version == config.enrichment_version and row.model_name == config.model_name
             and row.status == ProcessingStatus.COMPLETED and row.error_message is None
             and row.caption is None and row.detailed_caption is None and row.asr_text is None)
    return row if valid else None
def _resume(frames, path: Path, config: OCRConfig):
    groups: dict[str, list[dict[str, object]]] = {}
    if path.exists():
        for row in pd.read_parquet(path).to_dict(orient="records"):
            if row.get("enrichment_version") == config.enrichment_version:
                groups.setdefault(str(row.get("frame_id")), []).append(row)
    rows, todo, skipped, retried = {}, [], 0, 0
    for frame in frames:
        old = groups.get(frame["frame_id"], [])
        row = _valid(old[0], config) if len(old) == 1 else None
        if row:
            rows[frame["frame_id"]], skipped = row, skipped + 1
        else:
            retried += bool(old)
            todo.append(frame)
    return rows, todo, skipped, retried
def _failure(frame_id, config, stage, error):
    message = " ".join(str(error).split())[:300] or type(error).__name__
    row = FrameEnrichment(frame_id=frame_id, model_name=config.model_name, enrichment_version=config.enrichment_version,
                          status=ProcessingStatus.FAILED, error_message=message)
    return row, {"frame_id": frame_id, "enrichment_version": config.enrichment_version,
        "processing_stage": stage, "exception_category": type(error).__name__, "error_message": message}
def _parsed(frame_id, result, config):
    if not isinstance(result, OCRResult) or not isinstance(result.text, str):
        raise TypeError("OCR backend returned a malformed result")
    confidence = result.confidence
    if confidence is not None and (not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence))):
        raise ValueError("OCR confidence must be finite")
    text = normalize_text(result.text)
    row = FrameEnrichment(frame_id=frame_id, ocr_text=text or None, model_name=config.model_name,
                          enrichment_version=config.enrichment_version)
    evidence = {"frame_id": frame_id, "raw_output": str(result.raw_output)[:500]
        if result.raw_output is not None else None,
        "confidence": float(confidence) if confidence is not None else None}
    return row, evidence
def _write(output, order, rows, failures):
    table = pd.DataFrame([rows[k].model_dump(mode="json") for k in order if k in rows],
                         columns=FrameEnrichment.model_fields)
    write_parquet(table, output / "frame_enrichment.parquet", index=False)
    write_json([failures[key] for key in order if key in failures], output / "failures.json")
def _process(todo, rows, failures, evidence, engine, config):
    for start in range(0, len(todo), config.batch_size):
        valid = []
        for frame in todo[start:start + config.batch_size]:
            try:
                image = load_image(frame["image_path"], mode="RGB")
                if config.image_size: image.thumbnail((config.image_size, config.image_size))
                valid.append((frame["frame_id"], image))
            except Exception as error:
                rows[frame["frame_id"]], failures[frame["frame_id"]] = _failure(frame["frame_id"], config, "image_load", error)
        if not valid: continue
        try:
            results = list(engine.recognize_batch([item[1] for item in valid]))
            if len(results) != len(valid): raise ValueError("OCR backend returned the wrong result count")
        except Exception as error:
            results = [error] * len(valid)
        for (frame_id, _), result in zip(valid, results):
            try:
                rows[frame_id], evidence[frame_id] = _parsed(frame_id, result, config)
            except Exception as error:
                rows[frame_id], failures[frame_id] = _failure(frame_id, config, "backend", error)
def _report(config, path, rows, evidence, failures, old, started, elapsed, input_count, processed, skipped, retried, revision, disabled):
    complete = sum(row.status == ProcessingStatus.COMPLETED for row in rows.values())
    text = sum(row.ocr_text is not None for row in rows.values())
    confidence = [v["confidence"] for v in evidence.values() if v.get("confidence") is not None]
    ratio = lambda count: count / input_count if input_count else 0.0
    summary = {"min": min(confidence), "max": max(confidence), "mean": sum(confidence) / len(confidence)} if confidence else None
    return {"report_version": "ocr_report.v1", "artifact_version": "frame_enrichment.v1",
        "enrichment_version": config.enrichment_version, "dataset_version": config.dataset_version,
        "input_parquet_path": str(path), "backend": config.backend, "checkpoint": config.checkpoint, "resolved_revision": revision,
        "enabled": config.enabled, "device": config.device, "dtype": config.dtype,
        "batch_size": config.batch_size, "runtime_settings": asdict(config),
        "total_frames": input_count, "processed_frames": processed, "completed_frames": complete,
        "frames_with_text": text, "empty_text_frames": complete - text, "failed_frames": len(rows) - complete,
        "skipped_frames": skipped, "retried_frames": retried, "disabled_frames": disabled,
        "text_coverage_rate": ratio(text), "empty_text_rate": ratio(complete - text), "failure_rate": ratio(len(rows) - complete),
        "error_counts": dict(Counter(v["exception_category"] for v in failures.values())),
        "confidence_available": bool(confidence), "confidence_summary": summary,
        "raw_output_available": any(v.get("raw_output") is not None for v in evidence.values()),
        "raw_evidence": [evidence[k] for k in rows if k in evidence],
        "normalization_policy": "Unicode NFC; collapse whitespace; preserve case, diacritics, numbers, punctuation.",
        "start_time": started.isoformat(), "end_time": datetime.now(timezone.utc).isoformat(), "elapsed_time_sec": elapsed,
        "manual_review": old.get("manual_review", {"sample_count": 0, "status": "pending", "summary": "Human review pending."}),
        "known_limitations": ["Coverage is not OCR accuracy.", "Florence-2 has no calibrated OCR confidence."]}
def generate_ocr(frames_path: str | Path, output_dir: str | Path, config: OCRConfig,
                 engine: object | None = None, engine_factory: Callable[[OCRConfig], object] | None = None):
    """Generate or resume one deterministic independent OCR artifact."""
    started, began, path = datetime.now(timezone.utc), perf_counter(), Path(frames_path)
    frames = pd.read_parquet(path).to_dict(orient="records")
    order = [frame["frame_id"] for frame in frames]
    if len(order) != len(set(order)): raise ValueError("input frames contain duplicate frame_id values")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "ocr_report.json"
    old = read_json(report_path) if report_path.exists() else {}
    rows, todo, skipped, retried = (_resume(frames, output / "frame_enrichment.parquet", config)
        if config.enabled else ({}, [], 0, 0))
    evidence = {v["frame_id"]: v for v in old.get("raw_evidence", []) if v.get("frame_id") in rows}
    failures = {}
    if todo:
        engine = engine or (engine_factory or OCREngine)(config)
        _process(todo, rows, failures, evidence, engine, config)
    _write(output, order, rows, failures)
    revision = getattr(engine, "resolved_revision", None) or old.get("resolved_revision") or config.revision
    report = _report(config, path, rows, evidence, failures, old, started, perf_counter() - began,
        len(frames), len(todo), skipped, retried, revision, len(frames) if not config.enabled else 0)
    write_json(report, report_path)
    write_json({k: v for k, v in report.items() if k != "raw_evidence"}, output / "manifest.json")
    return report
