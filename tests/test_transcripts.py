import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import hcmai.transcripts.asr as asr_module
from hcmai.common.config import ASRConfig
from hcmai.common.schemas import TranscriptSegment
from hcmai.transcripts import (
    ASREngine, TranscriptStore, prepare_transcripts,
)


def test_qwen_segments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Inputs(dict):
        def to(self, *_args):
            return self
    class Processor:
        outputs = iter((
            {"transcription": " ", "language": None},
            {"transcription": " Cafe\u0301 Việt ", "language": "Vietnamese"},
            {"transcription": " hello ", "language": "English"},
        ))
        def apply_transcription_request(self, **options):
            self.options = options
            self.batches.append(len(options["audio"]))
            return Inputs(input_ids=np.zeros((len(options["audio"]), 2), dtype=int))
        def decode(self, ids, **_options):
            return [next(self.outputs) for _ in range(len(ids))]
    class Model:
        device = dtype = "cpu"
        def generate(self, **options):
            self.options = options
            return np.zeros((len(options["input_ids"]), 3), dtype=int)
    processor, model = Processor(), Model()
    processor.batches = []
    config = ASRConfig(device="cpu", batch_size=2, prompt="HCMAI")
    engine = ASREngine(config, model, processor, object())
    monkeypatch.setattr(asr_module, "read_audio", lambda *_: np.zeros(32_000))
    monkeypatch.setattr(engine, "_speech_regions", lambda _audio: [
        {"start": 0, "end": 800}, {"start": 1_600, "end": 9_600},
        {"start": 16_000, "end": 24_000},
    ])
    fake_torch = type("Torch", (), {"inference_mode": staticmethod(nullcontext)})
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    records = engine.transcribe(tmp_path / "video.mp4", "L21_V001")
    assert [row.language for row in records] == ["vietnamese", "english"]
    assert records[0].text == "Café Việt" and model.options["do_sample"] is False
    assert (records[0].start_ms, records[0].end_ms) == (100, 600)
    assert processor.batches == [2, 1]


class FakeASR:
    def __init__(self):
        self.calls = []
    def transcribe(self, _path, video_id):
        self.calls.append(video_id)
        return [] if video_id == "L21_V002" else [TranscriptSegment(
            segment_id=f"{video_id}_segment_000000", video_id=video_id,
            segment_index=0, start_ms=0, end_ms=800, text="text",
            language="vi")]


class FakeDiarizer:
    def __init__(self):
        self.calls = []
    def assign_speakers(self, path, records):
        video_id = path.stem
        self.calls.append(video_id)
        if video_id == "L21_V003" and self.calls.count(video_id) == 1:
            raise RuntimeError("diarization failed")
        return [
            record.model_copy(update={"speaker_id": "SPEAKER_00"})
            for record in records
        ]


def test_prepare_resume_and_stores(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    for video_id in ("L21_V003", "L21_V001", "L21_V002"):
        (videos / f"{video_id}.mp4").touch()
    transcripts = tmp_path / "artifacts/transcripts"
    asr, diarizer = FakeASR(), FakeDiarizer()
    first = prepare_transcripts(
        videos, transcripts, asr, diarizer=diarizer)
    assert (first.transcribed, first.no_speech) == (1, 1)
    assert not (transcripts / "L21/L21_V003.parquet").exists()
    partial = transcripts / "L21/L21_V003.parquet.partial"
    partial.touch()
    second = prepare_transcripts(
        videos, transcripts, asr, diarizer=diarizer)
    assert not second.failed and second.segments == 2
    assert asr.calls.count("L21_V003") == 2 and not partial.exists()
    output = transcripts / "L21/L21_V001.parquet"
    assert list(pd.read_parquet(output)) == list(TranscriptSegment.model_fields)
    assert TranscriptStore(transcripts).get_by_video("L21_V001")[0].speaker_id == "SPEAKER_00"
