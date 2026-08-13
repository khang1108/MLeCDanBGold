import sys
from types import SimpleNamespace

import numpy as np
import pytest

import hcmai.data.enrichment.transcripts.adapters.diarization as diarization_module
from hcmai.common.config import DiarizationConfig
from hcmai.common.schemas import TranscriptSegment
from hcmai.data.enrichment.transcripts.adapters.diarization import (
    DiarizationAdapter,
)
from hcmai.data.enrichment.transcripts.adapters.asr import DecodedAudio


def test_diarization_lazy_load_and_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Pipeline:
        loads = 0
        @classmethod
        def from_pretrained(cls, name, **options):
            cls.loads += 1
            cls.load = (name, options)
            return cls()
        def to(self, device):
            self.device = device
        def __call__(self, _audio):
            turn = lambda start, end: SimpleNamespace(start=start, end=end)
            return SimpleNamespace(exclusive_speaker_diarization=[
                (turn(1.5, 3.0), "SPEAKER_01"),
                (turn(0.5, 1.5), "SPEAKER_00"),
            ])
    tensor = SimpleNamespace(unsqueeze=lambda _axis: "waveform")
    torch = SimpleNamespace(
        from_numpy=lambda _audio: tensor, device=lambda value: value)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "pyannote", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "pyannote.audio", SimpleNamespace(Pipeline=Pipeline))
    monkeypatch.setattr(
        diarization_module,
        "read_audio",
        lambda *_: DecodedAudio(np.zeros(16_000), 16_000, 0),
    )
    config = DiarizationConfig(device="cpu")
    engine = DiarizationAdapter(config, hf_token="secret")
    segments = [
        TranscriptSegment(
            segment_id=f"L25_V001_segment_{index:06d}",
            video_id="L25_V001", segment_index=index,
            start_ms=start, end_ms=end, text="text", language="vi")
        for index, (start, end) in enumerate(((0, 2_000), (1_800, 2_800)))
    ]
    records = engine.assign_speakers("video.mp4", segments)
    engine.assign_speakers("video.mp4", segments)
    assert Pipeline.loads == 1 and Pipeline.load[1]["token"] == "secret"
    assert Pipeline.load[1]["revision"] == config.revision
    assert [record.speaker_id for record in records] == [
        "SPEAKER_00", "SPEAKER_01"]
    monkeypatch.setattr(
        diarization_module,
        "read_audio",
        lambda *_: DecodedAudio(np.empty(0), 16_000, 0),
    )
    assert engine.assign_speakers("silent.mp4", segments) == segments
