from types import SimpleNamespace
import inspect
from typing import Literal, cast
import pytest
import torch
from PIL import Image
from hcmai.common.schemas import RetrievalCandidate
from hcmai.data.pipeline import DataService
from hcmai.retrieval.reranking.adapters import qwen
from hcmai.retrieval.reranking.adapters.qwen import QwenAdapter, QwenRerankerError
from hcmai.retrieval.reranking.config import QwenRerankerConfig, RerankerConfig
from hcmai.retrieval.reranking.pipeline import RerankingService
class Processor:
    def __init__(
        self, malformed: bool | Literal["empty", "short"] = False
    ) -> None:
        self.tokenizer, self.calls, self.malformed = SimpleNamespace(
            get_vocab=lambda: {"no": 0, "yes": 1}), [], malformed
    def apply_chat_template(self, pairs, **_):
        self.calls.append(pairs)
        return ["prompt"] * len(pairs)
    def __call__(self, *, images, **_):
        if self.malformed == "empty":
            return {}
        values = [image.info.get("logits", image.getpixel((0, 0))[:2]) for image in images]
        if self.malformed == "short":
            values = values[:-1]
        features = torch.tensor(values).reshape(-1, 2)
        return {"input_ids": torch.ones(len(values), 1), "features": features}
class Model:
    def __init__(self, missing=False):
        self.lm_head = SimpleNamespace(weight=torch.eye(2))
        self.model, self.config, self.moves = self, SimpleNamespace(_commit_hash="rev"), []
        self.missing = missing
    def to(self, device):
        self.moves.append(device)
        return self
    def eval(self):
        return self
    def __call__(self, features, **_):
        if self.missing:
            return SimpleNamespace()
        return SimpleNamespace(last_hidden_state=features[:, None, :])
def image(no, yes):
    item = Image.new("RGB", (2, 2))
    item.info["logits"] = (no, yes)
    return item
def vision_info(pairs, **_):
    images = [pair[1]["content"][-1]["image"] for pair in pairs]
    return images, [], {}
def scorer(model=None, processor=None, dtype="bfloat16"):
    return QwenAdapter(
        QwenRerankerConfig(revision="pinned", dtype=dtype),
        model or Model(), processor or Processor(), vision_info)
def test_lazy_injected_lifecycle_order_types_and_native_policy():
    model, processor = Model(), Processor()
    value = scorer(model, processor)
    assert value.model is model and value._base_model is None
    scores = value.score_batch("query", [image(0, 1), image(1, 0)])
    assert scores[0] > scores[1] and all(type(x) is float for x in scores)
    assert all(torch.isfinite(torch.tensor(scores))) and len(processor.calls) == 1
    assert len(value.score_batch("query", [image(0, 0)])) == 1
    assert value.score_batch("query", []) == [] and model.moves == ["cpu"]
    assert "query" in str(processor.calls) and "<Document>:" in str(processor.calls)
    source = inspect.getsource(qwen)
    assert "trust_remote_code=True" not in source and "FrameStore" not in source
    assert "DenseRetriever" not in source and "faiss" not in source.lower()
@pytest.mark.parametrize(
    "no,yes,expected", [(0, 0, .5), (-1000, 1000, 1), (1000, -1000, 0)])
def test_official_probability_is_stable(no, yes, expected):
    assert scorer().score_batch("q", [image(no, yes)])[0] == pytest.approx(expected)
def test_probability_monotonicity_and_dtype_mapping():
    values = scorer().score_batch("q", [image(0, 1), image(0, 2), image(1, 0)])
    assert values[1] > values[0] > values[2]
    assert qwen._torch_dtype("bfloat16") is torch.bfloat16
    assert qwen._torch_dtype("float32") is torch.float32
@pytest.mark.parametrize("model,processor", [
    (Model(missing=True), Processor()), (Model(), Processor("empty")),
    (Model(), Processor("short"))])
def test_bounded_malformed_outputs(model, processor):
    with pytest.raises(QwenRerankerError, match="Qwen|processor|model"):
        scorer(model, processor).score_batch("q", [image(0, 0)])
@pytest.mark.parametrize("error", [RuntimeError("processor"), RuntimeError("model")])
def test_initialization_failure_is_cached(monkeypatch, error):
    calls = []
    def fail(_):
        calls.append(1)
        raise error
    monkeypatch.setattr(qwen, "_load_native", fail)
    value = QwenAdapter(QwenRerankerConfig())
    for _ in range(2):
        with pytest.raises(QwenRerankerError, match="initialization failed"):
            value.score_batch("q", [image(0, 0)])
    assert calls == [1]
def test_composition_preserves_candidates(tmp_path):
    records, candidates = {}, []
    for index in range(3):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (2, 2), (index, 3 - index, 0)).save(path)
        records[str(index)] = SimpleNamespace(image_path=path)
        candidates.append(RetrievalCandidate(frame_id=str(index), metadata={"i": index}))
    store = SimpleNamespace(get_frame=lambda frame_id: records[frame_id])
    output = RerankingService(
        cast(DataService, store), RerankerConfig(batch_size=2), scorer()
    ).rerank("q", candidates)
    assert len(output) == 3 and {x.frame_id for x in output} == {"0", "1", "2"}
    assert all(x.reranker_score == x.final_score for x in output)
