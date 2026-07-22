from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import pytest
from PIL import Image
from hcmai.common.schemas import RetrievalCandidate
from hcmai.reranker import MultimodalReranker, RerankerConfig

class Store:
    def __init__(self, records):
        self.records, self.calls = records, []
    def get(self, frame_id):
        self.calls.append(frame_id)
        if frame_id not in self.records:
            raise KeyError(frame_id)
        return self.records[frame_id]
    def filter_frame_ids(self, _):
        raise AssertionError("corpus enumeration is forbidden")

class Backend:
    instances = 0
    def __init__(self, result=None, error=None):
        type(self).instances += 1
        self.result, self.error, self.calls = result, error, []
    def __call__(self, query, images):
        self.calls.append((query, len(images)))
        if self.error:
            raise self.error
        return self.result(images)

def make_store(root: Path, count: int) -> Store:
    records = {}
    for index in range(count):
        path = root / f"f{index}.png"
        Image.new("RGB", (4, 4), (index, 0, 0)).save(path)
        records[f"f{index}"] = SimpleNamespace(image_path=path)
    return Store(records)

def candidate(index: int) -> RetrievalCandidate:
    return RetrievalCandidate(
        frame_id=f"f{index}", source_scores={"visual": index / 100},
        final_score=index / 100, metadata={"original": index},
    )

def test_success_identity_batches_scores_order_and_lifecycle(tmp_path):
    Backend.instances = 0
    store, inputs = make_store(tmp_path, 20), [candidate(i) for i in range(20)]
    before = [item.model_dump() for item in inputs]
    backend = Backend(lambda images: [image.getpixel((0, 0))[0] for image in images])
    reranker = MultimodalReranker(store, RerankerConfig(batch_size=6), backend)
    output = reranker.rerank("query", inputs)
    assert len(output) == 20 and {item.frame_id for item in output} == {f"f{i}" for i in range(20)}
    assert [len_ for _, len_ in backend.calls] == [6, 6, 6, 2]
    assert store.calls == [item.frame_id for item in inputs]
    assert [item.frame_id for item in output] == [f"f{i}" for i in reversed(range(20))]
    assert {item.frame_id: item.reranker_score for item in output} == {f"f{i}": float(i) for i in range(20)}
    assert all(item.final_score == item.reranker_score for item in output)
    assert output[-1].frame_id == "f0" and "reranker_fallback" not in output[-1].metadata
    assert before == [item.model_dump() for item in inputs] and Backend.instances == 1
    assert reranker.rerank("query", []) == [] and len(reranker.rerank("query", [inputs[0]])) == 1
    tied = MultimodalReranker(store, RerankerConfig(batch_size=20), Backend(lambda images: [1] * len(images)))
    assert [item.frame_id for item in tied.rerank("q", inputs)] == [item.frame_id for item in inputs]
    assert all(RetrievalCandidate.model_validate(item.model_dump()) for item in output)

def test_candidate_failures_keep_alignment_and_black_image(tmp_path):
    store, inputs = make_store(tmp_path, 8), [candidate(i) for i in range(8)]
    store.records["f1"].image_path = tmp_path / "missing.png"
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"broken")
    store.records["f2"].image_path, _ = corrupt, store.records.pop("f3")
    backend = Backend(lambda _: [0.4, float("nan"), float("inf"), float("-inf"), "bad"])
    output = MultimodalReranker(store, RerankerConfig(batch_size=10), backend).rerank("q", inputs)
    by_id = {item.frame_id: item for item in output}
    assert len(output) == 8 and backend.calls == [("q", 5)]
    assert by_id["f0"].reranker_score == 0.4 and by_id["f0"].final_score == 0.4
    assert all(by_id[f"f{i}"].reranker_score is None for i in range(1, 8))
    assert all(by_id[f"f{i}"].final_score == i / 100 for i in range(1, 8))
    assert all("reranker_fallback" in by_id[f"f{i}"].metadata for i in range(1, 8))
    assert store.calls == [item.frame_id for item in inputs]

@pytest.mark.parametrize("error", [TimeoutError("late"), RuntimeError("CUDA out of memory"), Exception("model")])
def test_backend_failure_is_request_fallback_and_cached(tmp_path, error):
    store, inputs, backend = make_store(tmp_path, 3), [candidate(i) for i in range(3)], Backend(error=error)
    reranker = MultimodalReranker(store, RerankerConfig(batch_size=3), backend)
    first, second = reranker.rerank("q", inputs), reranker.rerank("q", inputs)
    assert [item.frame_id for item in first] == [item.frame_id for item in inputs]
    assert [item.frame_id for item in second] == [item.frame_id for item in inputs]
    assert backend.calls == [("q", 3)]
    assert [item.final_score for item in first] == [item.final_score for item in inputs]
    assert all("reranker_fallback" in item.metadata for item in first)

def test_count_mismatch_and_duplicate_ids_preserve_request(tmp_path):
    store, inputs = make_store(tmp_path, 2), [candidate(0), candidate(0), candidate(1)]
    results, backend = iter([[0.5, 0.4], []]), Backend(lambda _: next(results))
    output = MultimodalReranker(store, RerankerConfig(batch_size=2), backend).rerank("q", inputs)
    assert [item.frame_id for item in output] == ["f0", "f0", "f1"]
    assert len(output) == len(inputs) and backend.calls == [("q", 2), ("q", 1)]
    assert all(item.reranker_score is None for item in output)
