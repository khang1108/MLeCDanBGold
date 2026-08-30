from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Sequence, cast
import pytest
from PIL import Image
from hcmai.common.schemas import RetrievalCandidate, RetrievalSource
from hcmai.corpus import Corpus
from hcmai.retrieval.reranking.config import RerankerConfig
from hcmai.retrieval.reranking.pipeline import (
    RerankerContractError,
    RerankerInvalidScoreError,
    RerankerTimeoutError,
    RerankerUnavailableError,
    RerankingService,
)

class Store:
    def __init__(self, records):
        self.records, self.calls = records, []
    def get(self, frame_id):
        self.calls.append(frame_id)
        if frame_id not in self.records:
            raise KeyError(frame_id)
        return self.records[frame_id]
    frame = get

    def image_path(self, frame_id):
        return self.get(frame_id).image_path
class Backend:
    instances = 0
    def __init__(self, result=None, error=None):
        type(self).instances += 1
        self.result, self.error, self.calls = result, error, []
    def score(self, query, images):
        self.calls.append((query, len(images)))
        if self.error:
            raise self.error
        result = cast(Callable[[object], Sequence[float]], self.result)
        return result(images)

def make_store(root: Path, count: int) -> Store:
    records = {}
    for index in range(count):
        path = root / f"f{index}.png"
        Image.new("RGB", (4, 4), (index, 0, 0)).save(path)
        records[f"f{index}"] = SimpleNamespace(image_path=path)
    return Store(records)

def candidate(index: int) -> RetrievalCandidate:
    return RetrievalCandidate(
        frame_id=f"f{index}",
        source_scores={RetrievalSource.VISUAL: index / 100},
        final_score=index / 100, metadata={"original": index},
    )

def test_success_identity_batches_scores_order_and_lifecycle(tmp_path):
    Backend.instances = 0
    store, inputs = make_store(tmp_path, 20), [candidate(i) for i in range(20)]
    before = [item.model_dump() for item in inputs]
    backend = Backend(lambda images: [image.getpixel((0, 0))[0] for image in images])
    reranker = RerankingService(
        cast(Corpus, store), RerankerConfig(batch_size=6), backend
    )
    output = reranker.rerank("query", inputs)
    assert len(output) == 20 and {item.frame_id for item in output} == {f"f{i}" for i in range(20)}
    assert [len_ for _, len_ in backend.calls] == [6, 6, 6, 2]
    assert store.calls == [item.frame_id for item in inputs]
    assert [item.frame_id for item in output] == [f"f{i}" for i in reversed(range(20))]
    assert {item.frame_id: item.reranker_score for item in output} == {f"f{i}": float(i) for i in range(20)}
    assert all(item.final_score == item.reranker_score for item in output)
    assert output[-1].frame_id == "f0"
    assert before == [item.model_dump() for item in inputs] and Backend.instances == 1
    assert reranker.rerank("query", []) == [] and len(reranker.rerank("query", [inputs[0]])) == 1
    tied = RerankingService(
        cast(Corpus, store), RerankerConfig(batch_size=20),
        Backend(lambda images: [1] * len(images)),
    )
    assert [item.frame_id for item in tied.rerank("q", inputs)] == [item.frame_id for item in inputs]
    assert all(RetrievalCandidate.model_validate(item.model_dump()) for item in output)

def test_candidate_image_failure_aborts_reranking(tmp_path):
    store, inputs = make_store(tmp_path, 8), [candidate(i) for i in range(8)]
    store.records["f1"].image_path = tmp_path / "missing.png"
    backend = Backend(lambda images: [0.4] * len(images))
    with pytest.raises(RerankerUnavailableError, match="frame_asset_missing"):
        RerankingService(
            cast(Corpus, store), RerankerConfig(batch_size=10), backend
        ).rerank("q", inputs)

@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError("late"), RerankerTimeoutError),
        (RuntimeError("CUDA out of memory"), RerankerUnavailableError),
        (Exception("model"), RerankerUnavailableError),
    ],
)
def test_backend_failure_aborts_request(tmp_path, error, expected):
    store, inputs, backend = make_store(tmp_path, 3), [candidate(i) for i in range(3)], Backend(error=error)
    reranker = RerankingService(
        cast(Corpus, store), RerankerConfig(batch_size=3), backend
    )
    with pytest.raises(expected):
        reranker.rerank("q", inputs)

def test_count_mismatch_aborts_request(tmp_path):
    store, inputs = make_store(tmp_path, 2), [candidate(0), candidate(0), candidate(1)]
    results, backend = iter([[0.5, 0.4], []]), Backend(lambda _: next(results))
    with pytest.raises(RerankerContractError):
        RerankingService(
            cast(Corpus, store), RerankerConfig(batch_size=2), backend
        ).rerank("q", inputs)


def test_invalid_score_aborts_request(tmp_path):
    store, inputs = make_store(tmp_path, 1), [candidate(0)]
    with pytest.raises(RerankerInvalidScoreError):
        RerankingService(
            cast(Corpus, store), RerankerConfig(),
            Backend(lambda _: [float("nan")])
        ).rerank("q", inputs)
