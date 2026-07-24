# Reranking — AI2-03 bounded multimodal reranker

This package owns model-agnostic bounded candidate reranking in
[`multimodal.py`](multimodal.py) and Qwen3-VL pair scoring in
[`qwen.py`](qwen.py). Branch `feat/ai2-reranker` is ready for engineering
review, but labelled quality evidence and production integration are still
blocking merge.

## 1. Task identity

| Field | Value |
|---|---|
| Task ID | AI2-03 |
| Owner | Khầy |
| Workstream | Multimodal reranking |
| Priority | P0 |
| Task-board status | In Progress (50%) |
| Task | Implement batched multimodal reranker |
| Branch | `feat/ai2-reranker` |
| Base | `main@9fbfaa2e0a2acb9b28e7c305528ccbadce34368f` |
| Implementation commit documented | `2f51545fce1208b34b7c6fef9881d895dca3d5ec` |
| Canonical package | `src/hcmai/reranking/` |
| Tests | [`tests/test_reranker.py`](../../../tests/test_reranker.py), [`tests/test_qwen_reranker.py`](../../../tests/test_qwen_reranker.py) |

The task board requires configured Qwen3-VL query-image scoring over existing
`RetrievalCandidate` inputs, `FrameStore` resolution, preserved IDs/count,
reranker/final scores, deterministic missing-image/OOM/timeout fallback,
Recall@1 or MRR impact, latency at depths 10/20/50/100, and one model load.
It explicitly prohibits corpus-wide retrieval.

The stale task-board package `src/aic/reranking/` maps to the canonical
`src/hcmai/reranking/` package without changing the domain structure.

## 2. Branch purpose

This branch owns a standalone bounded candidate reranker and a native
Qwen3-VL scoring adapter. The model-agnostic component resolves only supplied
candidate images, maps ordered scores back to those candidates, attaches
scores, and returns deterministic validated copies with fallback behavior.

It does not retrieve candidates, search FAISS, enumerate the corpus, construct
the application reranker, modify `SearchEngine`, define evaluation labels, or
claim ranking improvement from unlabelled compatibility data.

## 3. Implemented

- [`MultimodalReranker`](multimodal.py) preserves
  candidate count, identities, metadata, and duplicate inputs while resolving
  images through `FrameStore`.
- Configurable batches preserve score-to-candidate alignment. Successful
  `reranker_score` becomes `final_score`; no arbitrary fusion formula is used.
- Deterministic ordering is descending final score, then original position,
  then `frame_id`.
- Missing/corrupt images use per-candidate fallback. Timeout, OOM, malformed
  batch output, score-count mismatch, and model exceptions preserve the
  affected original ordering without retry.
- [`QwenRerankerScorer`](qwen.py) lazily loads and
  reuses one native model/processor pair, caches initialization failure, and
  implements official yes/no relevance-logit probability scoring.
- Native loading fixes `trust_remote_code=False`; no model loads at import.
- Real CPU compatibility used `Qwen/Qwen3-VL-Reranker-2B` revision
  `4bd860ac4f15ad1897a214615cccc700f8f71818`, BF16, batch size 1.

## 4. Not implemented or incomplete

- No labelled query/gold-frame set exists; baseline and reranked Recall@1/MRR
  have not been measured.
- Real Qwen latency exists only for one pair, three pairs, and one top-10
  compatibility run. Depths 20/50/100 were tested only with a fake scorer.
- Required `runs/reranker_baseline/` metrics and per-query results are absent.
- The local top-20 snapshot has three unlabelled Vietnamese queries and cannot
  support a quality claim.
- The branch is not wired into application startup or production
  `SearchEngine`.
- Current shared integration truncates to the rerank prefix and does not
  preserve the candidate tail when reranking is enabled; this is a Tech
  Lead-owned integration issue, not changed here.
- Representative GPU latency is unavailable.
- The branch is based on an older `main`; compatibility must be reviewed.

## 5. Verification evidence

### Engineering evidence

| Evidence type | Command or artifact | Result | Proves | Does not prove |
|---|---|---|---|---|
| Offline/regression tests | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src <isolated-python> -m pytest -q -p no:cacheprovider tests/test_reranker.py tests/test_qwen_reranker.py tests/test_schema.py` | 21 passed | Identity/count, batching, scores, deterministic fallbacks, lifecycle, schema | Ranking quality |
| Fake depth evidence | Temporary `/tmp` report | Depths 10/20/50/100 preserve count/IDs and batches for 3 queries | Functional bounded-depth behavior | Qwen latency or relevance |
| Native load/pairs | Temporary `/tmp` report | Load 58.25 s; peak RSS 4.75 GB; one pair 2.28 s; three pairs 6.86 s; stable objects | CPU compatibility and one-load lifecycle | Production throughput |
| Real top-10 | Temporary `/tmp` report | 10 IDs preserved; 21.55 s | End-to-end CPU top-10 plumbing | Recall/MRR or depth scaling |

### Quality evidence

There are no gold labels. The real top-10 fixture produced nearly identical
scores (`~0.440989`) on repetitive/black candidates. That observation is not
evidence of ranking improvement.

### Integration evidence

[`SearchEngine`](../search.py) exposes an optional reranker
hook, but application startup does not construct this scorer/reranker and the
tail-preservation/fallback warning contract is unresolved. This branch changes
neither file.

## 6. Artifacts

Available local, ignored plumbing:

```text
artifacts/fixture/candidates_top20.jsonl   # 3 queries × 20, no labels
data/aic_fixture/metadata/frames.parquet   # 100 canonical frames
```

Real CPU and functional-depth evidence exists only in `/tmp`; it is not a
permanent repository deliverable. The task-board-required
`runs/reranker_baseline/` directory, labelled predictions, Recall@1/MRR, and
real depth-20/50/100 latency reports are missing. No model files are committed.

## 7. Dependencies and cross-team contracts

| Dependency | Owner/task | Path or symbol | Use | Readiness | Blocking? | Modified here? |
|---|---|---|---|---|---|---|
| Candidate contract/output | AI1-04 / Tech Lead schema | [`RetrievalCandidate`](../common/schemas/retrieval.py) | Bounded input/output | Contract available; labelled top-100 absent | Yes for evaluation | No |
| Frame lookup | DE-03 | [`FrameStore`](../data/loader.py) | Resolve supplied IDs to images | Available | No | No |
| Candidate snapshot | AI1 | `artifacts/fixture/candidates_top20.jsonl` | Plumbing smoke | Unlabelled depth 20 only | Yes for quality/depths | No |
| Search orchestration | Tech Lead | [`SearchEngine`](../search.py) | Future integration | Hook exists; policy defects unresolved | Yes for system use | No |
| App startup/provider | SWE / Tech Lead | [`src/hcmai/app.py`](../app.py) | Construct shared online model once | Not implemented | Yes | No |
| Evaluation governance | Tech Lead / AI1 | `EvaluationQuery` and approved gold set | Recall@1/MRR | Missing | Yes | No |

Every dependency is referenced only; this branch does not modify AI1, DE,
shared search, schemas, or application code.

## 8. Current quality status

- Engineering: **PASS** — the standalone bounded contract and native CPU
  scorer are verified.
- Quality: **BLOCKED** — there are no approved labels or Recall@1/MRR results.
- Integration: **PENDING** — application and `SearchEngine` integration is
  Tech Lead/SWE-owned and absent.

## 9. Merge readiness

| Field | Decision |
|---|---|
| Merge target | Latest `main`, after compatibility and integration review |
| Current readiness | **READY FOR REVIEW — NOT READY TO MERGE** |
| Blocking conditions | Gold labels; frozen top-100 set; Recall@1/MRR; real depth latency; integration policy; current-main compatibility |
| Required approvals | AI1/evaluation owner for candidates/labels; Tech Lead for SearchEngine; SWE for startup |
| Downstream usage | Standalone experimental evaluation only |

Ready for review does not mean ready to merge.

## 10. Manual acceptance procedure

1. Obtain 15–20 approved labelled queries and one frozen top-100 candidate set
   per query from AI1/evaluation owners. Include easy, fine-grained action,
   text-dependent, object-attribute, temporally similar, and difficult-negative
   cases.
2. Run the identical candidates through the baseline and real Qwen reranker.
   Preserve configuration, checkpoint revision, per-query predictions, and
   latency under `runs/reranker_baseline/`.
3. Record:

   ```text
   query_id | query | original_top_k | reranked_top_k
   gold_or_manual_relevant_ids | relevant_rank_before | relevant_rank_after
   harmful_promotions | helpful_promotions | verdict | notes
   ```

4. Review frame thumbnails, original score/rank, reranker score, final
   score/rank, and visually tempting negatives for every query.
5. Measure Recall@1 and MRR on the unchanged candidate set. Report real P50/P95
   latency at depths 10, 20, 50, and 100 on the intended device.
6. PASS only if candidate IDs/count remain exact, Recall@1 does not regress,
   MRR improves, at least one additional relevant frame reaches rank 1 in a
   20-query set, and there is no critical harmful promotion. Any identity loss,
   systematic relevance regression, or text/visual shortcut is a blocking
   FAIL.
7. Preserve `metrics.json`, labelled inputs, per-query ranks, screenshots, and
   signed manual verdicts. The metrics file must follow the repository
   experiment output contract.

## 11. Known risks

- A plausible relevance score can still harm true ranking.
- Repetitive fixture frames hide semantic failures.
- CPU compatibility latency is not representative of production GPU latency.
- The shared prefix-only integration can drop candidate tails.
- Whole-request fallback needs warnings and latency accounting at integration.
- The stale branch base can conflict with current shared contracts.

## 12. Next actions

1. **AI1 / evaluation owner:** provide frozen labelled top-100 candidates.
2. **AI2:** run real Qwen quality and depth benchmarks on that exact set.
3. **Tech Lead:** freeze prefix-plus-tail, warnings, failure, and score policy
   for `SearchEngine`.
4. **SWE / Tech Lead:** construct the shared model once at application startup.
5. **AI2 + Tech Lead:** approve merge only after quality and integration gates.
