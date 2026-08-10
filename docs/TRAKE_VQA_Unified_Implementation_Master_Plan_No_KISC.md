
# MASTER IMPLEMENTATION PLAN
## Unified Multimodal Retrieval for KIS, Competition VQA, and TRAKE

**Project:** HCMAI / AIC HCMC 2026  
**Scope:** Sprint 2 to Sprint 6  
**Deployment assumption:** data and FAISS run locally; model inference may run on ThunderCompute or through an API; the operational objective is the earliest accurate submission.  
**Repository reviewed:** `src.zip` supplied in the conversation.

---

## Active scope override — 2026-08-06

This directive overrides every conflicting TRAKE implementation step later in
this document.

The active task-specific implementation scope is **Competition VQA only**.
Preserve existing KIS behavior and shared retrieval infrastructure when they
are required by VQA, but do not implement or modify TRAKE task logic in this
workstream.

Allowed TRAKE work is limited to stable integration seams needed for future
composition:

- authoritative `TRAKERequest`, `TRAKEResponse`, and submission contracts;
- the generic `TaskPipeline` / registry boundary;
- unified HTTP request and response dispatch;
- capability and health reporting;
- black-box integration fakes proving `input -> TRAKEPipeline -> output`.

The following TRAKE work is explicitly out of scope:

- event parsing or query decomposition;
- per-event retrieval policy or posting generation;
- candidate-video coverage ranking;
- monotonic, exhaustive, sparse, or other temporal alignment algorithms;
- gap and shot-transition penalties;
- original-frame refinement;
- k-best path generation;
- TRAKE-specific ranking, metrics, benchmarks, or ablations;
- any refactor of logic owned by the separate TRAKE implementation team.

Sections describing a TRAKE baseline, workflow, Sprint 5-6 tasks, evaluation,
or research direction are retained as historical/reference material only. They
must not be treated as executable tasks unless the user explicitly restores
TRAKE implementation scope in a later instruction.

The only required coordination contract with the TRAKE team is:

```text
TaskRequest
  -> PipelineRegistry
  -> TRAKEPipeline (externally owned black box)
  -> TaskResponse / TRAKEResponse
```

Do not depend on, duplicate, review, or rewrite the internal algorithm behind
that black box as part of the VQA workstream.

---

## 0. Evidence policy

This plan separates three kinds of statements:

- **[SOURCE]** verified from the uploaded source code or AIC 2026 specification.
- **[PAPER]** reported by the cited paper.
- **[PROPOSED]** an engineering or research decision inferred from the source and papers. It must be validated by implementation and experiment.

The VQA implementation should not be justified by intuition alone. The initial baseline is derived primarily from VRAG, SeViLA, NExT-GQA, GroundVQA, VideoTree, and query-aware frame-selection work. The TRAKE baseline follows the earlier DANTE/TARS-style monotonic-alignment survey, while sparse alignment is intentionally deferred until the exhaustive baseline is correct.

---

# 1. Executive decision

## 1.1 Recommended system shape

Build **one shared retrieval kernel** and place thin, task-specific pipelines above it:

```text
HTTP API
  |
  v
TaskRouter
  |-----------------------|-----------------------|
  v                       v                       v
KISPipeline           VQAPipeline           TRAKEPipeline
  |                       |                       |
  +-----------------------+-----------------------+
                          |
                          v
                     Shared Retrieval Kernel
        query processing / batched encoding / multimodal search
        fusion / caching / filtering / video aggregation / telemetry
                              |
             +----------------+-----------------+
             |                                  |
             v                                  v
       Local data + FAISS               Remote inference gateway
       frame/evidence stores            rerank / VLM / parser
```

## 1.2 Recommended VQA baseline

The first competition VQA implementation should be:

```text
event description + question
  -> split into retrieval query and answer question
  -> multimodal retrieval over the corpus
  -> group candidates by video
  -> expand top frames into short temporal windows
  -> query-aware localizer/filter selects answer-bearing evidence
  -> VLM answers several shortlisted windows
  -> normalize answers
  -> jointly rank video, frame, and answer
  -> return up to 100 submissions
```

This is a **VRAG-style retrieve-filter-answer baseline**, strengthened with a **SeViLA-style query-aware localizer** and **NExT-GQA-style grounding requirement**.

## 1.3 Deferred TRAKE baseline — reference only, do not implement

The first correct TRAKE implementation should be:

```text
raw TRAKE query
  -> ordered events E1 ... EN
  -> batched event embeddings
  -> per-event multimodal retrieval postings
  -> candidate-video coverage ranking
  -> exhaustive monotonic DP inside candidate videos
  -> local original-frame refinement
  -> k-best ordered paths
  -> official TRAKE submissions and metrics
```

Do **not** implement sparse DP first. Sparse DP must be tested against the exhaustive implementation and return the same optimum whenever its candidate set contains every frame.

---

# 2. VQA paper survey and implementation implications

## 2.1 Directly relevant papers

### VQA-1. VRAG: Retrieval-Augmented Video Question Answering for Long-Form Videos

**Reference:** Bao Tran Gia et al., CVPR Workshops 2025, [paper page](https://openaccess.thecvf.com/content/CVPR2025W/IViSE/html/Gia_VRAG_Retrieval-Augmented_Video_Question_Answering_for_Long-Form_Videos_CVPRW_2025_paper.html).

**[PAPER] Reported implementation**

1. GPT-4o decomposes a VQA prompt into a retrieval query and a question.
2. A multimodal search system retrieves top candidate segments using semantic, OCR, audio, and object evidence.
3. A reranker expands a relevant shot with three preceding and three succeeding shots, then asks a video MLLM for a relevance score.
4. For VQA, videos are divided into overlapping fixed-length chunks.
5. An MLLM filtering module makes a binary retain/discard decision for each chunk.
6. Retained segments are aggregated and passed to an answering module.
7. In the reported competition run, top ten reranked video results were processed by the VQA module.
8. Their small evaluation reported a better result for a 15-second chunk than a 30-second chunk, and a retrieval-based VideoLLaMA3 run outperformed their naive whole-video run.

**Limitations**

- The VQA experiment is very small, so its numerical result is not enough to fix our parameters.
- Binary MLLM filtering can be expensive if applied to every chunk of every candidate video.
- A fixed chunk size is unlikely to fit every question.
- The paper ranks segments, but AIC 2026 requires a concrete `frame_id` as well as an answer.

**Adoption**

- Use query decomposition.
- Reuse the existing multimodal retrieval system.
- Expand temporal context around retrieved frames.
- Run VLM filtering only after corpus pruning.
- Start with 8/15/30-second configurable windows and measure them.
- Bind each answer to an explicit evidence frame.

---

### VQA-2. SeViLA: Self-Chained Video Localization-Answering

**Reference:** Shoubin Yu et al., NeurIPS 2023, [paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f22a9af8dbb348952b08bd58d4734b50-Abstract-Conference.html), [official code](https://github.com/Yui010206/SeViLA).

**[PAPER] Reported implementation**

1. Uniformly sample `n` frames.
2. A BLIP-2-based Localizer independently scores each frame with a prompt asking whether the frame contains information needed to answer the question.
3. Select top `k` keyframes, where `k << n`; one reported setting is 32 candidate frames reduced to 4.
4. An Answerer receives the selected frame features and question to predict the answer.
5. A reverse chain creates pseudo-labels: a frame is positive when the Answerer can answer correctly from it.
6. The Localizer can be refined without manually labeled frame-level grounding data.

**Limitations**

- Its primary setup assumes a video has already been selected.
- The trained version requires QA labels; we currently do not possess gold training data.
- Independent top-k frame selection does not necessarily preserve temporal neighborhood or frame diversity.

**Adoption**

- Use the Localizer concept after corpus retrieval.
- Implement a training-free localizer first using a VLM relevance score.
- Keep the localizer and answerer as separate interfaces.
- Store answer-derived pseudo-labels later when the team creates a labeled benchmark.
- Apply temporal non-maximum suppression or diversity to prevent all selected frames from collapsing to one moment.

---

### VQA-3. Can I Trust Your Answer? Visually Grounded Video Question Answering / NExT-GQA

**Reference:** Junbin Xiao et al., CVPR 2024, [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Xiao_Can_I_Trust_Your_Answer_Visually_Grounded_Video_Question_Answering_CVPR_2024_paper.html), [dataset/code](https://github.com/doc-doc/NExT-GQA).

**[PAPER] Reported finding**

Strong QA accuracy does not imply that the model used the correct video evidence. The paper introduces temporal grounding labels tied to QA pairs and shows that answer predictions can be driven by language shortcuts or irrelevant context. It proposes a grounding mechanism that improves both evidence localization and QA.

**Limitations**

- The proposed Gaussian-mask learning is a supervised/weakly supervised model-training direction, not a direct corpus-retrieval implementation.
- The benchmark differs from AIC's exact video/frame/answer submission rule.

**Adoption**

- Treat VQA as a joint grounding-and-answering problem.
- Never rank a submission using answer confidence alone.
- Record an evidence frame/window for every generated answer.
- Evaluate frame correctness and answer correctness separately and jointly.
- Add a blind-text baseline to detect language-prior leakage in the internal benchmark.

---

### VQA-4. Grounded Question-Answering in Long Egocentric Videos

**Reference:** Shangzhe Di and Weidi Xie, CVPR 2024, [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Di_Grounded_Question-Answering_in_Long_Egocentric_Videos_CVPR_2024_paper.html), [official code](https://github.com/Becomebright/GroundVQA).

**[PAPER] Reported implementation direction**

The method integrates temporal query grounding and answer generation in one model to reduce error propagation between separate stages. It also uses LLM-generated data and proposes closed-ended QA evaluation to reduce ambiguity.

**Limitations**

- It is designed for long egocentric video and relies on task-specific training.
- An end-to-end grounding-answer model cannot search a large corpus efficiently without an outer retrieval stage.
- We do not yet have enough labeled data to train it as the initial system.

**Adoption**

- Keep a shared `GroundedAnswerCandidate` object that couples evidence and answer through the pipeline.
- Later compare two-stage and joint models on the internal benchmark.
- Do not use GroundVQA as the first production baseline.

---

### VQA-5. VideoTree: Adaptive Tree-based Video Representation for LLM Reasoning on Long Videos

**Reference:** Ziyang Wang et al., CVPR 2025, [paper](https://arxiv.org/abs/2405.19209), [official code](https://github.com/Ziyang412/VideoTree).

**[PAPER] Reported implementation**

1. Cluster frames by visual features.
2. Score clusters for relevance to the question.
3. Iteratively expand relevant clusters to obtain finer-grained frames.
4. Build a hierarchical tree carrying different levels of temporal detail.
5. Caption selected keyframes and pass the hierarchical text context to an LLM for answering.
6. The method is training-free and is reported to improve accuracy and reduce inference time against uniform-caption baselines.

**Limitations**

- Caption-based answering may miss details requiring original pixels, OCR, or fine action cues.
- Recursive tree construction for every corpus video would be too expensive.
- It does not directly solve corpus video retrieval.

**Adoption**

- Use hierarchy only inside a shortlisted video/window.
- Build an offline shot/scene hierarchy that can be reused across queries.
- Use query-adaptive expansion to refine ambiguous VQA and TRAKE windows.
- Preserve original images as well as captions in the final VLM context.

---

### VQA-6. M-LLM Based Video Frame Selection for Efficient Video Understanding

**Reference:** Kai Hu et al., CVPR 2025, [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Hu_M-LLM_Based_Video_Frame_Selection_for_Efficient_Video_Understanding_CVPR_2025_paper.html).

**[PAPER] Reported implementation**

A lightweight frame selector is trained using:

- a spatial signal based on single-frame importance judged by an MLLM;
- a temporal signal based on selecting multiple frames from captions;
- selected frames are then passed to a frozen downstream video MLLM.

**Limitations**

- Requires selector training and pseudo-supervision generation.
- Selector and answerer are optimized separately.
- Not a corpus retrieval method.

**Adoption**

- Preserve a pluggable `FrameSelector` interface.
- Log VLM relevance judgments as future pseudo-labels.
- Start with a deterministic/query-similarity selector, then train a lightweight selector only after benchmark data exists.

---

### VQA-7. VideoRAG: Scaling the Context Size and Relevance for Video Question Answering

**Reference:** Shivprasad Sagare et al., INLG 2024 System Demonstrations, [paper](https://aclanthology.org/2024.inlg-demos.3/).

**[PAPER] Reported implementation**

The system retrieves top-k query-relevant frames from a long video before passing them to the QA model. The paper reports qualitative improvement, especially for needle-in-a-haystack questions.

**Limitations**

- The publication is a short system demonstration.
- Evaluation is qualitative.
- It does not specify a strong grounded joint-ranking mechanism.

**Adoption**

- Use it as support for a minimal top-k-frame VQA baseline, not as sufficient evidence for final architecture choices.

---

### VQA-8. TimeChat and MA-LMM

**References:**  
- Shuhuai Ren et al., CVPR 2024, [TimeChat](https://openaccess.thecvf.com/content/CVPR2024/html/Ren_TimeChat_A_Time-sensitive_Multimodal_Large_Language_Model_for_Long_Video_CVPR_2024_paper.html).  
- Bo He et al., CVPR 2024, [MA-LMM](https://openaccess.thecvf.com/content/CVPR2024/html/He_MA-LMM_Memory-Augmented_Large_Multimodal_Model_for_Long-Term_Video_Understanding_CVPR_2024_paper.html).

**[PAPER] Reported implementation directions**

- TimeChat introduces timestamp-aware frame encoding and a sliding Q-Former for temporal localization and long-video reasoning.
- MA-LMM processes video online and stores historical information in a memory bank to avoid feeding all frames at once.

**Limitations**

- They are model-level long-video architectures.
- Loading or training such models does not remove the need to find the correct video in a large corpus.
- They are more expensive to integrate than a retrieval-augmented baseline.

**Adoption**

- Consider them answerer backends after the retrieval pipeline is stable.
- Require timestamps in every evidence item.
- Do not couple the application architecture to one specific VLM.

---

## 2.2 Survey conclusion for our system

The papers support five consistent implementation rules:

1. **Retrieve/localize before answer.** Feeding whole long videos to an MLLM is an expensive and weak baseline.
2. **Question-aware evidence selection is necessary.** Uniform frames can miss the decisive moment.
3. **Grounding and answer must remain coupled.** A plausible answer may be unsupported by the submitted frame.
4. **Coarse-to-fine evidence selection is practical.** Corpus retrieval should be cheap; expensive VLM processing should operate only on shortlisted windows.
5. **Start training-free when labels are absent.** Build logging and pseudo-label hooks now, and train selectors only after a reliable benchmark exists.

---

# 3. Current-source assessment and mapping

## 3.1 Existing reusable components

**[SOURCE] The repository already contains:**

- visual dense retrieval;
- caption, OCR, and ASR indexes;
- RRF fusion with task weights;
- frame materialization and evidence stores;
- local/remote embedding adapters;
- a remote reranking boundary;
- a one-frame VQA inference endpoint;
- latency fields in the public search response.

## 3.2 Gaps that block the selected plan

| Gap | Current source | Required change |
|---|---|---|
| Task routing | `SearchService._validate_task()` hard-codes VQA and TRAKE as unavailable | Registry of task pipelines |
| VQA contract | `VQARequest` accepts a known `frame_id` only | Competition request with event description + question |
| VQA visual backend | non-GLM branch loads `AutoTokenizer` and `AutoModelForCausalLM` | use an actual multimodal processor/model or an API adapter |
| Multiframe VQA | remote `/v1/vqa` accepts exactly one image | multi-image/window inference endpoint |
| Query encoding | each retriever encodes inside `search()` | batch once, reuse vectors |
| Fusion concurrency | RRF calls retrievers sequentially | concurrent modality execution |
| Telemetry | mutable `last_query_encoding_ms` on shared service objects | request-scoped traces |
| Filtering | active filters set `search_k = index.ntotal` | per-video/time posting or subset vector search |
| Reranking failure | exception aborts request | fallback to fused ranking with warning |
| Evaluation | recall cutoffs are `(1,5,10,100)` | official `(1,5,20,50,100)` and task metrics |
| TRAKE | schema/pipeline absent | event/posting/video/path contracts and exhaustive DP |

---

# 4. Goal and mission

## 4.1 Product mission

Deliver a competition retrieval service that:

1. accepts KIS, VQA, and TRAKE requests through stable contracts;
2. reuses one multimodal retrieval core;
3. returns an initial high-confidence result as early as possible;
4. degrades gracefully when a modality, reranker, or remote model is unavailable;
5. produces enough trace data to diagnose whether failure occurred in query parsing, retrieval, grounding, answering, temporal alignment, or ranking;
6. can emit ranked alternatives up to Top-100 for the official scoring protocol.

## 4.2 Research mission

Establish reproducible baselines for:

- VQA retrieval, grounding, and answering;
- exhaustive TRAKE monotonic alignment;
- accuracy-latency trade-offs;
- later sparse TRAKE optimization.

## 4.3 Non-goals for Sprint 2-6

- Training a new foundational VLM.
- Implementing sparse TRAKE DP before exhaustive DP.
- Calling an MLLM on every corpus video.
- Optimizing only answer text while ignoring evidence-frame correctness.
- Replacing every existing source module at once.

---

# 5. Target architecture

## 5.1 Core interfaces

```python
class TaskPipeline(Protocol):
    task_type: TaskType
    def execute(self, request: TaskRequest, context: RequestContext) -> TaskResponse: ...

class RetrievalKernel(Protocol):
    def encode_queries(self, request: BatchQueryRequest) -> QueryEmbeddingBatch: ...
    def retrieve(self, request: RetrievalBatchRequest) -> RetrievalBatchResult: ...

class InferenceGateway(Protocol):
    def rerank_windows(...): ...
    def localize_evidence(...): ...
    def answer_vqa(...): ...
    def parse_structured_query(...): ...

class FrameSelector(Protocol):
    def select(self, question, candidates, budget) -> list[EvidenceWindow]: ...

class TemporalAligner(Protocol):
    def align(self, events, postings, constraints) -> list[TrakePath]: ...
```

## 5.2 Request-scoped context

Do not store per-request timings in singleton retriever fields. Introduce:

```python
class RequestContext:
    request_id: str
    deadline_ms: int | None
    warnings: list[str]
    trace: PipelineTrace
    cache_policy: CachePolicy
    cancellation_token: CancellationToken | None
```

Each stage returns its own result and trace. This makes concurrent execution and multiple HTTP requests safe.

## 5.3 Main runtime components

```text
TaskRouter
PipelineRegistry
RequestContextFactory

QueryUnderstandingService
  - normalize
  - controlled expansion
  - VQA decomposition
  - TRAKE event parsing

RetrievalKernel
  - BatchQueryEncoder
  - ParallelModalitySearcher
  - FusionEngine
  - FilteredSearchStore
  - QueryEmbeddingCache

CandidateProcessing
  - TemporalDeduplicator
  - VideoDiversityRanker
  - VideoEvidenceAggregator
  - TemporalContextExpander

RemoteInferenceGateway
  - retry policy
  - deadline/timeout
  - circuit breaker
  - bulkhead semaphore
  - capability discovery

VQA
  - EvidenceBuilder
  - Frame/WindowLocalizer
  - MultiCandidateAnswerer
  - AnswerNormalizer
  - GroundedJointRanker

TRAKE
  - EventPostingRetriever
  - VideoCoverageRanker
  - ExhaustiveMonotonicAligner
  - OriginalFrameRefiner
  - KBestPathGenerator

Observability
  - stage traces
  - counters/histograms
  - structured logs
  - health/readiness
```

---

# 6. Task workflows

## 6.1 KIS workflow

```text
SearchRequest
 -> normalize exact user description
 -> optional controlled query variants
 -> batch encode each variant once
 -> search visual/caption/OCR/ASR concurrently
 -> per-variant/per-modality fusion
 -> bounded reranking with fallback
 -> temporal deduplication
 -> video diversity
 -> materialize Top-100
```


## 6.2 Competition VQA workflow

```text
VQARequest
  event_description
  question

1. Query decomposition
   retrieval_query := visually searchable event description
   question_query  := original question
   clue_queries    := optional object/OCR/ASR clues inferred without inventing facts

2. Corpus retrieval
   branch A: event description
   branch B: question text
   branch C: clue-specific retrieval when justified
   -> shared multimodal indexes
   -> fuse and group by video

3. Temporal candidate formation
   -> temporal dedup
   -> top candidate videos
   -> expand top frames to short windows
   -> merge overlapping windows

4. Evidence localization
   -> inexpensive similarity/coverage prefilter
   -> VLM relevance scoring only for shortlisted windows
   -> select diverse evidence frames/windows

5. Evidence bundle
   -> original images in chronological order
   -> caption/OCR/ASR/object evidence with timestamps
   -> retrieval and grounding scores

6. Answer generation
   -> answer several candidate windows/videos, not only the top frame
   -> require bounded structured output
   -> produce answer confidence and evidence frame selection

7. Normalization and joint ranking
   -> normalize Vietnamese/English answer variants
   -> rank by video retrieval + grounding + answer support
   -> diversify submissions across videos and evidence moments

8. Output
   -> <video_id>, <frame_id>, <answer>
   -> up to 100 ranked submissions
```

## 6.3 Deferred TRAKE baseline workflow — reference only

```text
raw query
 -> parser returns ordered events E1...EN
 -> batch event text and modality-specific query variants
 -> retrieve top postings for every event
 -> group postings by video
 -> rank videos by event coverage and evidence score
 -> for each candidate video:
      build dense event x keyframe score matrix
      run monotonic DP
      backtrack one path
      refine each event around selected keyframe on original frames
 -> generate k-best video/path/frame alternatives
 -> output <video_id>, <frame_id_1>, ..., <frame_id_N>
```

---

# 7. Proposed folder structure

The structure below extends the current repository without duplicating existing services.

```text
src/hcmai/
  api/
    routers/
      search.py                  # compatibility/general KIS endpoint
      vqa.py                     # competition VQA endpoint
      trake.py                   # TRAKE endpoint
      system.py

  common/
    schemas/
      task.py                    # discriminated request/response union
      search.py
      vqa.py                     # frame-level + competition VQA contracts
      trake.py
      telemetry.py
      evidence.py

  orchestration/
    task_router.py
    context.py
    pipeline.py                  # SearchService facade only
    pipelines/
      __init__.py
      base.py
      kis.py
      vqa.py
      trake.py
    setup.py
    materializer.py

  retriever/
    pipeline.py
    query_batch.py               # query vector batch contracts
    concurrent.py                # parallel modality execution
    cache.py                     # embedding cache
    filtered.py                  # subset/per-video search
    video_aggregation.py
    temporal_dedup.py
    diversity.py
    dense/
      index.py
      retriever.py
    fusion/
      rrf.py

  llm/
    gateway.py                   # stable remote inference facade
    resilience.py                # retry, circuit breaker, bulkhead
    capabilities.py
    adapters/
      http.py
      local.py

  vqa/
    __init__.py
    parser.py
    candidates.py
    windows.py
    evidence.py
    localizer.py
    answerer.py
    normalization.py
    ranking.py
    models.py

  trake/
    __init__.py
    parser.py
    postings.py
    video_ranker.py
    exhaustive_dp.py
    penalties.py
    refinement.py
    kbest.py
    models.py

  observability/
    tracing.py
    metrics.py
    logging.py

  evaluation/
    common.py
    kis.py
    vqa.py
    trake.py
    datasets.py
    benchmark.py

tests/
  unit/
    orchestration/
    retriever/
    vqa/
    trake/
  integration/
    test_kis_api.py
    test_vqa_api.py
    test_trake_api.py
    test_degraded_mode.py
  fixtures/
    tiny_corpus/
```

---

# 8. Engineering rules for every task

Every task below must follow this workflow:

1. Create a dedicated branch.
2. Write or update contracts before implementation.
3. Add unit tests for success and failure paths.
4. Add integration tests when public behavior changes.
5. Run formatting, lint, type checking, and tests.
6. Update README/config examples when behavior is user-visible.
7. The final task step is always commit and push.

Recommended quality command, adjusted to the project's final tooling:

```bash
ruff check .
ruff format --check .
mypy src/hcmai
pytest -q
```

---

# 9. Sprint 2 - Core retrieval production

## S2-T01. Introduce task-pipeline registry

**Goal:** remove task-specific branching from `SearchService` and provide one executable pipeline interface.

**Branch:** `feat/s2-task-pipeline-registry`  
**Files:** `orchestration/task_router.py`, `orchestration/pipelines/base.py`, `orchestration/pipeline.py`, tests.

**Steps**

1. Define a `TaskPipeline` protocol with `task_type` and `execute()`.
2. Define `PipelineRegistry` with `register()`, `get()`, duplicate-registration validation, and capability reporting.
3. Create `KISPipeline` as an adapter around the current search behavior; do not change ranking yet.
4. Modify `SearchService.search()` to ask the registry for the pipeline instead of calling `_validate_task()`.
5. Preserve the existing `SearchResponse` behavior for KIS and VKIS.
6. Map missing registered pipelines to `SearchPipelineUnavailableError`.
7. Add unit tests for registration, duplicate registration, unsupported task, and KIS backward compatibility.
8. Update health to derive task availability from the registry.
9. Run quality checks and API smoke tests.
10. **Commit and push:**
   ```bash
   git add src/hcmai/orchestration tests
   git commit -m "refactor(orchestration): add task pipeline registry"
   git push -u origin feat/s2-task-pipeline-registry
   ```

**Definition of Done**

- No hard-coded VQA/TRAKE rejection remains in `SearchService`.
- KIS responses are byte-for-byte equivalent for frozen fixtures.
- Health accurately reports registered pipelines.

---

## S2-T02. Add task-specific API contracts

**Goal:** prevent one generic `SearchRequest` from accumulating incompatible fields.

**Branch:** `feat/s2-task-contracts`  
**Files:** `common/schemas/task.py`, `search.py`, `vqa.py`, `trake.py`, API router tests.

**Steps**

1. Keep `SearchRequest` as the KIS/VKIS compatibility contract.
2. Define `VQARequest` with:
   - `event_description`;
   - `question`;
   - `top_k`;
   - optional filters;
   - optional language hint;
   - optional execution profile.
3. Define `VQASubmission` with:
   - `rank`, `video_id`, `frame_id`, `frame_idx`, `answer`;
   - retrieval, grounding, answer, and joint scores;
   - warnings/evidence summary.
4. Define `VQAResponse`.
5. Define `TRAKERequest` with raw query and optional already-parsed events.
6. Define `TRAKESubmission` and `TRAKEResponse`.
7. Define a discriminated `TaskRequest` / `TaskResponse` union where useful internally.
8. Add validators:
   - nonempty VQA event and question;
   - answer length bound;
   - TRAKE at least two ordered events when supplied;
   - maximum Top-100.
9. Export the contracts from `common/schemas/__init__.py`.
10. Add serialization tests and invalid-request tests.
11. **Commit and push:**
   ```bash
   git add src/hcmai/common/schemas tests/unit/common
   git commit -m "feat(schemas): add task-specific VQA and TRAKE contracts"
   git push -u origin feat/s2-task-contracts
   ```

**Definition of Done**

- Public contracts express the exact AIC outputs.
- Existing one-frame `VQARequest` is renamed or retained explicitly as an inference-provider contract, not confused with competition VQA.

---

## S2-T03. Replace mutable latency fields with request-scoped traces

**Goal:** make retrieval safe under concurrent modality execution and concurrent HTTP requests.

**Branch:** `refactor/s2-request-scoped-tracing`  
**Files:** `common/schemas/telemetry.py`, `observability/tracing.py`, retriever contracts.

**Steps**

1. Create `StageTrace` with start/end/duration, status, attempt count, cache hit, and error category.
2. Create `PipelineTrace` with named stages and aggregate helpers.
3. Change retriever return type from a bare list to:
   ```python
   RetrievalResult(candidates=[...], trace=RetrievalTrace(...), warnings=[...])
   ```
4. Remove or deprecate `last_query_encoding_ms` and `last_index_search_ms`.
5. Update `rank_candidates()` and `SearchLatency` population.
6. Ensure traces are created per request, never stored in singleton retriever objects.
7. Add concurrency tests that execute two retrieval calls simultaneously and verify traces do not overwrite each other.
8. Add JSON log fields for `request_id`, `task_type`, `stage`, `duration_ms`, and `status`.
9. **Commit and push:**
   ```bash
   git add src/hcmai/common src/hcmai/observability src/hcmai/retriever src/hcmai/orchestration tests
   git commit -m "refactor(observability): make retrieval traces request scoped"
   git push -u origin refactor/s2-request-scoped-tracing
   ```

**Definition of Done**

- No request timing depends on mutable shared fields.
- Parallel requests produce independent traces.

---

## S2-T04. Implement batched query encoding and vector-based search

**Goal:** encode a query or event batch once and reuse vectors across indexes.

**Branch:** `feat/s2-batched-query-encoding`  
**Files:** `retriever/query_batch.py`, `dense/retriever.py`, `retriever/pipeline.py`.

**Steps**

1. Define `QueryText`, `QueryEmbedding`, and `QueryEmbeddingBatch`, including model name, revision, source family, and normalized text.
2. Add `encode_text_batch(texts)` to the retrieval service.
3. Split `DenseRetriever.search()` into:
   - `encode(query_texts)`;
   - `search_vectors(query_vectors, top_k, filters)`.
4. For caption/OCR/ASR, encode one BGE batch and reuse it for all text indexes.
5. For TRAKE, accept all `E1...EN` in one batch.
6. Preserve single-query convenience methods for compatibility.
7. Validate embedding dimension/model against every target index.
8. Add a test encoder with a call counter; assert one call for three text modalities.
9. Benchmark old versus new encoding latency on a small fixture.
10. **Commit and push:**
   ```bash
   git add src/hcmai/retriever tests/unit/retriever
   git commit -m "feat(retriever): batch and reuse query embeddings"
   git push -u origin feat/s2-batched-query-encoding
   ```

**Definition of Done**

- Caption, OCR, and ASR do not independently encode identical query text.
- Event batches are supported without a Python loop of remote calls.

---

## S2-T05. Execute modalities concurrently with partial-failure support

**Goal:** overlap visual, caption, OCR, and ASR retrieval.

**Branch:** `feat/s2-parallel-modality-search`  
**Files:** `retriever/concurrent.py`, `fusion/rrf.py`, `retriever/pipeline.py`.

**Steps**

1. Create a `ModalitySearchJob` containing source, query vector batch, index, top-k, and filters.
2. Use a bounded `ThreadPoolExecutor` for local FAISS searches; do not create one executor per frame/event.
3. Return one `ModalitySearchResult` per source.
4. If an optional modality fails, record a warning and continue with successful sources.
5. Mark visual retrieval as required by default; make this configurable.
6. Update RRF to fuse only available sources and normalize configured weights over active sources when appropriate.
7. Pass `query_type` to every retriever; fix the current omission inside `RRFFusionRetriever`.
8. Add tests with artificial delays to prove wall-clock overlap.
9. Add tests for caption failure, ASR absence, and visual failure.
10. **Commit and push:**
   ```bash
   git add src/hcmai/retriever src/hcmai/common/config.py tests/unit/retriever
   git commit -m "feat(retriever): run modality searches concurrently"
   git push -u origin feat/s2-parallel-modality-search
   ```

**Definition of Done**

- Total search latency is approximately the maximum modality latency rather than the sum on the timing fixture.
- One optional source failure does not fail the request.

---

## S2-T06. Add resilient remote inference gateway

**Goal:** make ThunderCompute/API calls predictable under timeout and outage.

**Branch:** `feat/s2-inference-resilience`  
**Files:** `llm/gateway.py`, `llm/resilience.py`, `llm/adapters/http.py`, config.

**Steps**

1. Wrap the current HTTP adapter behind `InferenceGateway`.
2. Configure separate connect, read, write, and pool timeouts.
3. Add retry only for idempotent inference calls and transient failures:
   - connection reset;
   - timeout before response;
   - HTTP 429, 502, 503, 504.
4. Do not retry malformed requests or other deterministic 4xx responses.
5. Add exponential backoff with jitter and a maximum attempt count.
6. Add a circuit breaker:
   - closed under normal operation;
   - open after configurable consecutive failures;
   - half-open after cooldown;
   - close after a successful probe.
7. Add a semaphore bulkhead so one request cannot saturate every remote slot.
8. Propagate request deadlines; do not begin a retry when insufficient time remains.
9. Add capability discovery/readiness for embedding, reranking, multi-image VQA, and structured parsing.
10. Add unit tests using mocked HTTP responses and fake time.
11. **Commit and push:**
   ```bash
   git add src/hcmai/llm src/hcmai/common/config.py tests/unit/llm
   git commit -m "feat(llm): add retry timeout circuit breaker and bulkhead"
   git push -u origin feat/s2-inference-resilience
   ```

**Definition of Done**

- Remote failures are categorized and bounded.
- Circuit status appears in health.
- No infinite retries exist.

---

## S2-T07. Add reranking fallback to fused ranking

**Goal:** keep search usable when the reranker fails.

**Branch:** `fix/s2-reranker-fallback`  
**Files:** `orchestration/ranking.py`, `reranking/pipeline.py`, response warnings.

**Steps**

1. Define bounded reranker exceptions: unavailable, timeout, contract error, invalid score.
2. Wrap reranking in `rank_candidates()`.
3. On failure:
   - preserve the fused candidate order;
   - append a warning with a safe error category;
   - set reranking trace status to degraded;
   - never expose secrets or full remote responses.
4. Ensure partially scored batches are discarded unless explicitly supported.
5. Add config `reranker.required=false`.
6. Add tests for timeout, wrong result count, NaN score, and image load failure.
7. Verify KIS response remains valid under all fallback cases.
8. **Commit and push:**
   ```bash
   git add src/hcmai/orchestration src/hcmai/reranking src/hcmai/common tests
   git commit -m "fix(reranking): fall back to fused ranking on bounded failure"
   git push -u origin fix/s2-reranker-fallback
   ```

**Definition of Done**

- A reranker outage no longer produces a 500 for KIS/VQA/TRAKE candidate retrieval.

---

## S2-T08. Cache query embeddings and thumbnails

**Goal:** remove repeated encoding and image decode/resize work.

**Branch:** `feat/s2-retrieval-cache`  
**Files:** `retriever/cache.py`, `common/utils/image.py`, config, metrics.

**Steps**

1. Define an embedding-cache key:
   ```text
   model_name + revision + source_family + normalized_query + prompt_version
   ```
2. Implement an in-process LRU with TTL and memory bound.
3. Optionally add a disk cache interface but keep it disabled initially.
4. Cache immutable NumPy arrays; return read-only views or copies.
5. Invalidate naturally when model revision or prompt version changes.
6. Implement thumbnail caching as compressed JPEG bytes or resolved paths, not live PIL objects.
7. Include dataset version and frame ID in thumbnail keys.
8. Add cache hit/miss/eviction metrics.
9. Add tests for hit, miss, TTL expiry, model-version invalidation, and thread safety.
10. **Commit and push:**
   ```bash
   git add src/hcmai/retriever src/hcmai/common src/hcmai/observability tests
   git commit -m "feat(cache): cache query embeddings and thumbnails"
   git push -u origin feat/s2-retrieval-cache
   ```

**Definition of Done**

- Repeated identical queries avoid encoder calls.
- Cache cannot return embeddings from a different model/index version.

---

## S2-T09. Eliminate whole-corpus scans for video/time filters

**Goal:** support efficient candidate-local retrieval and refinement.

**Branch:** `feat/s2-filtered-vector-search`  
**Files:** `retriever/filtered.py`, `dense/index.py`, artifact builder.

**Steps**

1. During index build, persist normalized embedding arrays beside FAISS:
   ```text
   vectors.npy or vectors.f32.memmap
   ```
2. Build a posting table:
   - `video_id -> sorted embedding positions`;
   - timestamps aligned to positions.
3. At load time, memory-map the vectors and posting arrays.
4. For a small filtered subset:
   - resolve allowed positions without a pandas-wide mask;
   - compute `subset_vectors @ query_vector`;
   - partial-sort only the subset.
5. For unrestricted search, continue using FAISS.
6. Add a configurable threshold deciding subset brute force versus FAISS.
7. For future IVF indexes, keep an interface capable of using FAISS selectors.
8. Verify filtered results exactly match exhaustive `IndexFlatIP + postfilter`.
9. Benchmark one-video, ten-video, and unrestricted filters.
10. **Commit and push:**
    ```bash
    git add src/hcmai/retriever src/hcmai/embedding tests/unit/retriever
    git commit -m "feat(index): add exact subset search for video and time filters"
    git push -u origin feat/s2-filtered-vector-search
    ```

**Definition of Done**

- Searching one candidate video does not call `index.search(..., ntotal)`.
- Exactness tests pass against the old full-scan behavior.

---

## S2-T10. Make modality loading independently degradable

**Goal:** load visual, caption, OCR, and ASR indexes independently.

**Branch:** `fix/s2-optional-modality-loading`  
**Files:** `orchestration/setup.py`, health, config.

**Steps**

1. Load the visual index in its own required block.
2. Iterate text index paths independently.
3. Validate each loaded text index against visual dataset version.
4. Skip missing/invalid optional indexes with a startup warning.
5. Build `RetrievalService` with the sources that succeeded.
6. Expose active/inactive modality details in health.
7. Add startup tests for:
   - all sources;
   - visual only;
   - missing OCR;
   - mismatched ASR dataset version;
   - missing visual.
8. **Commit and push:**
   ```bash
   git add src/hcmai/orchestration src/hcmai/api tests/integration
   git commit -m "fix(startup): degrade independently for optional retrieval modalities"
   git push -u origin fix/s2-optional-modality-loading
   ```

**Definition of Done**

- Missing OCR or ASR does not disable visual/caption retrieval.

---

## S2-T11. Complete latency, tracing, and health reporting

**Goal:** expose enough data to optimize time-to-correct-submission.

**Branch:** `feat/s2-observability`  
**Files:** `observability/*`, `api/routers/system.py`, schemas.

**Steps**

1. Define standard stage names:
   - parse;
   - expansion;
   - encode;
   - search per modality;
   - fusion;
   - video aggregation;
   - rerank;
   - localization;
   - answer;
   - temporal alignment;
   - refinement;
   - materialization.
2. Emit structured logs with request and candidate counts.
3. Add latency histograms and failure counters.
4. Add `time_to_first_candidate` and, for anytime pipelines, `time_to_first_submission`.
5. Add readiness fields for each task and remote capability.
6. Redact prompts/images/answers from logs by default; allow bounded previews only in debug mode.
7. Add trace assertions in integration tests.
8. Document how to capture a benchmark trace.
9. **Commit and push:**
   ```bash
   git add src/hcmai/observability src/hcmai/api src/hcmai/common tests docs
   git commit -m "feat(observability): add stage traces and capability health"
   git push -u origin feat/s2-observability
   ```

**Definition of Done**

- Every request reports stage-level latency and degraded components.
- Logs can distinguish encoder, FAISS, network, VLM, and materialization delays.

---

## Sprint 2 release gate

Before Sprint 3 begins:

- run 100 repeated KIS requests over a tiny corpus;
- run at least 20 concurrent requests;
- inject remote timeout and missing-modality faults;
- verify no 500 response caused by optional services;
- save a frozen performance report;
- tag `core-retrieval-v1`.

---

# 10. Sprint 3 - Complete KIS

## S3-T01. Freeze a KIS golden retrieval path

**Goal:** create a stable reference before adding expansions and diversity.

**Branch:** `test/s3-kis-golden-path`

**Steps**

1. Build a tiny deterministic corpus fixture with visual/caption/OCR/ASR evidence.
2. Create at least 20 KIS queries covering visual, text, speech, and mixed cues.
3. Store expected candidate identities and minimum rank constraints.
4. Add API golden tests and direct-service tests.
5. Record exact model/index versions in the fixture manifest.
6. Add a command to regenerate expected outputs only with an explicit flag.
7. Document baseline recall and latency.
8. **Commit and push:**
   ```bash
   git add tests/fixtures tests/integration src/hcmai/evaluation docs
   git commit -m "test(kis): add frozen golden retrieval benchmark"
   git push -u origin test/s3-kis-golden-path
   ```

**Definition of Done**

- Any later ranking change produces a visible regression diff.

---

## S3-T02. Implement controlled query expansion

**Goal:** improve recall without inventing unsupported details.

**Branch:** `feat/s3-controlled-query-expansion`

**Steps**

1. Define a structured expansion output:
   - faithful paraphrases;
   - entity aliases;
   - translated variants;
   - modality hints;
   - explicit negative constraints.
2. Create a strict prompt that preserves names, numbers, colors, and actions.
3. Limit expansion count and token length.
4. Add deterministic no-LLM fallback: original query only.
5. Batch encode all variants.
6. Fuse variant results with a configurable penalty for generated variants.
7. Reject expansions that contradict source constraints using validation rules.
8. Add tests for proper nouns, numbers, negation, Vietnamese-English translation, and provider failure.
9. **Commit and push:**
   ```bash
   git add src/hcmai/query_suggestions src/hcmai/orchestration/pipelines/kis.py tests
   git commit -m "feat(kis): add bounded faithful query expansion"
   git push -u origin feat/s3-controlled-query-expansion
   ```

**Definition of Done**

- Original query is always retained.
- A failed expansion call never blocks KIS.
- No expansion silently changes a hard constraint in tests.

---

## S3-T03. Add temporal deduplication

**Goal:** prevent nearly identical adjacent frames from consuming Top-100.

**Branch:** `feat/s3-temporal-dedup`

**Steps**

1. Define dedup groups by `video_id` and timestamp/frame distance.
2. Keep the highest-scoring candidate in each local neighborhood.
3. Preserve suppressed candidates as alternates for later frame refinement.
4. Make the time window configurable by task.
5. Do not deduplicate across different videos.
6. Add tests for ties, repeated shots, and distant repeated events.
7. Add before/after diversity statistics to evaluation.
8. **Commit and push:**
   ```bash
   git add src/hcmai/retriever/temporal_dedup.py src/hcmai/orchestration tests
   git commit -m "feat(ranking): add task-aware temporal candidate deduplication"
   git push -u origin feat/s3-temporal-dedup
   ```

**Definition of Done**

- Top lists contain fewer adjacent duplicates without changing the best candidate per neighborhood.

---

## S3-T04. Add video diversity for Top-100

**Goal:** improve the chance that the correct video appears before later cutoffs.

**Branch:** `feat/s3-video-diversity`

**Steps**

1. Add a configurable maximum candidates per video for early ranking.
2. Implement a round-robin or score-penalized diversity policy.
3. Preserve strict score order inside each video's list.
4. Allow task-specific profiles:
   - KIS balanced;
   - VQA stronger video diversity;
   - TRAKE disabled until video aggregation.
5. Compare Recall@1/5/20/50/100 with and without diversity.
6. Add deterministic tie-breaking.
7. **Commit and push:**
   ```bash
   git add src/hcmai/retriever/diversity.py src/hcmai/common/config.py tests
   git commit -m "feat(ranking): diversify top results across videos"
   git push -u origin feat/s3-video-diversity
   ```

**Definition of Done**

- Diversity improves or preserves correct-video recall on the internal benchmark.
- Top-1 is not forcibly changed unless configured.

---

## S3-T05. Calibrate multimodal fusion

**Goal:** replace unverified equal weights with measured task profiles.

**Branch:** `exp/s3-fusion-calibration`

**Steps**

1. Export per-source ranks and raw scores for every benchmark query.
2. Normalize source scores separately; do not compare uncalibrated cosine scales directly.
3. Evaluate:
   - equal RRF;
   - task-weighted RRF;
   - score normalization + weighted sum;
   - query-conditioned modality gating as an experiment.
4. Use a frozen development split only for tuning.
5. Select weights by official cutoffs and latency, not one metric alone.
6. Save the chosen profile in versioned YAML.
7. Add config-validation and regression tests.
8. **Commit and push:**
   ```bash
   git add configs src/hcmai/retriever/fusion src/hcmai/evaluation reports
   git commit -m "exp(fusion): calibrate task-specific multimodal ranking"
   git push -u origin exp/s3-fusion-calibration
   ```

**Definition of Done**

- Every production weight has a benchmark report and version.

---

## S3-T06. Implement KIS regression benchmark

**Goal:** produce one command that reports quality and speed.

**Branch:** `feat/s3-kis-benchmark`

**Steps**

1. Implement official cutoffs `(1,5,20,50,100)`.
2. Report frame correctness, video recall, and latency p50/p90/p95.
3. Report source ablations and degraded-mode results.
4. Save machine-readable JSON and a Markdown summary.
5. Fail CI only on stable deterministic metrics; keep model-heavy benchmark as a scheduled/manual job.
6. Add dataset and model provenance to every report.
7. **Commit and push:**
   ```bash
   git add src/hcmai/evaluation tests reports docs
   git commit -m "feat(evaluation): add KIS quality and latency regression benchmark"
   git push -u origin feat/s3-kis-benchmark
   ```

---


# 11. Sprint 4 - Competition VQA

## S4-T01. Specify VQA baseline profiles

**Goal:** make the surveyed baselines executable and comparable.

**Branch:** `docs/s4-vqa-baselines`

**Steps**

1. Define four profiles:
   - `vqa_single_frame`: retrieve then answer one frame;
   - `vqa_vrag`: retrieve, window, filter, answer;
   - `vqa_localizer`: query-aware frame selector then answer;
   - `vqa_hierarchical`: coarse-to-fine scene refinement.
2. Define fixed budgets for candidate videos, windows, frames, and VLM calls.
3. Define common output and metrics.
4. Record paper inspiration and deviations.
5. Add YAML config schemas.
6. **Commit and push:**
   ```bash
   git add docs configs src/hcmai/common/config.py
   git commit -m "docs(vqa): specify paper-derived competition baselines"
   git push -u origin docs/s4-vqa-baselines
   ```

---

## S4-T02. Implement competition VQA parser

**Goal:** split the prompt into retrieval description and answer question.

**Branch:** `feat/s4-vqa-parser`

**Steps**

1. Define `ParsedVQAQuery`:
   - retrieval query;
   - original question;
   - question type;
   - required modalities;
   - answer language;
   - optional clue queries;
   - parser confidence.
2. Use the raw `event_description` directly as the default retrieval query.
3. Add a structured LLM parser only for controlled enrichment; do not permit invented scene facts.
4. Add deterministic rules for common question forms:
   - count;
   - color;
   - text/OCR;
   - speech;
   - before/after/action;
   - identity/object.
5. Validate parser output against the original fields.
6. Fall back to raw inputs on any parser failure.
7. Add Vietnamese and English tests.
8. **Commit and push:**
   ```bash
   git add src/hcmai/vqa/parser.py src/hcmai/common/schemas/vqa.py tests/unit/vqa
   git commit -m "feat(vqa): parse competition prompts into retrieval and answer queries"
   git push -u origin feat/s4-vqa-parser
   ```

---

## S4-T03. Add event-query and question-query retrieval branches

**Goal:** retrieve both the described event and answer-bearing clues.

**Branch:** `feat/s4-vqa-retrieval-branches`

**Steps**

1. Build a batch containing:
   - event description;
   - original question;
   - validated clue variants.
2. Route modality weights by question type:
   - OCR questions increase OCR branch;
   - speech questions increase ASR branch;
   - color/count remain visual-heavy.
3. Search all branches through the shared retrieval kernel.
4. Tag every candidate with branch and source provenance.
5. Fuse at frame level, then calculate video-level evidence.
6. Add ablation flags for event-only and question-only retrieval.
7. Add tests showing that OCR/ASR questions activate the expected branch without disabling visual search.
8. **Commit and push:**
   ```bash
   git add src/hcmai/vqa/candidates.py src/hcmai/retriever tests/unit/vqa
   git commit -m "feat(vqa): add event and question retrieval branches"
   git push -u origin feat/s4-vqa-retrieval-branches
   ```

---

## S4-T04. Aggregate VQA evidence by video

**Goal:** rank candidate videos before expensive VLM work.

**Branch:** `feat/s4-vqa-video-aggregation`

**Steps**

1. Define `VideoEvidenceCandidate`.
2. Aggregate:
   - best event rank;
   - best question rank;
   - number of modalities;
   - number of distinct temporal neighborhoods;
   - OCR/ASR clue coverage;
   - reranker score when available.
3. Add a coverage bonus and duplicate penalty.
4. Retain top `C` videos, configurable by profile.
5. Preserve frame lists for each video.
6. Add tests where one video has one extreme frame and another has consistent multi-source evidence.
7. Log correct-video candidate depth in benchmark mode.
8. **Commit and push:**
   ```bash
   git add src/hcmai/retriever/video_aggregation.py src/hcmai/vqa tests
   git commit -m "feat(vqa): rank videos from multi-branch evidence coverage"
   git push -u origin feat/s4-vqa-video-aggregation
   ```

---

## S4-T05. Form and merge temporal evidence windows

**Goal:** provide temporal context without processing the entire video.

**Branch:** `feat/s4-vqa-windows`

**Steps**

1. Convert top frames to windows centered on timestamps.
2. Support configurable 8, 15, and 30-second windows.
3. Clamp windows to video boundaries.
4. Merge overlapping windows within the same video.
5. Keep the source frames and scores attached.
6. Sample frames by:
   - shot boundaries when available;
   - otherwise timestamp-uniform sampling plus source frames.
7. Enforce chronological order.
8. Add tests for video start/end, overlap merging, and sparse metadata.
9. **Commit and push:**
   ```bash
   git add src/hcmai/vqa/windows.py src/hcmai/data tests/unit/vqa
   git commit -m "feat(vqa): build bounded temporal evidence windows"
   git push -u origin feat/s4-vqa-windows
   ```

---

## S4-T06. Build timestamped multimodal evidence bundles

**Goal:** give the answerer structured evidence, not only images.

**Branch:** `feat/s4-vqa-evidence-bundle`

**Steps**

1. Define `EvidenceItem` with source, text/value, frame ID, timestamp, confidence, and provenance.
2. For each window, collect:
   - captions;
   - OCR;
   - ASR overlapping the window;
   - detected objects if available;
   - selected images.
3. Deduplicate repeated text while preserving timestamp ranges.
4. Truncate by a deterministic evidence budget.
5. Never represent missing evidence as negative evidence.
6. Serialize the bundle for local and remote inference.
7. Add tests for source absence, duplicated ASR, and multilingual text.
8. **Commit and push:**
   ```bash
   git add src/hcmai/vqa/evidence.py src/hcmai/common/schemas/evidence.py tests
   git commit -m "feat(vqa): construct timestamped multimodal evidence bundles"
   git push -u origin feat/s4-vqa-evidence-bundle
   ```

---

## S4-T07. Fix and verify multimodal VLM capability

**Goal:** ensure the configured checkpoint truly accepts images and multiple frames.

**Branch:** `fix/s4-multimodal-vlm-loader`

**Steps**

1. Inspect model metadata at startup and identify the supported multimodal class.
2. Replace the generic tokenizer-only branch with `AutoProcessor` and an appropriate vision-language conditional-generation model.
3. Refuse startup when a configured VQA model cannot consume image inputs.
4. Add a capability probe:
   - one image;
   - multiple images;
   - timestamp/text evidence;
   - maximum pixel/frame budget.
5. Extend the remote provider contract from one image to ordered multi-image input.
6. Keep the old one-frame endpoint for backward compatibility.
7. Add a fake multimodal backend for tests.
8. Add a startup smoke test with a tiny supported checkpoint or mocked model.
9. **Commit and push:**
   ```bash
   git add src/hcmai/llm src/hcmai/common/schemas/vqa.py tests/unit/llm
   git commit -m "fix(vqa): load and validate a true multimodal answer model"
   git push -u origin fix/s4-multimodal-vlm-loader
   ```

**Definition of Done**

- The VQA capability cannot report ready when only a text model is loaded.

---

## S4-T08. Implement query-aware window localizer/filter

**Goal:** select answer-bearing windows before answer generation.

**Branch:** `feat/s4-vqa-localizer`

**Steps**

1. Define a `WindowLocalizer` interface.
2. Implement `SimilarityLocalizer` using existing retrieval evidence.
3. Implement `VLMYesNoLocalizer` inspired by SeViLA/VRAG:
   - prompt whether the window contains evidence needed to answer;
   - request structured relevance score and selected frame IDs.
4. Process only top candidate windows.
5. Add temporal diversity so selected windows do not all overlap.
6. Return localizer confidence and reasons as machine-readable labels, not free-form chain-of-thought.
7. Add timeout fallback to similarity ranking.
8. Add tests for correct structured response, malformed response, duplicate windows, and timeout.
9. **Commit and push:**
   ```bash
   git add src/hcmai/vqa/localizer.py src/hcmai/llm tests/unit/vqa
   git commit -m "feat(vqa): add query-aware evidence window localization"
   git push -u origin feat/s4-vqa-localizer
   ```

---

## S4-T09. Answer multiple candidate windows

**Goal:** avoid committing the entire query to one possibly wrong frame/video.

**Branch:** `feat/s4-vqa-multi-candidate-answering`

**Steps**

1. Define `GroundedAnswerCandidate` containing video, window, evidence frames, answer, model confidence, and warnings.
2. Answer the top `M` windows across top `C` videos under a call budget.
3. Batch windows when the provider supports it.
4. Require structured output:
   - short answer;
   - selected evidence frame ID;
   - answerability flag;
   - confidence bucket;
   - optional normalized answer.
5. Reject a returned frame ID outside the supplied evidence set.
6. Preserve unanswered candidates for fallback ranking only when configured.
7. Run independent calls concurrently under the gateway bulkhead.
8. Add tests for wrong-frame identity, empty answer, and partial provider failure.
9. **Commit and push:**
   ```bash
   git add src/hcmai/vqa/answerer.py src/hcmai/vqa/models.py src/hcmai/llm tests
   git commit -m "feat(vqa): answer and ground multiple shortlisted candidates"
   git push -u origin feat/s4-vqa-multi-candidate-answering
   ```

---

## S4-T10. Normalize answers

**Goal:** make semantically equivalent answers consistent without changing meaning.

**Branch:** `feat/s4-vqa-answer-normalization`

**Steps**

1. Implement deterministic normalization:
   - trim;
   - Unicode normalization;
   - whitespace;
   - case where appropriate;
   - punctuation;
   - Vietnamese number words and digits for count questions;
   - color aliases only from a controlled dictionary.
2. Preserve raw answer and normalized answer.
3. Add an optional semantic-equivalence service for evaluation only, not for rewriting submissions by default.
4. Never add information absent from the model answer.
5. Add tests for Vietnamese/English counts, colors, yes/no, and proper names.
6. **Commit and push:**
   ```bash
   git add src/hcmai/vqa/normalization.py tests/unit/vqa
   git commit -m "feat(vqa): add deterministic short-answer normalization"
   git push -u origin feat/s4-vqa-answer-normalization
   ```

---

## S4-T11. Implement grounded joint ranking

**Goal:** rank the complete `(video, frame, answer)` submission.

**Branch:** `feat/s4-vqa-joint-ranking`

**Steps**

1. Define score components:
   - video retrieval score;
   - event/frame retrieval score;
   - localizer score;
   - multimodal evidence coverage;
   - answer confidence;
   - answer consistency across neighboring windows;
   - penalties for unsupported/ambiguous answers.
2. Implement a transparent linear baseline:
   ```text
   joint = wv*video + wf*frame + wg*grounding + wa*answer + wc*consistency
   ```
3. Normalize each component before combination.
4. Keep every component in the API/debug response.
5. Add deterministic tie-breaking.
6. Add answer-diversity and video-diversity options for Top-100.
7. Tune weights only on development data; use equal/documented defaults before labels.
8. Add tests proving high answer confidence cannot overcome an invalid video/frame.
9. **Commit and push:**
   ```bash
   git add src/hcmai/vqa/ranking.py configs tests/unit/vqa
   git commit -m "feat(vqa): jointly rank grounded video frame and answer candidates"
   git push -u origin feat/s4-vqa-joint-ranking
   ```

---

## S4-T12. Add neighbor-window fallback for temporal questions

**Goal:** answer questions whose evidence spans multiple moments.

**Branch:** `feat/s4-vqa-temporal-fallback`

**Steps**

1. Detect temporal/causal question types from parsed query.
2. If the first answer is unanswerable or low confidence, expand the selected window before and after.
3. Add adjacent shots rather than arbitrary distant frames.
4. Rebuild evidence and retry once under a strict budget.
5. Record the fallback in trace and warnings.
6. Do not loop indefinitely.
7. Add tests for before/after, state change, and no-neighbor cases.
8. **Commit and push:**
   ```bash
   git add src/hcmai/vqa/windows.py src/hcmai/vqa/answerer.py tests
   git commit -m "feat(vqa): retry temporal questions with bounded neighbor context"
   git push -u origin feat/s4-vqa-temporal-fallback
   ```

---

## S4-T13. Produce ranked Top-100 VQA submissions

**Goal:** exploit official ranking cutoffs.

**Branch:** `feat/s4-vqa-topk-submissions`

**Steps**

1. Convert ranked candidates to official `video_id, frame_id, answer`.
2. Diversify:
   - candidate videos;
   - evidence windows;
   - nearby valid frames;
   - normalized answer forms only when semantically equivalent.
3. Prevent exact duplicate submissions.
4. Ensure rank order is deterministic.
5. Emit the first high-confidence submission immediately when streaming is enabled.
6. Continue generating later alternatives under the remaining deadline.
7. Add serialization and uniqueness tests.
8. **Commit and push:**
   ```bash
   git add src/hcmai/orchestration/pipelines/vqa.py src/hcmai/api/routers/vqa.py tests
   git commit -m "feat(vqa): return anytime ranked competition submissions"
   git push -u origin feat/s4-vqa-topk-submissions
   ```

---

## S4-T14. Create internal VQA benchmark and metrics

**Goal:** replace manual impressions with reproducible evaluation.

**Branch:** `feat/s4-vqa-benchmark`

**Steps**

1. Define annotation:
   - correct video;
   - valid frame interval;
   - canonical answers;
   - accepted aliases;
   - question type;
   - modality requirements;
   - ambiguity notes.
2. Annotate a pilot set with two people and adjudication.
3. Freeze development and test partitions.
4. Implement:
   - video recall;
   - frame interval accuracy;
   - answer exact/alias accuracy;
   - joint AIC-style correctness;
   - Top-k score;
   - latency and VLM-call cost.
5. Add blind-text, uniform-frame, event-only, and question-only baselines.
6. Report failure taxonomy:
   - wrong video;
   - wrong moment;
   - missing modality;
   - correct evidence/wrong answer;
   - unsupported answer.
7. Add a reproducible benchmark command.
8. **Commit and push:**
   ```bash
   git add src/hcmai/evaluation/vqa.py datasets/vqa tests reports
   git commit -m "feat(evaluation): add grounded competition VQA benchmark"
   git push -u origin feat/s4-vqa-benchmark
   ```

---

## Sprint 4 release gate

A VQA release candidate is accepted only when:

- it returns valid Top-100 submissions;
- every answer is bound to supplied visual evidence;
- remote model outage degrades to retrieval results rather than crashing;
- a benchmark report compares at least:
  - single frame;
  - naive whole candidate video where feasible;
  - VRAG-style windows;
  - localizer-enhanced windows;
- latency and API cost are reported.

---

# 12. DEFERRED - Sprint 5-6 TRAKE reference plan (do not implement)

> **Scope guard:** Every task in this section is inactive and externally owned.
> Do not create its branches, production code, tests, metrics, or benchmarks
> under the current VQA-only directive.

## S5-T01. Add TRAKE contracts and event parser

**Goal:** convert the query into validated ordered events.

**Branch:** `feat/s5-trake-parser`

**Steps**

1. Define `TrakeEvent`, `ParsedTRAKEQuery`, `EventConstraint`.
2. Accept both raw query and explicit event list.
3. Use a structured parser to extract `E1...EN` without changing order.
4. Preserve exact event descriptions and create bounded visual paraphrases.
5. Extract optional OCR/ASR/object cues only when stated.
6. Validate:
   - at least two events;
   - unique event indices;
   - no missing intermediate event;
   - no invented event.
7. Add deterministic parsing for numbered/bulleted input.
8. Fall back to user-supplied explicit events when parser fails.
9. Add Vietnamese/English tests.
10. **Commit and push:**
    ```bash
    git add src/hcmai/trake/parser.py src/hcmai/common/schemas/trake.py tests/unit/trake
    git commit -m "feat(trake): add ordered event contracts and parser"
    git push -u origin feat/s5-trake-parser
    ```

---

## S5-T02. Retrieve batched event postings

**Goal:** obtain per-event candidate frames with complete provenance.

**Branch:** `feat/s5-trake-event-postings`

**Steps**

1. Batch encode all event texts and variants.
2. Search visual and available text modalities concurrently.
3. Define `EventPosting`:
   - event ID;
   - video/frame identity;
   - frame index/timestamp;
   - source scores/ranks;
   - fused score;
   - variant provenance.
4. Deduplicate identical event/frame postings.
5. Keep a configurable depth for the exhaustive baseline; depth must be high enough to measure recall.
6. Persist optional debug matrices for analysis.
7. Add tests for event ordering, mapping identity, and missing modalities.
8. **Commit and push:**
   ```bash
   git add src/hcmai/trake/postings.py src/hcmai/retriever tests/unit/trake
   git commit -m "feat(trake): retrieve multimodal postings for event batches"
   git push -u origin feat/s5-trake-event-postings
   ```

---

## S5-T03. Rank candidate videos by event coverage

**Goal:** prioritize the correct video before DP.

**Branch:** `feat/s5-trake-video-ranking`

**Steps**

1. Group all postings by `video_id`.
2. Compute:
   - fraction of events covered;
   - best rank per event;
   - fused score per event;
   - temporal-order feasibility;
   - modality coverage.
3. Implement a transparent baseline score:
   ```text
   coverage bonus + sum reciprocal event ranks + order-feasibility bonus
   ```
4. Penalize videos requiring duplicate/inverted timestamps for many events.
5. Return top candidate videos with component scores.
6. Evaluate correct-video Recall@1/5/20/50/100.
7. Add tests for full coverage versus one very strong event.
8. **Commit and push:**
   ```bash
   git add src/hcmai/trake/video_ranker.py tests/unit/trake
   git commit -m "feat(trake): rank videos by ordered event evidence coverage"
   git push -u origin feat/s5-trake-video-ranking
   ```

---

## S5-T04. Implement exhaustive monotonic DP

**Goal:** establish the correctness reference.

**Branch:** `feat/s5-trake-exhaustive-dp`

**Steps**

1. For each candidate video, enumerate all indexed keyframes in temporal order.
2. Build dense semantic scores `S[i,t]` for every event and keyframe.
3. Implement recurrence:
   ```text
   D[i,t] = S[i,t] + max_{u<t}(D[i-1,u] - gap_penalty(u,t))
   ```
4. Implement the prefix-maximum optimization for the configured linear penalty where mathematically valid.
5. Store predecessor pointers.
6. Backtrack the best path.
7. Handle impossible paths and too-few-frame videos.
8. Add a slow brute-force enumerator for tiny test cases and compare exact optimum.
9. Test strict increasing frame indices and deterministic ties.
10. Benchmark complexity by event count and video length.
11. **Commit and push:**
    ```bash
    git add src/hcmai/trake/exhaustive_dp.py tests/unit/trake
    git commit -m "feat(trake): add exhaustive monotonic alignment baseline"
    git push -u origin feat/s5-trake-exhaustive-dp
    ```

**Definition of Done**

- DP matches brute-force optimum on all tiny randomized tests.

---

## S5-T05. Add configurable temporal and shot penalties

**Goal:** distinguish plausible sequences from semantically strong but incoherent paths.

**Branch:** `feat/s5-trake-penalties`

**Steps**

1. Define a `TransitionPenalty` protocol.
2. Implement:
   - no penalty;
   - linear gap penalty;
   - bounded preferred-gap penalty;
   - shot-jump penalty when shot metadata exists.
3. Keep hard monotonic order as the baseline.
4. Log transition components for every returned path.
5. Do not manually select one penalty without development experiments.
6. Add synthetic tests for short gaps, long gaps, repeated actions, and cuts.
7. **Commit and push:**
   ```bash
   git add src/hcmai/trake/penalties.py src/hcmai/trake/exhaustive_dp.py tests
   git commit -m "feat(trake): add configurable temporal coherence penalties"
   git push -u origin feat/s5-trake-penalties
   ```

---

## S5-T06. Refine selected keyframes on original video frames

**Goal:** target AIC's narrow accepted frame intervals.

**Branch:** `feat/s5-trake-frame-refinement`

**Steps**

1. Resolve selected keyframe to original video timestamp/frame index.
2. Decode a configurable local frame window from the original video.
3. Score dense frames against the corresponding event.
4. Include motion/transition signals when the event is action-boundary specific.
5. Select the best frame while preserving global event order.
6. When local refinements invert order, solve a small constrained local alignment.
7. Cache decoded windows for paths sharing the same neighborhood.
8. Add tests with synthetic videos and known frame maxima.
9. Measure keyframe-only versus refined accuracy.
10. **Commit and push:**
    ```bash
    git add src/hcmai/trake/refinement.py src/hcmai/data tests/unit/trake
    git commit -m "feat(trake): refine aligned events on original video frames"
    git push -u origin feat/s5-trake-frame-refinement
    ```

---

## S5-T07. Generate k-best ordered paths

**Goal:** produce diverse valid alternatives for official cutoffs.

**Branch:** `feat/s5-trake-kbest`

**Steps**

1. Define path identity by video plus ordered frame tuple.
2. Implement a k-best method:
   - keep multiple predecessor hypotheses per DP state; or
   - use best-path deviation with a priority queue.
3. Enforce strict temporal order for every path.
4. Diversify across:
   - candidate video;
   - event neighborhoods;
   - nearby refined frames.
5. Prevent duplicate paths.
6. Produce the first path before later alternatives when streaming.
7. Add tiny-case tests against full path enumeration.
8. **Commit and push:**
   ```bash
   git add src/hcmai/trake/kbest.py src/hcmai/orchestration/pipelines/trake.py tests
   git commit -m "feat(trake): generate diverse k-best monotonic paths"
   git push -u origin feat/s5-trake-kbest
   ```

---

## S5-T08. Implement official TRAKE metrics

**Goal:** reproduce the competition scoring locally.

**Branch:** `feat/s5-trake-metrics`

**Steps**

1. Represent ground truth as one video and one accepted interval per event.
2. Score zero when submitted video is wrong.
3. For the correct video, score the fraction of event frames inside intervals.
4. Implement R@1, R@5, R@20, R@50, and R@100.
5. Implement the final average across cutoffs.
6. Add boundary tests for inclusive interval endpoints.
7. Add the example from the AIC specification as a unit test.
8. **Commit and push:**
   ```bash
   git add src/hcmai/evaluation/trake.py tests/unit/evaluation
   git commit -m "feat(evaluation): implement official TRAKE scoring"
   git push -u origin feat/s5-trake-metrics
   ```

---

## S5-T09. Create internal TRAKE benchmark

**Goal:** obtain publishable and actionable evidence.

**Branch:** `data/s5-trake-benchmark`

**Steps**

1. Select a frozen subset of videos.
2. Write 100-200 queries with 3-6 ordered events.
3. Have two independent annotators mark one short valid interval per event.
4. Add alternate valid intervals when necessary.
5. Adjudicate disagreements.
6. Split development and test by video to prevent leakage.
7. Include categories:
   - action sequence;
   - state change;
   - OCR;
   - ASR;
   - repeated event;
   - long gaps;
   - short gaps;
   - ambiguous/OOD concepts.
8. Store annotation version and audit history.
9. Add a validator and benchmark loader.
10. **Commit and push:**
    ```bash
    git add datasets/trake src/hcmai/evaluation/datasets.py docs
    git commit -m "data(trake): add versioned internal alignment benchmark"
    git push -u origin data/s5-trake-benchmark
    ```

**Note:** Do not commit copyrighted raw videos when repository policy forbids it; commit manifests and annotation files only.

---

## S6-T01. Integrate the complete TRAKE baseline pipeline

**Goal:** expose end-to-end TRAKE through the API.

**Branch:** `feat/s6-trake-pipeline`

**Steps**

1. Assemble parser, postings, video ranker, exhaustive DP, refinement, and k-best generator.
2. Add execution budgets and candidate limits to config.
3. Register `TRAKEPipeline`.
4. Mount `/api/v1/trake`.
5. Populate full stage traces.
6. Add degraded behavior:
   - parser fallback;
   - missing text modalities;
   - VLM refinement unavailable;
   - insufficient event candidates.
7. Add end-to-end tests on the tiny corpus.
8. Run official metric tests.
9. Update health and README.
10. **Commit and push:**
    ```bash
    git add src/hcmai/orchestration/pipelines/trake.py src/hcmai/api src/hcmai/orchestration tests
    git commit -m "feat(trake): integrate exhaustive competition pipeline"
    git push -u origin feat/s6-trake-pipeline
    ```

---

## S6-T02. Run baseline ablations and freeze the reference

**Goal:** establish the baseline that later sparse optimization must beat.

**Branch:** `exp/s6-trake-baseline-ablation`

**Steps**

1. Evaluate:
   - independent event Top-1;
   - video coverage + greedy alignment;
   - exhaustive DP without penalty;
   - exhaustive DP with penalties;
   - keyframe-only;
   - original-frame refinement;
   - Top-1 versus k-best.
2. Report correct-video recall and per-event accuracy separately.
3. Report p50/p90/p95 latency and remote cost.
4. Save all configs and commit hashes.
5. Select one reference profile without using the test split.
6. Tag the code `trake-exhaustive-baseline-v1`.
7. **Commit and push:**
   ```bash
   git add reports configs
   git commit -m "exp(trake): freeze exhaustive alignment baseline results"
   git push -u origin exp/s6-trake-baseline-ablation
   git tag trake-exhaustive-baseline-v1
   git push origin trake-exhaustive-baseline-v1
   ```

---

# 13. Dependency order

```text
S2-T01 -> S2-T02 -> S2-T03 -> S2-T04 -> S2-T05
                        |          |          |
                        +----------+----------+
                                   v
                              S2-T06/T07
                                   |
                     S2-T08 -> S2-T09 -> S2-T11
                                   |
                               Sprint 3
                              KIS baseline
                                   |
                               Sprint 4
                     VQA parser/retrieval/windows
                                   |
                 VLM capability -> localizer -> answerer
                                   |
                  normalization -> joint rank -> Top-100
                                   |
                               Sprint 5
                TRAKE parser/postings/video ranking/exhaustive DP
                                   |
                    refinement -> k-best -> metrics/benchmark
                                   |
                               Sprint 6
                       integration and baseline freeze
```

---

# 14. Recommended execution profiles

## `fast`

- original query only;
- small top-k per modality;
- no remote reranking;
- one candidate video/window;
- one VQA answer call;
- TRAKE top few videos and no VLM verification.

## `balanced`

- controlled expansions;
- multimodal concurrent retrieval;
- bounded reranking;
- several candidate videos;
- VQA localizer and multiple answers;
- TRAKE refinement.

## `accurate`

- deeper retrieval;
- more candidate videos/windows;
- stronger remote reranker;
- local frame refinement;
- larger k-best budget.

## `competition_anytime`

1. execute `fast`;
2. emit first submission;
3. continue `balanced`;
4. emit improved/diverse submissions;
5. continue selected `accurate` refinements until deadline/operator stop.

---

# 15. Evaluation matrix

## 15.1 KIS

- video Recall@1/5/20/50/100;
- accepted-frame accuracy;
- latency;
- modality ablations;
- diversity effect.


## 15.2 VQA

- correct-video recall;
- evidence-frame accuracy;
- answer accuracy;
- joint `(video, frame, answer)` accuracy;
- grounded accuracy;
- Top-k official score;
- time to first submission;
- VLM calls and GPU/API seconds.

## 15.3 Deferred TRAKE evaluation reference — do not execute

- correct-video Recall@K;
- event-frame accuracy;
- query R-Score;
- official Final Score;
- keyframe versus refined frame;
- alignment latency;
- time to first full/partial path;
- k-best gain.

---

# 16. Key risks and controls

| Risk | Control |
|---|---|
| VLM returns plausible but unsupported answer | evidence-frame constraint, joint ranking, grounded benchmark |
| Remote inference is slow/unavailable | deadline, retries, circuit breaker, fallback, anytime profile |
| Query expansion changes meaning | structured schema, strict validation, original query retained |
| Text modality noise hurts fusion | task profiles, calibration, source ablations |
| Filtered refinement scans full index | persisted vector memmap + per-video postings |
| Parallel retrieval corrupts timing | request-scoped trace objects |
| VQA localizer consumes too many calls | cheap prefilter and strict candidate budget |
| TRAKE DP is blamed for corpus scoring cost | separate video pruning, scoring, alignment traces |
| No labels make optimization subjective | frozen human-annotated VQA and TRAKE benchmarks |

---

# 17. Definition of the first complete system

The system is considered functionally complete for these sprints when:

1. KIS, VQA, and TRAKE have mounted API routes and typed contracts.
2. All tasks use the same retrieval kernel.
3. Optional modality or remote-reranker failures do not crash retrieval.
4. VQA returns ranked `(video_id, frame_id, answer)` submissions grounded in supplied evidence.
5. TRAKE returns ranked `(video_id, frame_id_1...frame_id_N)` paths.
6. Official cutoffs are implemented.
7. Internal benchmarks and baseline reports exist.
8. Exhaustive TRAKE DP is frozen as a correctness reference.
9. The code is ready for the next research phase: sparse posting-level TRAKE alignment and adaptive expansion.

---

# 18. Decisions requiring team confirmation

The plan uses the following defaults unless the team changes them:

1. **VQA initial design:** training-free retrieval-localization-answering; no selector training until benchmark labels exist.
2. **VQA evidence unit:** short temporal window with one submitted representative evidence frame.
3. **Remote model:** model-agnostic gateway; the checkpoint is selected by benchmark rather than hard-coded into application logic.
4. **TRAKE first baseline:** exhaustive DP on candidate videos, followed by original-frame refinement.
5. **Git integration:** one branch and PR per task; squash or merge policy is a repository-level team choice.

---

# References

1. Bao Tran Gia et al. **VRAG: Retrieval-Augmented Video Question Answering for Long-Form Videos.** CVPR Workshops 2025.  
   https://openaccess.thecvf.com/content/CVPR2025W/IViSE/html/Gia_VRAG_Retrieval-Augmented_Video_Question_Answering_for_Long-Form_Videos_CVPRW_2025_paper.html

2. Shoubin Yu et al. **Self-Chained Image-Language Model for Video Localization and Question Answering.** NeurIPS 2023.  
   https://proceedings.neurips.cc/paper_files/paper/2023/hash/f22a9af8dbb348952b08bd58d4734b50-Abstract-Conference.html

3. Junbin Xiao et al. **Can I Trust Your Answer? Visually Grounded Video Question Answering.** CVPR 2024.  
   https://openaccess.thecvf.com/content/CVPR2024/html/Xiao_Can_I_Trust_Your_Answer_Visually_Grounded_Video_Question_Answering_CVPR_2024_paper.html

4. Shangzhe Di and Weidi Xie. **Grounded Question-Answering in Long Egocentric Videos.** CVPR 2024.  
   https://openaccess.thecvf.com/content/CVPR2024/html/Di_Grounded_Question-Answering_in_Long_Egocentric_Videos_CVPR_2024_paper.html

5. Ziyang Wang et al. **VideoTree: Adaptive Tree-based Video Representation for LLM Reasoning on Long Videos.** CVPR 2025.  
   https://arxiv.org/abs/2405.19209

6. Kai Hu et al. **M-LLM Based Video Frame Selection for Efficient Video Understanding.** CVPR 2025.  
   https://openaccess.thecvf.com/content/CVPR2025/html/Hu_M-LLM_Based_Video_Frame_Selection_for_Efficient_Video_Understanding_CVPR_2025_paper.html

7. Shivprasad Sagare et al. **VideoRAG: Scaling the Context Size and Relevance for Video Question Answering.** INLG 2024.  
   https://aclanthology.org/2024.inlg-demos.3/

8. Shuhuai Ren et al. **TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding.** CVPR 2024.  
   https://openaccess.thecvf.com/content/CVPR2024/html/Ren_TimeChat_A_Time-sensitive_Multimodal_Large_Language_Model_for_Long_Video_CVPR_2024_paper.html

9. Bo He et al. **MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding.** CVPR 2024.  
   https://openaccess.thecvf.com/content/CVPR2024/html/He_MA-LMM_Memory-Augmented_Large_Multimodal_Model_for_Long-Term_Video_Understanding_CVPR_2024_paper.html
