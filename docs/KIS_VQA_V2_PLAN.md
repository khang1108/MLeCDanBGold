# HCMAI KIS + VQA V2 Optimization Plan

Status: **Approved engineering roadmap**
Scope: **KIS + Competition Q&A/VQA only**
TRAKE: **externally owned; out of scope**

This document converts the current code audit and research survey into an
implementation plan that can be executed task-by-task by humans or coding
agents.

The goal is not to rewrite HCMAI around a new foundation model. The goal is to
fix measurable structural problems in the existing KIS/VQA pipelines, preserve
multimodal evidence, improve temporal grounding, and establish a clean
quality/latency baseline before P2 model research.

---

# 1. Outcome we want

At the end of V2, the system should behave conceptually as follows:

```text
Natural-language input
        |
        v
QueryPlanner
- task intent
- visual/OCR/speech/mixed modality need
- temporal need
- bounded subqueries
        |
        v
Query-conditioned multimodal retrieval
- SigLIP visual
- caption/BGE
- OCR/BGE
- ASR/BGE
        |
        v
RRF / calibrated rank-space fusion
+ full provenance
        |
        +-----------------------------+
        |                             |
        v                             v
      KIS V2                        VQA V2
        |                             |
  gated/evidence-              video aggregation
  preserving rerank                  |
        |                        temporal peaks
  second rank fusion                 |
        |                       bounded windows
  shot-aware NMS                     |
        |                       frame selection
        |                             |
        |                      frame-bound evidence
        |                             |
        |                      semantic localization
        |                             |
        |                       adaptive VLM input
        |                       1-2 / 2-4 / 4-8
        |                             |
        |                     grounded answerability
        |                     + confidence + ranking
        |                             |
        v                             v
 canonical Top-100               canonical Top-100
```

V2 is considered successful only if the frozen evaluation shows a better
quality/latency frontier than the current baseline.

---

# 2. Non-goals

Do not use this plan to:

- implement or review TRAKE internals;
- replace the entire model stack before measuring architecture fixes;
- train a new temporal/video encoder as the first optimization;
- introduce a generalized agent/query-planning framework;
- move local corpus/index data to the GPU VM without a measured reason;
- optimize only Top-1 while harming Top-20/50/100;
- claim a paper-derived method works on HCMAI before an ablation demonstrates it.

---

# 3. Source discipline

Implementation decisions in this plan come from three classes.

## 3.1 SOURCE — code-audit findings

The following were observed in the reviewed baseline and must be **verified on
the active branch before editing**, because teammates may have changed them.

### KIS audit findings

1. Four modality sources are fused with RRF and task-level/static weighting.
2. BGE query encoding can be reused across caption/OCR/ASR.
3. Qwen visual reranking is bounded to a top retrieval depth.
4. In the audited reranking path, reranker output can replace the fused score
   for reranked candidates rather than preserve retrieval evidence.
5. The visual reranker receives the image/query but not necessarily the OCR/ASR
   evidence responsible for retrieval.
6. KIS duplicate suppression is primarily time-window based.
7. Image loading/decode and multiple Qwen batches are plausible latency sources.

### VQA audit findings

1. VQA uses event/question retrieval branches, video aggregation, temporal
   windows, evidence construction, localization, VLM answering, and joint
   ranking.
2. The audited required-modality boost for OCR/ASR is stored in source evidence
   but does not become a first-class candidate score.
3. Video aggregation combines semantic retrieval scores and heuristic coverage
   bonuses that may have incompatible numerical scales.
4. Overlap merging can be transitive, allowing a nominal local window to grow
   much larger.
5. After merged-window construction, sampled frames may favor the earliest
   frames instead of the strongest/diverse evidence.
6. The audited localizer relies heavily on lexical token overlap, which is weak
   for Vietnamese query / English caption paraphrases.
7. Evidence can be flattened across frames, weakening explicit timestamp/frame
   association.
8. Multi-image VQA capability exists in the inference layer, while the audited
   orchestration may still choose one frame from a multi-frame window.
9. Event context and confidence/answerability handling require stronger
   grounding contracts.

These findings motivate tasks; they are not permission to skip verification.

## 3.2 PAPER — research grounding

The following papers support the design principles, not HCMAI-specific gains.

### QD-DETR — query-dependent relevance

**Query-Dependent Video Representation for Moment Retrieval and Highlight
Detection**, Moon et al., CVPR 2023.

Key lesson used here:

> video/clip relevance should depend explicitly on the query rather than treating
> all clips or all query contexts identically.

HCMAI adaptation:

- query-intent/modality planning;
- query-conditioned fusion and localization.

Reference:
https://openaccess.thecvf.com/content/CVPR2023/html/Moon_Query-Dependent_Video_Representation_for_Moment_Retrieval_and_Highlight_Detection_CVPR_2023_paper.html

### CG-DETR — suppress irrelevant query/clip interaction

**Correlation-guided Query-Dependency Calibration in Video Representation
Learning for Temporal Grounding**, Moon et al., 2023.

Key lesson used here:

> not every clip deserves the same amount of query-conditioned interaction;
> irrelevant evidence should be prevented from dominating representation or
> saliency.

HCMAI adaptation:

- query-conditioned window scoring;
- evidence pruning;
- do not reward modality/temporal coverage independent of relevance.

### UniVTG — temporal grounding as an explicit task

**UniVTG: Towards Unified Video-Language Temporal Grounding**, Lin et al., ICCV
2023.

Key lesson used here:

- temporal localization should be measured directly rather than hidden inside
  end-to-end answer accuracy.

HCMAI adaptation:

- correct-window and frame-budget recall gates.

### ChatVTG — coarse-to-fine multi-granularity localization

**ChatVTG: Video Temporal Grounding via Chat with Video Dialogue Large Language
Models**, CVPR Workshop 2024.

Key lesson used here:

- segment/multi-granularity descriptions can support coarse temporal
  localization before fine frame refinement.

HCMAI adaptation:

- P2 shot/window caption index;
- region-first, frame-second retrieval.

Reference:
https://openaccess.thecvf.com/content/CVPR2024W/PVUW/html/Qu_ChatVTG_Video_Temporal_Grounding_via_Chat_with_Video_Dialogue_Large_CVPRW_2024_paper.html

### NumPro — explicit temporal/frame identity

**Number it: Temporal Grounding Videos like Flipping Manga**, CVPR 2025.

Key lesson used here:

- explicit ordered frame identifiers can help a VLM associate visual content
  with temporal position.

HCMAI adaptation:

- numbered/timestamped chronological frames in temporal VQA prompts.

Reference:
https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Number_it_Temporal_Grounding_Videos_like_Flipping_Manga_CVPR_2025_paper.html

### VideoQA-TA — explicit temporal information for VideoQA

**VideoQA-TA: Temporal-Aware Multi-Modal Video Question Answering**, Wu et al.,
COLING 2025.

Key lesson used here:

- fine-grained video/question alignment and explicit temporal information can
  improve video reasoning.

HCMAI adaptation:

- preserve frame/timestamp identity;
- evaluate temporal VQA separately;
- use ordered multi-frame context when required.

Reference:
https://aclanthology.org/2025.coling-main.483/

## 3.3 PROPOSED — HCMAI hypotheses to test

The following are hypotheses until ablation confirms them:

- query-conditioned source routing improves OCR/speech/mixed KIS;
- preserving retrieval rank after Qwen reranking avoids destroying OCR/ASR
  evidence;
- BGE semantic localization improves multilingual window ranking;
- bounded local windows improve VQA evidence recall versus transitive merging;
- relevance/diversity frame selection beats earliest-frame sampling;
- multi-frame + numbered timestamps improves temporal VQA;
- adaptive rerank/VLM depth reduces latency without harming official score.

---

# 4. Core engineering principles

## 4.1 Measure the stage that fails

Do not diagnose all end-to-end failures as embedding-model failures.

Use the decomposition:

```text
KIS:
query -> retrieval -> fusion -> rerank -> dedup/diversity -> accepted frame

VQA:
query -> correct video -> correct window -> sufficient evidence -> correct answer
```

## 4.2 Preserve candidate provenance

Every important candidate should retain:

- frame/video identity;
- retrieval branch;
- modality rank/score;
- fused rank/score;
- reranker rank/score;
- localization score;
- evidence IDs;
- warnings/fallback status.

## 4.3 Prefer rank-space fusion until calibrated

SigLIP and BGE source scores are not assumed to share the same probability
scale.

Default V2 strategy:

```text
weighted RRF / second-stage RRF
```

Raw-score weighted sums require a calibration experiment.

## 4.4 Expensive inference comes after pruning

Never use Qwen over the full corpus.

Use:

```text
cheap retrieval
 -> small candidate set
 -> local evidence selection
 -> expensive VLM verification/reasoning
```

## 4.5 Temporal context is bounded and structured

A configured window must stay bounded.

Frames given to a temporal answerer must be:

- selected intentionally;
- chronological;
- identified by frame ID/timestamp;
- associated with their caption/OCR/ASR evidence.

---

# 5. Milestone overview

| Milestone | Goal                                        | Must complete before                  |
| --------- | ------------------------------------------- | ------------------------------------- |
| M0        | Frozen evaluation + traceable baseline      | all optimization claims               |
| M1        | Shared QueryPlan + runtime retrieval policy | KIS/VQA query-aware changes           |
| M2        | KIS P0 correctness                          | KIS P1 reranker/perf research         |
| M3        | VQA P0 correctness                          | learned selectors/model research      |
| M4        | P1 quality + latency                        | P2 architecture/model experiments     |
| M5        | P2 research                                 | optional / competition-time dependent |

Recommended execution order:

```text
EVAL-01 -> EVAL-02 -> EVAL-03
                    |
                    v
                 ARCH-01
                    |
                 RET-01
                    |
        +-----------+-----------+
        |                       |
      KIS-01                  VQA-01
        |                       |
      KIS-02                  VQA-02
        |                       |
      KIS-03                  VQA-03
      KIS-04                  VQA-04
                                |
                              VQA-05
                                |
                              VQA-06
                                |
                              VQA-07
                              VQA-08
                              VQA-09
                              VQA-10
        |                       |
        +-----------+-----------+
                    |
                    v
                  P1/P2
```

---

# 6. M0 — Measurement foundation

## EVAL-01 — Frozen KIS/VQA development set

Priority: **P0**
Effort: **Medium**
Dependencies: none

### Objective

Create a versioned internal benchmark that allows architecture changes to be
compared reproducibly.

### KIS categories

At minimum:

```text
visual_scene
visual_object
visual_action
ocr
speech
mixed
temporal_sequence
hard_negative
```

Hard-negative queries should distinguish near-matches such as:

```text
positive:
  female anchor wearing red beside a map screen

negative:
  female anchor wearing red beside a chart
  female anchor beside a map wearing blue
  male anchor wearing red beside a map
```

### VQA categories

At minimum:

```text
color
count
identity
object
action
ocr
speech
temporal
causal
general
```

Each labeled VQA example should contain as much of the following as practical:

```text
query/event description
question
correct video
accepted temporal interval
acceptable supporting frame IDs
raw acceptable answers
normalized/alias answers
notes for ambiguous cases
```

### Suggested code/artifacts

```text
data/evaluation/ or tests/fixtures/evaluation/
src/hcmai/evaluation/
scripts/evaluate_kis.py
scripts/evaluate_vqa.py
runs/<experiment-id>/
```

Use the repository's existing evaluation conventions instead of duplicating
schemas.

### Metrics

KIS:

- official Mean Top-k R-Score when available;
- Hit/Recall@1/5/20/50/100;
- MRR;
- accepted-frame accuracy;
- category breakdown.

VQA:

- correct-video Recall@1/5/10;
- correct-window Recall@1/3/8;
- selected-frame/evidence Recall@budget;
- raw answer exact match;
- normalized/alias match;
- joint video-frame-answer accuracy.

### Tests

- malformed evaluation rows rejected;
- canonical frame IDs resolve correctly;
- metric fixtures with hand-computed expected values;
- deterministic results.

### Definition of Done

- frozen query-set version exists;
- same config/commit produces same metric output;
- metrics are broken down by task category;
- run output records query-set version and git commit.

---

## EVAL-02 — Stage-wise observability

Priority: **P0**
Effort: **Small/Medium**
Dependencies: none

### Objective

Make latency and candidate loss visible at each stage.

### Required trace stages

Shared:

```text
query planning
encoding
visual search
caption search
OCR search
ASR search
fusion
```

KIS:

```text
reranker image load/decode
reranker inference
second fusion
NMS/diversity
materialization
```

VQA:

```text
candidate merge
video aggregation
temporal peak/window construction
frame selection
evidence construction
semantic localization
VLM call
fallback
joint ranking
```

### Record

- duration;
- input/output count;
- cache hit/miss;
- backend/provider;
- fallback/warning;
- remote calls/images per call.

### Definition of Done

A single baseline run can answer:

- where time is spent;
- how many candidates are discarded at each stage;
- whether a remote fallback occurred;
- how many VLM images/calls were used.

---

## EVAL-03 — Freeze current baseline

Priority: **P0**
Effort: **Small once EVAL-01/02 exist**

### Objective

Produce the baseline metrics against which every V2 task is compared.

### Required outputs

```text
runs/baseline-<date-or-id>/
  config snapshot
  git commit
  query-set version
  predictions
  metrics.json
  latency.json or trace summaries
  failures/warnings
```

### Definition of Done

No P0 optimization task should be described as an improvement without a
comparison to this baseline.

---

# 7. M1 — Shared query planning and retrieval policy

## ARCH-01 — Introduce QueryPlan

Priority: **P0**
Effort: **Medium**
Dependencies: EVAL-01 preferred

### Objective

Represent what evidence a query requires without routing every request through
an LLM.

### Proposed contract

Conceptually:

```python
class QueryIntent(Enum):
    VISUAL = "visual"
    OCR = "ocr"
    SPEECH = "speech"
    MIXED = "mixed"
    TEMPORAL = "temporal"

class QueryPlan:
    original_query: str
    intent: QueryIntent
    required_modalities: set[RetrievalSource]
    visual_query: str | None
    caption_query: str | None
    ocr_query: str | None
    asr_query: str | None
    subqueries: tuple[str, ...]
    temporal_required: bool
```

VQA may extend with:

```text
event_description
question
answer_type
```

### Implementation strategy

Start with deterministic rules and existing parser information.

Optional LLM rewriting can be added later behind a bounded/configured provider.

### Rules

- preserve original query/event;
- bounded number of generated subqueries;
- log planner decision;
- no arbitrary LLM-produced source weights;
- no generalized agent framework.

### Likely areas

```text
src/hcmai/query/                 new if repository structure allows
src/hcmai/vqa/parser.py
src/hcmai/orchestration/
src/hcmai/common/schemas/       only if cross-component contract is required
```

Avoid a shared schema change if a private orchestration model is enough.

### Tests

- visual query classification;
- explicit OCR phrase;
- explicit speech phrase;
- mixed visual+text query;
- temporal cue;
- Vietnamese and English variants;
- deterministic output.

### Definition of Done

KIS and VQA can request a policy based on a verified `QueryPlan` without
changing canonical request/response identity.

---

## RET-01 — Runtime query-conditioned modality policy

Priority: **P0**
Effort: **Medium**
Dependencies: ARCH-01

### Objective

Replace one task-wide source policy with a runtime policy derived from query
intent.

### Target behavior

Profiles conceptually include:

```text
visual
ocr
speech
mixed
temporal
```

A profile determines:

- active sources;
- RRF weights;
- candidate budgets;
- whether expensive visual reranking is relevant.

### Starting policy philosophy

Do not freeze arbitrary numeric weights in this plan.

Expected qualitative behavior:

| Intent   | Visual     | Caption | OCR             | ASR             |
| -------- | ---------- | ------- | --------------- | --------------- |
| visual   | high       | high    | low             | low             |
| OCR      | medium     | medium  | high            | low/off         |
| speech   | low/medium | medium  | low             | high            |
| mixed    | high       | high    | medium          | medium          |
| temporal | high       | high    | query-dependent | query-dependent |

Tune numbers on the frozen dev set.

### Likely areas

```text
src/hcmai/retriever/pipeline.py
src/hcmai/retriever/fusion/rrf.py
src/hcmai/common/config.py
```

Prefer an optional policy argument with a backward-compatible default so TRAKE
shared callers do not break.

### Tests

- legacy/default call produces deterministic current-style behavior;
- OCR policy changes source weights/activation;
- failed optional source still returns partial result;
- provenance records actual policy.

### Definition of Done

The same `RetrievalService` can serve visual/OCR/speech/mixed queries with
explicit runtime policies and no duplicated search implementation.

---

## RET-02 — Fusion calibration harness

Priority: **P1**
Effort: **Small/Medium**
Dependencies: EVAL-01, RET-01

### Objective

Tune source weights with measurements instead of intuition.

### Experiments

- static equal RRF;
- per-intent weighted RRF;
- optional normalized raw-score + RRF hybrid;
- source ablations.

### Gate

Raw-score fusion is not promoted unless calibration improves the relevant
frozen categories without destabilizing others.

---

# 8. M2 — KIS V2

## KIS-01 — Query-aware KIS retrieval

Priority: **P0**
Effort: **Medium**
Dependencies: ARCH-01, RET-01

### Objective

Make KIS retrieval respect the evidence type requested by the query.

### Implementation steps

1. Build/obtain `QueryPlan` in KIS orchestration.
2. Create runtime retrieval policy.
3. Search only/primarily relevant sources according to policy.
4. Preserve per-source rank/score and branch provenance.
5. Return fused candidates exactly as before at the public boundary.

### Metrics

Compare category-level:

```text
visual Hit@K
OCR Hit@K
speech Hit@K
mixed Hit@K
temporal Hit@K
```

### Definition of Done

- OCR/speech queries demonstrably use OCR/ASR ranking features;
- visual performance does not regress beyond approved tolerance;
- public KIS response contract remains compatible.

---

## KIS-02 — Evidence-preserving reranking

Priority: **P0**
Effort: **Medium**
Dependencies: KIS-01

### Objective

Prevent an image-only reranker from destroying good OCR/ASR/caption retrieval
results.

### Current hypothesis

The audited path can behave approximately as:

```text
multimodal fused rank
    -> top-N Qwen image scores
    -> Qwen score becomes final score
```

Verify this on the active branch.

### V2 strategy

Keep both ranks:

```text
retrieval_rank
visual_reranker_rank
```

Then fuse them in rank space:

```text
second_stage_score =
    RRF(retrieval_rank, weight=query-policy)
  + RRF(reranker_rank, weight=query-policy)
```

For OCR/speech intent, visual-reranker contribution should be weaker or can be
skipped after measurement.

### Important rule

Do not add raw Qwen score directly to raw RRF score without calibration.

### Tests

Hand-computed fixtures:

- strong ASR candidate remains competitive after visual reranking;
- visual candidate can improve on visual queries;
- reranker failure exactly falls back to retrieval rank;
- identity remains unchanged.

### Ablation

```text
K0 current
K1 query-aware retrieval only
K2 + preserve retrieval via second fusion
K3 + gated Qwen by intent
```

### Definition of Done

K2/K3 improve or preserve the frozen quality metric and specifically avoid
OCR/ASR regressions caused by image-only reranking.

---

## KIS-03 — Adaptive rerank depth

Priority: **P1**
Effort: **Small**
Dependencies: EVAL-02, KIS-02

### Objective

Reduce Qwen image calls without losing accepted-frame recall.

### Benchmark

At minimum:

```text
rerank depth 20
30
50
100
```

Record:

- pre-rerank correct-frame Recall@depth;
- final official/category metrics;
- reranker image load/decode latency;
- Qwen inference latency.

### Promotion rule

Use the smallest depth whose quality loss is within an explicitly accepted
margin.

Later adaptive strategy:

- high retrieval margin/confidence -> smaller depth;
- ambiguous retrieval -> larger bounded depth.

### Definition of Done

Measured latency reduction with no material quality regression.

---

## KIS-04 — Shot-aware duplicate suppression

Priority: **P1**
Effort: **Small/Medium**
Dependencies: KIS-02

### Objective

Avoid suppressing distinct shots merely because they occur within a fixed
number of seconds.

### Proposed logic

Prefer suppression when:

```text
same video
AND
same shot OR very high visual similarity
AND
close timestamp
```

Different shots should normally survive even when temporally close.

### Tests

- adjacent frames same shot -> suppress duplicate;
- 2-second shot cut -> keep both;
- no shot ID -> deterministic time/similarity fallback;
- top-K stability.

### Metrics

- duplicate-frame rate;
- video/shot diversity;
- accepted-frame accuracy.

---

## KIS-05 — Multimodal evidence-aware reranker

Priority: **P1/P2**
Effort: **Medium/Large**
Dependencies: KIS-02, EVAL-01

### Objective

Test whether top candidates benefit from a reranker that sees the evidence that
retrieved them.

### Proposed prompt/input

```text
Query: ...
Image: <image>
Caption: ...
OCR: ...
Speech near timestamp: ...
```

### Constraints

- run only on a small top-N;
- preserve candidate identity;
- compare against second-stage fusion without multimodal Qwen;
- do not promote if latency cost is not justified.

---

## KIS-06 — Temporal KIS query mode

Priority: **P1**
Effort: **Medium**
Dependencies: ARCH-01, KIS-01

This is **not TRAKE**.

### Objective

Handle a single KIS query containing explicit order such as:

```text
first A, then B
```

### Proposed baseline

1. detect temporal intent;
2. decompose into a bounded number of ordered subqueries;
3. retrieve each subquery;
4. bonus same-video ordered matches;
5. return ordinary KIS frame/video candidates.

Do not implement TRAKE alignment algorithms under this task.

---

# 9. M3 — VQA P0 correctness

The VQA priority is to preserve the correct video/window/evidence before
changing the VLM.

## VQA-01 — Make required modality affect candidate ranking

Priority: **P0**
Effort: **Small**
Dependencies: EVAL-03 preferred

### Objective

Ensure OCR questions really reward OCR evidence and speech questions really
reward ASR evidence.

### Implementation

Verify the active scoring path and remove dead/diagnostic-only boost logic.

Use explicit features such as:

```text
branch support rank
required modality rank
```

Prefer rank-space fusion until score calibration exists.

### Tests

- OCR candidate order changes when OCR evidence is strong;
- speech candidate order changes when ASR evidence is strong;
- general visual query is not accidentally dominated by OCR/ASR.

### Definition of Done

Required modality is observable in candidate ranking and trace/provenance.

---

## VQA-02 — Rewrite video aggregation on comparable features

Priority: **P0**
Effort: **Medium**
Dependencies: VQA-01

### Objective

Prevent arbitrary coverage bonuses from numerically overwhelming semantic
retrieval evidence.

### V1 feature design

Use rank-comparable features:

```text
best event-support rank
best contextual-question rank
best required-modality rank
local temporal support rank/density
```

Candidate formula can use weighted RRF.

### Do not

- add modality-count bonuses larger than the semantic score scale without
  normalization;
- reward scattered noisy hits simply for existing in many neighborhoods.

### Metrics

- correct-video Recall@1/5/10;
- category breakdown;
- regression on general VQA.

### Definition of Done

New aggregation beats or matches baseline correct-video recall and has a
clear, testable score decomposition.

---

## VQA-03 — Bounded temporal peak/window construction

Priority: **P0**
Effort: **Medium**
Dependencies: VQA-02

### Objective

Replace transitive overlap merging with local evidence regions whose duration
remains bounded.

### Target algorithm

```text
candidate frames sorted by relevance
    |
    v
select strongest uncovered temporal peak
    |
construct fixed window around peak
    |
temporal NMS against accepted windows
    |
continue until window budget
```

### Hard invariant

For configured duration `W`:

```text
window.end_ms - window.start_ms <= W
```

### Tests

- chain overlap A-B-C does not create a giant A-C window;
- boundary near video start/end;
- duplicate peaks collapse deterministically;
- source anchor remains inside its window.

### Metrics

- correct-window Recall@1/3/8;
- average/max window duration;
- number of windows/query.

---

## VQA-04 — Question-aware frame selector

Priority: **P0**
Effort: **Medium**
Dependencies: VQA-03

### Objective

Replace earliest-frame truncation with a fixed-budget evidence selector.

### Training-free V1 algorithm

Inputs:

```text
frames inside bounded window
query/event/question
retrieval source anchors
modality evidence
```

Selection priorities:

1. keep at least one strong retrieval/evidence anchor;
2. score query/question relevance;
3. include required OCR/ASR evidence frame when appropriate;
4. add temporal diversity via MMR or before/after coverage;
5. sort selected frames chronologically before answering.

### Example budget=4

```text
F1 strongest question-relevant frame
F2 strongest required-modality frame
F3 useful earlier context
F4 useful later context
```

Deduplicate when the same frame fills multiple roles.

### Metrics

- accepted-frame/evidence Recall@1/2/4/8;
- temporal coverage;
- answer accuracy using oracle selected window.

### Definition of Done

Selector improves evidence recall versus earliest-frame sampling at the same
frame budget.

---

## VQA-05 — BGE semantic localizer

Priority: **P0**
Effort: **Medium**
Dependencies: VQA-03

### Objective

Replace lexical overlap as the primary semantic signal for multilingual
window localization.

### Window text representation

Conceptually:

```text
[CAPTION]
...
[OCR]
...
[ASR]
...
```

Encode:

```text
query/event+question
window evidence text
```

with BGE-M3 or the configured compatible text encoder.

### Fusion

Start with rank fusion between:

- retrieval/window rank;
- semantic evidence rank;
- required-modality rank.

Retain lexical overlap only as a diagnostic or small ablation feature.

### Tests

- Vietnamese query / English paraphrase fixture;
- exact OCR string fixture;
- no-evidence window;
- deterministic ordering.

### Metrics

- correct-window Recall@1/3/8;
- category breakdown, especially OCR/speech/temporal.

---

## VQA-06 — Frame-bound evidence contract

Priority: **P0**
Effort: **Medium**
Dependencies: VQA-03

### Objective

Keep each caption/OCR/ASR item attached to the frame/time that supports it.

### Conceptual model

```python
FrameEvidence(
    frame_id,
    timestamp_ms,
    caption,
    ocr_items,
    asr_segments,
    provenance,
)
```

Reuse current schemas if they already represent this sufficiently.

### Serialization to VLM

```text
Frame 0 | id=... | t=12.4s
Caption: ...
OCR: ...
Speech: ...

Frame 1 | id=... | t=15.8s
...
```

Do not flatten all caption strings separately from all OCR/ASR strings such
that temporal ownership disappears.

### Tests

- evidence remains attached through window -> answerer;
- ASR intervals are not assigned to invented frames;
- character/item budget pruning preserves the most relevant evidence rather
  than only earliest evidence.

---

## VQA-07 — Multi-frame answerer

Priority: **P0**
Effort: **Medium**
Dependencies: VQA-04, VQA-06

### Objective

Use the existing multi-image VQA capability for question types that require
multiple frames.

### Routing policy baseline

| Answer type       | Initial image budget |
| ----------------- | -------------------- |
| OCR / COLOR       | 1-2                  |
| COUNT             | 1-3                  |
| SPEECH            | 1-2 + ASR            |
| IDENTITY / ACTION | 2-4                  |
| GENERAL           | 2-4                  |
| TEMPORAL / CAUSAL | 4-8 bounded/ordered  |

These are experiment defaults, not immutable constants.

### Requirements

- capability-check the configured backend;
- selected frames must be chronological;
- evidence and frame IDs must be supplied consistently;
- deterministic single-frame fallback when multi-image is unavailable;
- provider may only return supplied frame IDs.

### Metrics

Run oracle-window answer evaluation:

```text
single-frame
vs
multi-frame
```

Break down by answer type.

### Definition of Done

Multi-frame mode improves the categories that require temporal/contextual
reasoning without imposing its full cost on simple questions.

---

## VQA-08 — NumPro-style explicit temporal prompt

Priority: **P0 for temporal subset**
Effort: **Small**
Dependencies: VQA-07

### Objective

Make temporal order explicit to Qwen.

### Input format

```text
[FRAME 0 | 12.4 sec]
<image>
...
[FRAME 1 | 15.8 sec]
<image>
...
```

Prompt states:

```text
The frames are in chronological order.
Use frame order/timestamps when the question requires temporal reasoning.
```

### Ablation

```text
multi-frame without explicit numbers/time
vs
multi-frame + frame number
vs
multi-frame + frame number + timestamp
```

### Metrics

Temporal/causal oracle-window answer accuracy.

---

## VQA-09 — Preserve event description in answer prompt

Priority: **P0**
Effort: **Small**
Dependencies: VQA-07

### Objective

Avoid losing referential context in questions such as:

```text
Event: man in red jacket talks to cashier
Question: what is he holding?
```

### Answer prompt

Must preserve:

```text
Event context: ...
Question: ...
```

### Metrics

Identity/reference/general VQA subset.

---

## VQA-10 — Answerability and confidence contract

Priority: **P0**
Effort: **Medium**
Dependencies: VQA-07

### Objective

Remove fake/default confidence semantics from fallback/ranking logic.

### Desired result concept

```text
answer
selected_frame_ids
answerable
provider confidence if meaningful
system confidence if separately calibrated
evidence IDs
warnings
```

### Rules

- do not label a constant `0.5` as meaningful model confidence;
- do not hard-code grounded/answerable true;
- provider confidence and system-ranking score are different concepts;
- fallback decisions must use real observable signals.

### Tests

- unanswerable provider response;
- malformed/invented frame ID rejected/degraded;
- no confidence returned;
- deterministic fallback.

---

# 10. M4 — VQA/KIS P1 quality and latency

## VQA-11 — Confidence/evidence-driven fallback

Priority: **P1**
Effort: **Medium**
Dependencies: VQA-10, VQA-05

Trigger bounded extra computation when one or more are true:

- low retrieval/video margin;
- low semantic-localizer margin;
- answerable=false;
- independent windows disagree;
- required modality evidence missing;
- current frame sample has poor temporal coverage.

Fallback options:

- sample more frames inside the same bounded window;
- test the next-ranked local window;
- expand with a separately configured larger-but-bounded context;
- second VLM pass.

Do not retry indefinitely.

Measure:

- accuracy gain on triggered queries;
- trigger rate;
- additional calls/query;
- p95 latency.

---

## VQA-12 — Contextual question/clue retrieval

Priority: **P1**
Effort: **Medium**
Dependencies: ARCH-01

### Objective

Avoid generic question branches such as:

```text
What color is the cup?
```

retrieving unrelated cups across the entire corpus.

### Compare

```text
event only
question only
event + question
rule-generated contextual clue
LLM rewritten contextual clue
```

Example:

```text
event: man and woman at a desk
question: what color is the cup?
contextual clue: cup near the man and woman at the desk
```

Promotion requires candidate/video recall gain.

---

## PERF-01 — Image/thumbnail decode cache

Priority: **P1**
Effort: **Medium**
Dependencies: EVAL-02

### Objective

Reduce repeated disk/PIL/decode cost in KIS reranking and VQA evidence display.

Cache keys must include relevant corpus/image variant versions.

Measure:

- hit rate;
- image-load p50/p95;
- memory usage;
- end-to-end effect.

---

## PERF-02 — Mixed-precision embedding benchmark

Priority: **P2**
Effort: **Small**
Dependencies: EVAL-03

Test SigLIP inference:

```text
fp32
bf16/fp16 when supported
```

Measure:

- embedding throughput;
- VRAM;
- retrieval recall/rank stability.

Do not change precision from config naming alone; verify actual model load dtype.

---

## PERF-03 — FAISS exact vs ANN/GPU benchmark

Priority: **P2**
Effort: **Medium**
Dependencies: corpus-scale profiling

Compare where useful:

```text
IndexFlatIP baseline
HNSW
IVF-Flat
IVF-PQ if memory pressure justifies it
FAISS GPU
```

Report:

- Recall@20/50/100/500 against exact baseline;
- p50/p95;
- index build time/size;
- memory.

Do not replace exact search if corpus scale does not justify it.

---

# 11. M5 — P2 research experiments

These tasks are intentionally after the P0/P1 pipeline is measurable.

## RES-01 — Multi-granularity shot/window captions

Priority: **P2**
Effort: **Large**

### Hypothesis

Single-frame captions lose action transitions and event context.

### Proposed index hierarchy

```text
frame caption
shot caption
15-30 second window caption
```

Online:

```text
query
 -> coarse shot/window retrieval
 -> candidate region
 -> frame-level refinement
```

Research grounding: ChatVTG/coarse-to-fine temporal grounding.

### Gate

Promote only if it improves event/temporal KIS or VQA window recall enough to
justify offline captioning/index cost.

---

## RES-02 — Learned question-aware frame selector

Priority: **P2**
Effort: **Large**

Only attempt after the training-free selector from VQA-04 is stable and there
is a labeled benchmark.

Compare:

```text
earliest frames
uniform temporal sample
BGE relevance + MMR
learned selector
```

Do not train without an evaluation set that measures evidence recall and
end-to-end VQA.

---

## RES-03 — Alternative temporal/video encoders

Priority: **P2**
Effort: **Large**

Candidates may include temporal grounding/video-language encoders such as
UniVTG/QD-DETR-family ideas or newer proven models available at experiment
 time.

### Rule

Do not replace SigLIP/BGE globally.

Test an alternative as an additional clip/window representation first.

Required comparison:

- candidate/video/window recall;
- latency;
- memory;
- offline preprocessing cost;
- gain after the V2 orchestration fixes.

---

# 12. KIS ablation matrix

Run each step separately.

| Run | Query policy   | Retrieval preserved after Qwen | Qwen gating   | Shot-aware NMS    |
| --- | -------------- | ------------------------------ | ------------- | ----------------- |
| K0  | current/static | no/current                     | no/current    | time-only/current |
| K1  | query-aware    | current                        | current       | current           |
| K2  | query-aware    | yes, second rank fusion        | current       | current           |
| K3  | query-aware    | yes                            | yes by intent | current           |
| K4  | query-aware    | yes                            | yes           | shot-aware        |

Primary comparison:

```text
K0 vs K2
```

This tests the central hypothesis that image-only reranking can erase useful
multimodal evidence.

Report every category separately.

---

# 13. VQA ablation matrix

| Run | Video aggregation | Window        | Frame selector      | Localizer       | Answerer                                    |
| --- | ----------------- | ------------- | ------------------- | --------------- | ------------------------------------------- |
| Q0  | current           | overlap merge | earliest/current    | lexical/current | 1 frame/current                             |
| Q1  | rank-compatible   | bounded peaks | current             | current         | current                                     |
| Q2  | rank-compatible   | bounded       | relevance/diversity | current         | current                                     |
| Q3  | rank-compatible   | bounded       | relevance/diversity | BGE semantic    | current                                     |
| Q4  | rank-compatible   | bounded       | relevance/diversity | BGE             | adaptive multi-frame                        |
| Q5  | rank-compatible   | bounded       | relevance/diversity | BGE             | multi-frame + numbered time + event context |

Do not implement Q5 in one giant PR.

Each row should be reproducible so the source of gain/regression is known.

---

# 14. Stage-based debugging decision tree

Use this before proposing a model replacement.

```text
Is correct video in top candidates?
 |
 +-- NO --> inspect query planning / modality routing / fusion / retrieval
 |
 +-- YES
      |
      v
 Is accepted temporal window retrieved?
      |
      +-- NO --> inspect video aggregation / peaks / windows / semantic localizer
      |
      +-- YES
           |
           v
 Is sufficient evidence in selected frame budget?
           |
           +-- NO --> inspect frame selector / evidence pruning
           |
           +-- YES
                |
                v
 Does VLM answer correctly with oracle evidence?
                |
                +-- NO --> prompt / multi-frame input / VLM capability/model
                |
                +-- YES --> joint ranking / confidence / normalization issue
```

---

# 15. Task dependency table

| ID      | Priority    | Dependency      | Main success criterion                  |
| ------- | ----------- | --------------- | --------------------------------------- |
| EVAL-01 | P0          | none            | frozen reproducible query set           |
| EVAL-02 | P0          | none            | per-stage latency/count tracing         |
| EVAL-03 | P0          | EVAL-01/02      | baseline metrics recorded               |
| ARCH-01 | P0          | EVAL preferred  | deterministic QueryPlan                 |
| RET-01  | P0          | ARCH-01         | runtime modality policy                 |
| RET-02  | P1          | EVAL, RET-01    | measured fusion weights                 |
| KIS-01  | P0          | RET-01          | category-aware retrieval gain           |
| KIS-02  | P0          | KIS-01          | preserve multimodal evidence after Qwen |
| KIS-03  | P1          | KIS-02, traces  | lower latency at same quality           |
| KIS-04  | P1          | KIS-02          | fewer false duplicate suppressions      |
| KIS-05  | P1/P2       | KIS-02          | evidence-aware rerank gain worth cost   |
| KIS-06  | P1          | ARCH/KIS-01     | ordered KIS subset improvement          |
| VQA-01  | P0          | baseline        | required modality changes rank          |
| VQA-02  | P0          | VQA-01          | better correct-video recall             |
| VQA-03  | P0          | VQA-02          | bounded windows + window recall         |
| VQA-04  | P0          | VQA-03          | frame evidence recall at fixed budget   |
| VQA-05  | P0          | VQA-03          | semantic window recall gain             |
| VQA-06  | P0          | VQA-03          | frame/timestamp evidence preserved      |
| VQA-07  | P0          | 04/06           | multi-frame gain where needed           |
| VQA-08  | P0 temporal | VQA-07          | temporal QA gain                        |
| VQA-09  | P0          | VQA-07          | reference/context QA gain               |
| VQA-10  | P0          | VQA-07          | meaningful answerability/confidence     |
| VQA-11  | P1          | 05/10           | bounded fallback quality/latency gain   |
| VQA-12  | P1          | ARCH-01         | contextual retrieval recall gain        |
| PERF-01 | P1          | tracing         | reduced decode/load latency             |
| PERF-02 | P2          | baseline        | same retrieval quality, less compute    |
| PERF-03 | P2          | scale profile   | ANN speedup at acceptable recall        |
| RES-01  | P2          | stable V2       | event/window retrieval gain             |
| RES-02  | P2          | labels + VQA-04 | learned selector beats training-free    |
| RES-03  | P2          | stable V2       | temporal encoder gain worth cost        |

---

# 16. Recommended PR/task grouping

Keep changes small enough to isolate regressions.

Recommended grouping:

```text
PR 1  EVAL-01 + metric fixtures
PR 2  EVAL-02 + tracing
PR 3  ARCH-01
PR 4  RET-01
PR 5  KIS-01
PR 6  KIS-02
PR 7  VQA-01 + focused tests
PR 8  VQA-02
PR 9  VQA-03
PR 10 VQA-04
PR 11 VQA-05
PR 12 VQA-06
PR 13 VQA-07 + VQA-09
PR 14 VQA-08 temporal prompt ablation
PR 15 VQA-10
```

P1 tasks can proceed after the relevant P0 track is stable.

Do not combine a new query planner, new fusion algorithm, new frame selector,
and new VLM prompt in the same PR.

---

# 17. Definition of Done for every optimization PR

Every task report should contain:

## Verified baseline behavior

- files/call sites inspected;
- current branch behavior;
- whether the audit hypothesis still exists.

## Implementation

- files created/modified;
- public interface changes;
- compatibility/fallback.

## Tests

- focused unit tests;
- integration/regression tests when relevant;
- no live model/corpus in unit tests.

## Measurement

For algorithm/optimization tasks:

- frozen query-set version;
- baseline config;
- experiment config;
- metrics before/after;
- p50/p95 or relevant latency;
- category breakdown;
- run directory / commit.

## Decision

Classify result as:

```text
KEEP
REJECT
NEEDS_MORE_DATA
```

Do not keep an optimization solely because the code is more sophisticated.

---

# 18. Immediate P0 work queue

If no teammate has already implemented these changes, the recommended immediate
queue is:

```text
1. EVAL-01  frozen KIS/VQA benchmark
2. EVAL-02  stage tracing
3. EVAL-03  baseline run
4. ARCH-01  QueryPlan
5. RET-01   query-conditioned modality policy

KIS track:
6. KIS-01   query-aware retrieval
7. KIS-02   evidence-preserving reranking

VQA track:
8. VQA-01   required-modality ranking fix
9. VQA-02   video aggregation rewrite
10. VQA-03  bounded temporal windows
11. VQA-04  relevance/diversity frame selector
12. VQA-05  BGE semantic localizer
13. VQA-06  frame-bound evidence
14. VQA-07  adaptive multi-frame answerer
15. VQA-09  event context in prompt
16. VQA-08  numbered/timestamped temporal prompt
17. VQA-10  answerability/confidence contract
```

The first implementation action for any item is always:

> inspect the active branch and verify that the issue still exists.

---

# 19. Final architecture acceptance criteria

V2 is ready to become the new baseline when all of the following are true:

### KIS

- query intent can affect retrieval policy;
- OCR/ASR evidence is not silently destroyed by image-only reranking;
- reranking has deterministic fallback;
- duplicate suppression respects shot/evidence distinctions;
- Top-100 remains canonical and deterministic;
- category metrics and latency are recorded.

### VQA

- required modality affects candidate ranking;
- video aggregation uses interpretable/comparable features;
- temporal windows are bounded;
- frame selection is relevance/evidence aware;
- semantic localizer handles multilingual paraphrase better than lexical-only
  baseline;
- evidence remains bound to frame/time;
- temporal/general questions can use ordered multi-frame VLM input;
- simple questions use a cheaper bounded input policy;
- event description is preserved in reasoning context;
- answerability/confidence are not fake constants;
- joint answer remains attached to canonical evidence;
- video/window/evidence/reasoning metrics are independently measurable.

### System

- no TRAKE internals were changed;
- shared contract compatibility is documented;
- no optimization claim lacks an experiment record;
- P0 architecture is complete before P2 model replacement is treated as the
  primary strategy.

---

# 20. One-sentence strategy

**Make retrieval query-aware, preserve the multimodal evidence that found each
candidate, localize bounded temporal evidence before reasoning, give the VLM
only the minimum sufficient structured frames, and measure every stage before
changing models.**
