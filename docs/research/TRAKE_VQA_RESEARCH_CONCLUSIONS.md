# TRAKE and VQA Research Conclusions

Date: 2026-07-30

Status: architecture research export. The 2025 competition contract is prior
evidence; unresolved 2026 rules must not be inferred from it.

## Constraints

- Return official video names and canonical integer `frame_idx` values.
- Preserve `frame_id -> video_id -> frame_idx`; never infer `frame_idx` from
  timestamps, FPS, filenames, or neighboring frames.
- Reuse pretrained components without training or fine-tuning.
- Target a 1--3 month competition cycle on an L40 or A6000.
- OCR and ASR are textual evidence; visual frames use the visual embedding
  space.

## TRAKE

### Task evidence

The 2025 contract describes an ordered event list `E1...EN`. Each submitted row
contains one video and exactly `N` chronological frame indices. A wrong video
receives zero row credit; for the correct video, credit is the fraction of
events whose selected frame falls in its accepted interval.

No surveyed paper implements this exact competition contract end to end. The
nearest research areas are video-corpus moment retrieval, temporal grounding,
ordered step localization, and sequence alignment.

### Surveyed foundations

- [TVR/XML](https://arxiv.org/abs/2001.09099) separates corpus retrieval from
  moment localization and explicitly combines video and subtitle evidence.
- [Moment-DETR](https://arxiv.org/abs/2107.09609) predicts query-conditioned
  temporal spans and saliency scores.
- [UniVTG](https://arxiv.org/abs/2307.16715) unifies moment retrieval,
  highlight detection, and video summarization, including zero-shot analysis.
- [Drop-DTW](https://arxiv.org/abs/2108.11996) performs robust monotonic
  alignment while permitting outlier elements to be dropped.
- [TFVTG](https://arxiv.org/abs/2408.16219) provides training-free temporal
  grounding using pretrained vision-language components.
- [TDGV](https://arxiv.org/abs/2407.12066) studies simultaneous grounding of a
  sequence of textual queries in one video.
- [CrossTask](https://arxiv.org/abs/1903.08225) demonstrates ordered step
  constraints for instructional-video alignment.

### Recommended TRAKE architecture

For every event `Ei`:

1. Retrieve frame or temporal-window candidates independently.
2. Fuse visual, caption, OCR, and ASR ranks using the TRAKE task
   configuration.
3. Group candidates by canonical `video_id`.
4. Score videos by event coverage, not only their best individual frame.
5. Jointly decode one candidate per event inside each video.

For video `v`, let `Ci(v)` be the candidates for event `Ei`. Decode:

```text
argmax(c1...cN)
    sum_i event_score(Ei, ci)
  - lambda_gap * sum_i gap_penalty(ci, c{i+1})
```

subject to:

```text
ci in Ci(v)
video(ci) = v
timestamp(c1) <= ... <= timestamp(cN)
```

Dynamic programming, Viterbi decoding, or bounded beam search can solve this
lattice. The decoder operates on existing canonical `frame_id` values. A
predicted window must select an existing canonical frame within that window
before submission materialization.

Because official ranking evaluates cutoffs `{1,5,20,50,100}`, the final
implementation should retain the best `B` valid paths across candidate videos,
not only one path.

### TRAKE progression

1. Event-wise keyframe retrieval, video event-coverage aggregation, and
   monotonic dynamic programming.
2. Multi-scale window embeddings and coarse-to-fine alignment.
3. Bounded training-free temporal reranking on only the top videos/windows.

Generative event decomposition is not part of the default path because the
input already provides an ordered event list.

## VQA

### Task evidence

The 2025 contract requires `<video_name>,<frame_idx>,<answer>`. Credit requires
the correct video, a canonical frame inside the accepted interval, and the
correct answer. Answers are bounded to 100 characters and may be Vietnamese or
English. The public rules mention both semantic and exact comparison, so both
forms must remain measurable until the 2026 scorer is confirmed.

### Surveyed foundations

- Temporal-grounding research treats evidence localization as an intermediate
  step for VideoQA rather than asking one model to search an entire corpus.
- Moment-DETR and UniVTG provide evidence for query-conditioned candidate
  localization.
- The configured
  [GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking)
  supports image-text inference and can answer one bounded question about a
  supplied frame.

### Approved VQA architecture

The user approved answering over multiple candidates:

```text
VQA-specific multimodal retrieval
    -> top-M canonical frames
    -> one answer per frame using image + caption/OCR/ASR evidence
    -> rerank (frame, answer, evidence) pairs
    -> ranked competition rows
```

The answer generator must not rewrite candidate identity. Each result retains
the exact input `frame_id`, materializes `video_id` and `frame_idx` through the
canonical mapping, and preserves the submitted answer string.

The current hosted `/v1/vqa` endpoint implements single-frame bounded
generation. The next VQA component needs bounded batching plus an explicit
answer-grounding scorer. Candidate count `M` and score composition remain
task-configuration decisions, not user-selectable runtime profiles.

### VQA scoring boundary

The reranker should consume:

- question;
- candidate image;
- candidate caption, OCR, and temporally aligned ASR evidence;
- generated answer;
- retrieval and multimodal relevance scores.

It may reorder answer-frame pairs but may not create a new frame identity. The
final score must remain separately inspectable from retrieval, fusion, and
answer-grounding scores.

## Conclusions

- TRAKE is a joint constrained sequence problem, not `N` independent KIS
  searches.
- VQA is retrieval plus grounded answering over several candidate frames, not
  top-1 frame answering.
- Both tasks reuse the four retrieval modalities, but require task-specific
  stages after fusion: monotonic temporal decoding for TRAKE and answer-pair
  reranking for VQA.
- No surveyed evidence justifies hard-coded optimal modality weights without
  task judgments. Any initial weights must be recorded as hypotheses.
