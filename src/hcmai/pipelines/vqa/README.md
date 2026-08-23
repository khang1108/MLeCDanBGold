# VQA pipeline

`pipeline.py` is the task entry point. It keeps scene localization separate
from question answering:

```text
event hints -> temporal scene localization -> bounded evidence
question    -> question-conditioned evidence selection -> VLM answer
                                                      -> ranking -> submission
```

## Package responsibilities

- `domain/`: VQA-private value objects and narrow ports for retrieval, data,
  and inference dependencies.
- `query/`: deterministic interpretation of the event/question pair and
  conservative answer normalization.
- `reasoning/`: chronological OCR/caption/ASR evidence construction and
  bounded multi-frame VQA inference, including temporal neighbor expansion
  for a bounded retry.
- `output/`: grounded-answer ranking and canonical submission materialization.

Scene localization is always provided by the shared `TemporalEvidenceCore`.
The VQA head does not silently fall back to unordered frame-to-video-window
localization when that dependency is unavailable.

The orchestration package owns dispatch and dependency wiring only. Import
`VQAPipeline` from `hcmai.pipelines.vqa.pipeline`.
