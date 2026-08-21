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
  bounded multi-frame VQA inference.
- `output/`: grounded-answer ranking and canonical submission materialization.
- `legacy_localization/`: the fallback frame-to-video-to-window localization
  path. Application startup selects it only when
  `search.progressive.architecture: legacy`; do not add new localization
  behavior here.

The orchestration package owns dispatch and dependency wiring only. Its former
`workflows.vqa` module remains a compatibility import; new code imports
`VQAPipeline` from `hcmai.pipelines.vqa.pipeline`.
