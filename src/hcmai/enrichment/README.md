# Enrichment — AI2-01 resumable frame captions

This component directory is the task-board home for offline frame enrichment.
Branch `feat/ai2-caption-baseline` introduces `CaptionConfig`,
`FrameCaptioner`, and `generate_captions` at the canonical
[`caption.py`](caption.py) ownership path. Its literal Task Board
implementation is complete and ready for engineering review, not merge.

## 1. Task identity

| Field | Value |
|---|---|
| Task ID | AI2-01 |
| Owner | Khầy |
| Workstream | Caption enrichment |
| Priority | P0 |
| Task-board status | Complete |
| Task | Implement resumable frame-caption baseline |
| Branch | `feat/ai2-caption-baseline` |
| Base | `main@9fbfaa2e0a2acb9b28e7c305528ccbadce34368f` |
| Implementation commit documented | `fd97e2ffa4571c943883bf3dc9bc97f28a17e19b` |
| Canonical source | [`src/hcmai/enrichment/caption.py`](caption.py) |
| Canonical test | [`tests/test_caption.py`](../../../tests/test_caption.py) |

The task board requires Florence-2 to be loaded once, concise captions to be
generated in configurable batches, and versioned status/error rows to resume
without regenerating valid completed frames. Its stale planned paths
`src/aic/enrichment/caption.py` maps by package root to the implemented
`src/hcmai/enrichment/caption.py`. The task-board-owned
[`scripts/generate_enrichment.py`](../../../scripts/generate_enrichment.py)
is a thin entry point delegating to the same reusable module CLI; it contains
no model or argument-parsing logic.

Task-board inputs are the DE-02 100-frame fixture, then canonical
`frames.parquet`, image paths, and enrichment configuration. Required outputs
are `frame_enrichment.parquet`, `manifest.json`, and `failures.json`.

## 2. Branch purpose

This branch owns a small, offline frame-caption job. It reads canonical frame
records, resolves their image paths, generates concise Florence-2 captions,
and writes one `FrameEnrichment` row per input frame in deterministic order.
It also owns per-frame failures, configuration evidence, and resume behavior.

It does not own frame extraction, schema governance, text/visual fusion,
embedding generation, retrieval, `SearchEngine`, API startup, full-corpus
execution, or acceptance of caption quality for downstream search.

## 3. Implemented

- [`CaptionConfig`](caption.py) configures checkpoint,
  revision, prompt, decoding, device, dtype/precision, image size, batch size,
  enrichment version, dataset version, and write interval.
- `FrameCaptioner` lazily loads one processor/model pair and reuses it across
  batches. A fully resumed job does not call or load the backend.
- `generate_captions` reads canonical Parquet records, preserves `frame_id`,
  resolves relative `image_path` values against an explicit dataset root,
  isolates missing/corrupt-image and model failures, and reconstructs exactly
  one final row per input.
- Completed rows require a non-empty caption; failed, incomplete, malformed,
  pending, or duplicated prior rows are retried. Parquet nulls are normalized
  before validation so valid completed rows are skipped rather than retried.
- A same-version resume fails fast when the model, revision, prompt, decoding,
  or other effective configuration differs from the retained manifest.
- Parquet, failure, and manifest checkpoints are written to sibling temporary
  files and replace their targets only after a successful write.
- The module CLI provides the task-board executable behavior:

  ```bash
  PYTHONPATH=src .venv/bin/python -m hcmai.enrichment.caption \
    --config <caption-config> \
    --frames data/aic_fixture/metadata/frames.parquet \
    --dataset-root data/aic_fixture \
    --output artifacts/enrichment/<enrichment-version>
  ```

- The assigned script entry point delegates to that same CLI:

  ```bash
  PYTHONPATH=src .venv/bin/python scripts/generate_enrichment.py --help
  ```

- The real fixture used native
  `florence-community/Florence-2-base-ft` at revision
  `0b03b6f15a4a211370fb204aee4e7dd48887ea37`, without remote code.

## 4. Not implemented or incomplete

- No retained, approved 30-frame semantic review proves caption correctness,
  diversity, or search value.
- The 100-frame artifact is a fixture result, not a validated full-corpus
  enrichment.
- No caption retrieval/fusion experiment or labelled quality metric exists.
- No shared component joins this artifact into embeddings or `SearchEngine`.
- The branch is based on an older `main`; current-main compatibility must be
  reviewed before merge.
- Detailed captions are intentionally not enabled; the task board requires a
  separate ablation before adding them.

## 5. Verification evidence

### Engineering evidence

| Evidence type | Command or artifact | Result | Proves | Does not prove |
|---|---|---|---|---|
| Offline tests | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_caption.py tests/test_schema.py` | 6 passed | Lifecycle, batching, failures, identities, schema, resume | Real caption accuracy |
| Fixture artifact | `artifacts/enrichment/florence2_native_base_ft_caption_v1/frame_enrichment.parquet` | 100 completed; 100 unique keys; 0 empty captions | Artifact and identity contract | Semantic usefulness |
| Resume artifact | Same directory, `manifest.json` | 100 skipped; 0 failed | Valid rows are not regenerated | Full-corpus reliability |
| Compatibility smoke | Temporary CPU evidence, not committed | 5 frames; model identity stable; load 30.20 s; generation 7.65 s | Native CPU compatibility and reuse | Production/GPU latency |

### Quality evidence

The artifact has non-empty captions, but no durable accepted review sheet or
labelled metric. Repeated and low-information fixture frames make non-empty
rate an unsafe quality proxy.

### Integration evidence

No repository consumer joins the caption Parquet into retrieval or search.
The branch is standalone.

## 6. Artifacts

Required local, Git-ignored layout:

```text
artifacts/enrichment/florence2_native_base_ft_caption_v1/
├── frame_enrichment.parquet
├── manifest.json
└── failures.json
```

All three files are currently present locally. They are fixture-only and are
not available after a fresh clone. The latest manifest is a resume run, so it
does not preserve the original 100-frame generation duration. No full-corpus
artifact or permanent manual-review artifact exists.

## 7. Dependencies and cross-team contracts

| Dependency | Owner/task | Path or symbol | Use | Readiness | Blocking? | Modified here? |
|---|---|---|---|---|---|---|
| Frame fixture | DE-02 | `data/aic_fixture/metadata/frames.parquet` | Frame IDs and image paths | Validated 100-frame local fixture | No for review | No |
| Enrichment contract | Tech Lead | [`FrameEnrichment`](../common/schemas/frame.py) | Output validation | Available | No | No |
| Artifact I/O | Shared | `hcmai.common.utils.io` | JSON/Parquet writing | Available | No | No |
| Enrichment fusion | Tech Lead / AI1 decision | No approved consumer exists | Downstream searchable evidence | Missing | Yes for downstream use | No |
| Search integration | Tech Lead / SWE | [`SearchEngine`](../search.py) | Future serving path | Not integrated | Yes for system acceptance | No |

All dependencies above are referenced dependencies; they are not owned or
modified by this branch.

## 8. Current quality status

- Task Board implementation status: **COMPLETE**.
- Engineering: **PASS** — the task-board artifact, lifecycle, failure, and
  resume contracts and assigned script entry point are verified.
- Practical caption quality: **PENDING MANUAL REVIEW** — captions are
  non-empty, but semantic accuracy and retrieval value have not passed an
  approved manual gate.
- Integration: **NOT INTEGRATED** — the artifact is not consumed by the
  search path.

## 9. Merge readiness

| Field | Decision |
|---|---|
| Merge target | Latest `main`, after compatibility review |
| Current readiness | **READY FOR REVIEW — NOT READY TO MERGE** |
| Blocking conditions | 30-frame semantic review; downstream fusion decision; current-main compatibility |
| Required approvals | AI2 owner for quality evidence; Tech Lead for integration/merge |
| Downstream usage | Experimental fixture evidence only |

Ready for review does not mean ready to merge.

## 10. Manual acceptance procedure

1. On this branch, open
   `artifacts/enrichment/florence2_native_base_ft_caption_v1/` and the input
   `data/aic_fixture/metadata/frames.parquet`.
2. Run a read-only review exporter or inspect the Parquet directly:

   ```bash
   PYTHONPATH=src .venv/bin/python - <<'PY'
   import pandas as pd
   frames = pd.read_parquet("data/aic_fixture/metadata/frames.parquet")
   caps = pd.read_parquet(
       "artifacts/enrichment/florence2_native_base_ft_caption_v1/"
       "frame_enrichment.parquet"
   )
   print(frames.merge(caps, on="frame_id")[["frame_id", "image_path", "caption"]])
   PY
   ```

3. Select 30 frames before judging outputs: all black/no-object frames plus
   people, vehicles, indoor/outdoor scenes, small objects, text-heavy scenes,
   blur, unusual angles, and visually similar adjacent frames.
4. Record:

   ```text
   frame_id | image_path | caption | main_subject_correct | action_correct
   scene_correct | hallucination | important_omission | search_value
   verdict | notes
   ```

5. Use verdicts `Correct`, `Mostly correct`, `Partial`, `Incorrect`,
   `Hallucinated`, `Too vague`, or `Correct empty / not applicable`.
6. PASS only if at least 80% are `Correct`/`Mostly correct`, at least 80% add
   search value, no invented named entity or visible text is accepted, no more
   than 10% are incorrect, and black/adjacent-frame behavior is explicitly
   judged. Any systematic hallucination or misleading searchable evidence is
   a downstream-blocking FAIL.
7. Preserve the reviewed CSV/JSON and selection policy under an ignored
   `runs/caption_manual_review/` directory.

## 11. Known risks

- Generic captions may satisfy non-empty checks while adding little retrieval
  discrimination.
- Repetitive adjacent frames can inflate apparent consistency.
- Florence may invent objects, actions, or visible text.
- CPU timing is compatibility evidence, not production throughput.
- The stale branch base can conflict with newer shared schemas or data code.

## 12. Next actions

1. **AI2 owner:** complete and retain the 30-frame semantic review.
2. **Tech Lead:** approve the quality gate and decide the enrichment-fusion
   contract/owner.
3. **AI2 + AI1:** run a frozen-candidate caption retrieval/fusion experiment
   only after labels are available.
4. **Tech Lead:** review current-main compatibility and merge readiness.
5. **DE/AI2:** schedule full-corpus enrichment only after quality promotion.
