# Enrichment — AI2-02 optional independent OCR

This directory owns offline enrichment components. Branch
`feat/ai2-ocr-enrichment` adds the public `OCRConfig`, `OCRResult`,
`OCREngine`, and `generate_ocr` interfaces in [`ocr.py`](ocr.py). Its generic
pipeline is ready for engineering review, but backend quality and downstream
integration are not ready to merge.

## 1. Task identity

| Field | Value |
|---|---|
| Task ID | AI2-02 |
| Owner | Khầy |
| Workstream | OCR enrichment |
| Priority | P1 |
| Task-board status | In Progress (50%) |
| Task | Implement optional OCR evidence channel |
| Branch | `feat/ai2-ocr-enrichment` |
| Base | `main@47ebe06492a917749d7c16523b484df5be5a568f` |
| Implementation commit documented | `ad59ed2cb7b5977587f4adad15c09a93d071385e` |
| Canonical source | [`src/hcmai/enrichment/ocr.py`](ocr.py) |
| Canonical test | [`tests/test_ocr.py`](../../../tests/test_ocr.py) |

The task board requires configured OCR to run independently from captions,
normalization that retains useful raw/confidence evidence, Vietnamese
diacritics/numbers/signs/subtitle checks, and a coverage/error report. Inputs
are the frame fixture, OCR model/config, and enrichment artifact. Outputs are
OCR fields in `frame_enrichment.parquet` and `ocr_report.json`. Acceptance
requires disablement, canonical empty text, at least 20 reviewed samples, and
unchanged frame identifiers.

The stale task-board path `src/aic/enrichment/ocr.py` maps directly to
`src/hcmai/enrichment/ocr.py`. ASR remains stretch work and is not included.

## 2. Branch purpose

This branch owns a generic, optional, resumable OCR artifact pipeline and one
operational Florence-2 OCR backend. It reads canonical frames, decodes only
their images, normalizes searchable OCR text, writes canonical
`FrameEnrichment` rows, and records coverage, raw evidence, confidence support,
failures, and resume counts.

It does not generate captions, run ASR, retrieve frames, fuse OCR into search,
or assert that a backend is accurate because it completed successfully. OCR
backend quality and downstream promotion are separate decisions.

## 3. Implemented

- [`OCRConfig`, `OCRResult`, `OCREngine`, and `generate_ocr`](ocr.py)
  provide a narrow injectable backend boundary and one artifact path.
- `enabled=False` avoids engine construction, image decoding, and OCR calls.
- Valid completed rows resume by frame/version/model; failed, malformed,
  missing, or duplicated rows are retried without duplicate composite keys.
- Unicode NFC normalization collapses whitespace while preserving Vietnamese
  diacritics, case, numbers, and punctuation.
- A successful no-text result uses `ocr_text=None`; it is not a failure.
- Missing/corrupt images and malformed/backend failures retain `frame_id` and
  bounded errors.
- Caption, detailed-caption, and ASR fields remain unused.
- `ocr_report.json`, `manifest.json`, `failures.json`, and Parquet are written
  independently of AI2-01 caption artifacts.
- The Florence backend uses native
  `florence-community/Florence-2-base-ft` revision
  `0b03b6f15a4a211370fb204aee4e7dd48887ea37`.

## 4. Not implemented or incomplete

- Florence OCR is not semantically acceptable for unrestricted Vietnamese
  text fusion. The frozen review found weak diacritics, omissions, and
  suspected hallucinations.
- No tested backend has been approved or promoted as the production OCR
  backend.
- The optional channel is not wired into embeddings, reranking, or search.
- No full-corpus OCR artifact exists.
- Florence does not expose calibrated OCR confidence; the implementation does
  not fabricate it.
- No explicit Tech Lead approval exists to merge only the disabled generic
  infrastructure while backend quality remains rejected.

## 5. Verification evidence

### Engineering evidence

| Evidence type | Command or artifact | Result | Proves | Does not prove |
|---|---|---|---|---|
| Offline tests | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_ocr.py tests/test_schema.py` | 11 passed | Disablement, batching, normalization, failures, resume, IDs, schema | OCR accuracy |
| Fixture artifact | `artifacts/enrichment/florence2_ocr_v1/frame_enrichment.parquet` | 100 completed, 100 unique; 90 text, 10 empty | Artifact contract and operational run | Text correctness |
| Resume artifact | Same directory, `manifest.json` | 100 skipped; 0 failed | Backend-free completed resume | General corpus reliability |
| Manual review | Same directory, `ocr_report.json` | 20 viewed; 18 partial, 1 poor, 1 correct empty; 4 suspected hallucinations | Known Florence quality limitations | Backend acceptance |

### Quality evidence

The local frozen-20 ablation evidence reports Florence CER `0.5746`, number
token accuracy `0.5733`, accented-token exact recall `0/61`, and correct
no-text behavior on `1/2`. Coverage (`90%`) is not OCR accuracy. This evidence
rejects Florence for unrestricted downstream text use.

### Integration evidence

No current source consumer joins OCR output into captions, embeddings,
reranking, or `SearchEngine`. The noisy artifact has therefore not
contaminated those components.

## 6. Artifacts

Current local, ignored Florence artifact:

```text
artifacts/enrichment/florence2_ocr_v1/
├── frame_enrichment.parquet
├── ocr_report.json
├── manifest.json
└── failures.json
```

It contains a 100-frame fixture run and resume evidence. The 20-sample review
is embedded in the report, but its contact-sheet path was temporary and will
not exist in another clone. Generated artifacts are not committed. There is
no approved full-corpus artifact.

Experimental Paddle/VietOCR source and comparison evidence exist only on the
child branch `feat/ai2-ocr-vietnamese-backend`, not on this branch.

## 7. Dependencies and cross-team contracts

| Dependency | Owner/task | Path or symbol | Use | Readiness | Blocking? | Modified here? |
|---|---|---|---|---|---|---|
| Frame fixture | DE-02 | `data/aic_fixture/metadata/frames.parquet` | IDs and image paths | Validated locally | No for engineering review | No |
| Enrichment format | AI2-01 convention | Expected from `feat/ai2-caption-baseline` | Version/manifest/failure semantics | Reused by behavior, no code import | No | No |
| Shared schema | Tech Lead | [`FrameEnrichment`](../common/schemas/frame.py) | Canonical OCR row | Available | No | No |
| Future fusion/indexing | Tech Lead / AI1 | No approved OCR consumer exists | Searchable evidence | Missing | Yes for downstream use | No |
| Search serving | Tech Lead / SWE | [`SearchEngine`](../search.py) | Future integration | Not integrated | Yes for system acceptance | No |

These are referenced dependencies, not owned or modified by this branch.

## 8. Current quality status

- Engineering: **PASS** — the independent optional artifact pipeline, report,
  failure handling, and resume contract are verified.
- Quality: **FAIL** — the Florence baseline is operational but unsafe for
  unrestricted Vietnamese OCR use.
- Integration: **PENDING** — intentionally not consumed downstream.

## 9. Merge readiness

| Field | Decision |
|---|---|
| Merge target | `main` only after explicit scope/quality decision |
| Current readiness | **EXPERIMENTAL — DO NOT MERGE** |
| Blocking conditions | No approved backend; no downstream safety policy; no Tech Lead approval for infrastructure-only merge |
| Required approvals | AI2 quality review and Tech Lead backend/fusion decision |
| Downstream usage | **Disabled**; do not fuse current OCR text |

Ready for engineering review does not mean ready to merge.

## 10. Manual acceptance procedure

1. Use exactly the 20 frame IDs recorded at
   `artifacts/enrichment/florence2_ocr_v1/ocr_report.json` under
   `manual_review.samples[*].frame_id` (or recover the frozen list from the
   retained ablation evidence). Do not replace difficult samples.
2. View every original at full source resolution. Record:

   ```text
   frame_id | image_path | ground_truth | ocr_text | boxes
   line_confidences | numbers_correct | diacritics_correct
   major_region_detected | hallucination | safe_for_downstream
   verdict | notes
   ```

3. Transcribe only readable principal text and important numbers. Mark tiny or
   unreadable regions as ignored before viewing OCR output.
4. Inspect high-confidence wrong strings explicitly; confidence is never an
   acceptance substitute.
5. Compare raw backend output, normalized text, boxes, and the source image.
   Assign `Safe for downstream`, `Unsafe/noisy`, `Numeric-only value`, or
   `No readable text`.
6. Recommended unrestricted-use PASS gate: correct no-text `100%`, zero
   high-confidence hallucinations, major-region recall at least `90%`, number
   accuracy at least `95%`, accented-token recall at least `80%`, and CER no
   greater than `0.20` on readable text. Numeric-only promotion still requires
   at least `95%` important-number accuracy.
7. Any invented principal text, systematic accent loss, or misleading
   high-confidence output is a downstream-blocking FAIL.
8. Preserve ground truth, predictions, metrics, and manual verdicts under an
   ignored `runs/ocr_manual_acceptance/` directory.

The current reviewed Florence result does not pass this procedure.

## 11. Known risks

- Non-empty OCR and high coverage can hide incorrect strings.
- Florence can hallucinate text and lacks calibrated confidence.
- Broadcast overlays favor numbers and may mask poor Vietnamese recognition.
- Keeping raw output in reports can increase artifact size.
- Future fusion could degrade retrieval if enabled before a precision gate.

## 12. Next actions

1. **Tech Lead:** decide whether disabled generic infrastructure may merge
   independently of backend promotion.
2. **AI2 owner:** retain a reproducible frozen-20 review package with images,
   ground truth, boxes, predictions, and verdicts.
3. **AI2 + Tech Lead:** approve an OCR backend or keep the feature disabled.
4. **AI1 / search owner:** design any fusion only after quality promotion and
   measure the identical retrieval candidate set.
5. **DE:** provide full-corpus frames only after the backend passes the gate.
