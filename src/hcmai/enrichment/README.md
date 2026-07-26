# Enrichment — AI2-01 captions and AI2-02 OCR

This package owns the two independent offline frame-enrichment components
implemented by AI2: resumable Florence-2 captions in
[`caption.py`](caption.py) and optional OCR evidence in [`ocr.py`](ocr.py).
The integration branch combines the canonical task branches without wiring
either artifact into embeddings, retrieval, reranking, `SearchEngine`, or an
application lifecycle.

Both engineering components are complete under their literal Task Board
contracts. That does not make their generated text production-ready:
caption semantics still need an approved manual review, and evaluated OCR
quality failed the unrestricted downstream-use gate.

## Task identity

| Task | Canonical source | Test | Engineering | Quality | Integration |
|---|---|---|---|---|---|
| AI2-01 caption enrichment | [`caption.py`](caption.py) | [`tests/test_caption.py`](../../../tests/test_caption.py) | Complete | Unverified | Not integrated |
| AI2-02 optional OCR | [`ocr.py`](ocr.py) | [`tests/test_ocr.py`](../../../tests/test_ocr.py) | Complete | Failed | Disabled |

The Task Board's stale `src/aic/enrichment/` root maps directly to the
repository's `src/hcmai/enrichment/` root. The AI2-01 assigned entry point is
[`scripts/generate_enrichment.py`](../../../scripts/generate_enrichment.py).

## Caption enrichment

`CaptionConfig`, `FrameCaptioner`, and `generate_captions` implement a bounded
offline job:

- configure checkpoint, revision, prompt, decoding, device, dtype, image size,
  batch size, enrichment version, dataset version, and write interval;
- lazily load and reuse one processor/model pair;
- preserve canonical `frame_id` ordering;
- write one validated `FrameEnrichment` row per input frame;
- isolate missing/corrupt-image and model failures with bounded errors;
- retry failed, pending, malformed, empty-caption, or duplicated prior rows;
- skip exactly one valid completed row on resume;
- write `frame_enrichment.parquet`, `manifest.json`, and `failures.json`.

The module and assigned wrapper share one CLI implementation:

```bash
PYTHONPATH=src .venv/bin/python -m hcmai.enrichment.caption \
  --config <caption-config> \
  --frames <frames.parquet> \
  --output artifacts/enrichment/<caption-version>

PYTHONPATH=src .venv/bin/python scripts/generate_enrichment.py --help
```

The retained fixture evidence used
`florence-community/Florence-2-base-ft` revision
`0b03b6f15a4a211370fb204aee4e7dd48887ea37`: 100 completed rows,
100 unique composite keys, zero empty captions, zero failures, and a fully
resumed pass that skipped all 100 rows.

This proves the artifact and lifecycle contracts, not caption correctness.
No durable approved semantic review, labelled retrieval-value metric, or
full-corpus caption artifact exists.

## OCR enrichment

`OCRConfig`, `OCRResult`, `OCREngine`, and `generate_ocr` implement a separate
optional artifact path:

- `enabled=False` avoids engine construction, image decoding, and OCR calls;
- native Florence `<OCR>` loading is lazy and reused;
- Unicode NFC normalization collapses whitespace while preserving case,
  Vietnamese diacritics, numbers, and punctuation;
- a successful no-text result uses `ocr_text=None` and is not a failure;
- raw output and finite confidence are retained when the backend provides
  meaningful values; confidence is never fabricated;
- missing/corrupt images and malformed/backend failures preserve `frame_id`
  and write bounded failure details;
- caption, detailed-caption, and ASR fields remain unused;
- completed rows resume while failed, malformed, missing, or duplicated rows
  are retried;
- `frame_enrichment.parquet`, `ocr_report.json`, `manifest.json`, and
  `failures.json` are written independently from caption artifacts.

The retained Florence OCR fixture used the same pinned revision: 100 completed
rows, 100 unique keys, 90 with text, 10 canonical empty results, zero failures,
and 100 skipped rows on resume. Its frozen 20-sample visual review found
18 partial results, one poor result, one correct empty result, and four
suspected hallucinations. Florence CER was `0.5746`, important-number accuracy
was `57.3%`, and accented-token exact recall was `0/61`.

Therefore:

```text
AI2-02 engineering: complete
AI2-02 practical OCR quality: failed
AI2-02 downstream usage: disabled
Production OCR backend selected: no
```

The experimental branch `feat/ai2-ocr-vietnamese-backend` is intentionally
excluded from this integration branch. Its PaddleOCR and Paddle-detector plus
VietOCR adapters are negative ablation evidence, not approved dependencies or
production implementations.

## Verification evidence

### Engineering

| Component | Command or evidence | Result | Does not prove |
|---|---|---|---|
| Caption tests | `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_caption.py tests/test_schema.py` | Lifecycle, batching, failures, IDs, schema, resume | Caption accuracy |
| Caption fixture | `artifacts/enrichment/florence2_native_base_ft_caption_v1/` | 100 complete and resumable | Full-corpus reliability |
| OCR tests | `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_ocr.py tests/test_schema.py` | Disablement, normalization, failures, IDs, report, resume | OCR accuracy |
| OCR fixture | `artifacts/enrichment/florence2_ocr_v1/` | 100 complete and resumable | Safe downstream text |

Generated evidence under `artifacts/` is Git-ignored and does not exist after
a fresh clone. No model files are committed.

### Quality

- Caption text is structurally valid and non-empty, but semantic accuracy,
  diversity, hallucination rate, and searchable value are unverified.
- OCR coverage is not OCR accuracy. The reviewed Florence output is unsafe for
  unrestricted Vietnamese text fusion.

### Integration

No current package consumer joins caption or OCR Parquet rows into embeddings,
retrieval, reranking, `SearchEngine`, backend startup, or HTTP APIs. This
aggregation does not add that integration.

## Dependencies and ownership

| Dependency | Owner/task | Use | Modified here? |
|---|---|---|---|
| Frame fixture and canonical IDs | DE-02 | Input IDs and image paths | No |
| `FrameEnrichment` | Tech Lead shared schema | Canonical output validation | No |
| Artifact I/O | Shared utilities | JSON and Parquet writing | No |
| Enrichment fusion/indexing | Tech Lead / AI1 | Future searchable evidence | No |
| Search serving/startup | Tech Lead / SWE | Future system integration | No |

These are referenced dependencies, not owned by the enrichment package.

## Manual acceptance

### Caption

Review at least 30 diverse frames containing black/no-object frames, people,
vehicles, indoor/outdoor scenes, small objects, text-heavy scenes, motion
blur, unusual angles, and similar adjacent frames. Record main subject,
action, scene, hallucination, important omission, search value, verdict, and
notes. Repeated generic or invented descriptions block downstream promotion.

### OCR

Use the frozen 20 samples and compare the full-resolution source with visible
ground truth, OCR regions, raw and normalized text, line confidence, numbers,
Vietnamese diacritics, missed main regions, and hallucinations. High confidence
must never substitute for visual correctness. Current reviewed results fail
this downstream gate.

## Known risks

- Florence can invent objects, actions, scene text, or OCR strings.
- Repetitive fixture frames can conceal low semantic diversity.
- Ignored local artifacts are not permanent reproducibility evidence.
- Enabling noisy OCR would contaminate embeddings and retrieval.
- Neither task branch was originally based on the current `main`; combined
  regression and schema compatibility must be reviewed before promotion.

## Next actions

1. AI2 owner: complete and retain caption semantic review before promotion.
2. Tech Lead: keep OCR disabled or approve a narrowly scoped, evidence-backed
   use policy.
3. Tech Lead / AI1: define enrichment fusion only after quality approval.
4. SWE / Tech Lead: integrate approved components during application startup;
   this package must not construct application-wide models itself.
