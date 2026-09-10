# HCMAI 2026 Technical Report

This directory contains the repository-grounded LaTeX source for the HCMAI 2026 multimodal video-retrieval technical report.

## Publication contract

- Language: English.
- Format: clean, two-column A4, pdfLaTeX.
- Page limit: the compact title, abstract, and all main-body sections must end on or before page 4.
- The first four pages contain prose and equations only. All figures and tables are restricted to the appendix by a LaTeX build guard.
- References and the comprehensive appendix are outside the four-page body limit.
- Quantitative retrieval evaluation is intentionally excluded.
- Replace every bracketed value in `metadata.tex` before official submission.

## Build and validate

From this directory:

```bash
make statistics  # stream artifacts into generated JSON/TeX (requires PyArrow)
make pdf         # compile build/main.pdf from checked-in generated data
make check       # compile, enforce page/float rules, resolve references, verify inputs
make clean
```

`make pdf` deliberately does not depend on `statistics`: the report remains
buildable when the large local artifact tree is unavailable. To regenerate from
a different artifact location or Python environment, use for example:

```bash
make statistics ARTIFACTS_ROOT=/path/to/artifacts PYTHON=/path/to/python
```

The generator reads only the finalized Parquet files listed in
`tools/summarize_artifacts.py`; it excludes duplicate intermediate shards under
`artifacts/batches/`. Each file is scanned one lightweight identity column at a
time with `ParquetFile.iter_batches`, an 8,192-row batch limit, and
`use_threads=False`. It never materializes a complete Parquet table. Streamed
counts are checked against every Parquet footer, frame-aligned artifacts are
checked against the canonical frame total, and frame/video totals are checked
against `frame_store/manifest.json`. Complete results are atomically written to
`generated/artifact-statistics.json` and `generated/artifact-statistics.tex`.

`make check` requires four 1440×900 (or at least 1200×700) live application captures:

```text
figures/screenshots/kis-results.png
figures/screenshots/trake-results.png
figures/screenshots/video-inspector.png
figures/screenshots/collaborative-workspace.png
```

The LaTeX appendix overlays numbered callouts, so screenshots should remain unannotated PNGs. Do not include credentials, private URLs, browser chrome, or unrelated personal data.

## Source layout

```text
main.tex                       document entry point and page-limit guard
metadata.tex                   submission placeholders
preamble.tex                   typography, palette, diagram and callout styles
sections/main-body.tex         the four-page text/equation body
appendices/appendix.tex        TikZ diagrams, data chart, tables, screenshots, traceability
generated/artifact-statistics.* checked-in streamed statistics consumed by LaTeX
references.bib                 primary papers, model cards, official documentation
tools/summarize_artifacts.py   bounded-memory Parquet statistics generator
tools/check_report.py          publication-constraint and generated-input checks
figures/screenshots/           live UI captures
build/                         generated files
```

## Authority and scope

The report documents the current implementation and the corpus prepared from competition videos with one-frame-per-second C++/FFmpeg sampling. Claims were checked in this order:

1. current artifact manifests (`artifacts/frame_store/manifest.json`, index metadata, and `artifacts/reports/finalize_report.json`);
2. current implementation and typed contracts under `src/hcmai/`, `offline/`, `llm/`, and `frontend/src/`;
3. checked-in configuration under `configs/`;
4. maintained runbooks and README files.

The report presents the current one-frame-per-second C++/FFmpeg self-extraction pipeline as the canonical preparation route. Top-K reciprocal-rank fusion is kept distinct from the full-corpus adaptive fusion used by KIS/TRAKE temporal alignment. The proposed VLM reranking UI is not reported as an active feature.

Known release caveats are stated in the body and appendix: the local visual/context index metadata omit some provenance fields, and the promoted `build_report.json` is absent from this checkout. Regenerate or reconcile those artifacts before claiming an immutable official bundle.

## Screenshot capture states

Use a 1440×900 viewport and the running React/FastAPI application:

1. **KIS:** query and controls visible, populated result cards, latency summary, and one expanded alignment.
2. **TRAKE:** sequential `E1..En` input and at least one complete same-video path with the path-level submission control.
3. **Video inspector:** result opened at its timestamp, custom timeline visible (preferably with hover thumbnail), and evidence metadata visible.
4. **Collaborative Workspace:** query history/replay, manual video opening, and shared submission-file tree. A submission picker/editor may be open if it does not obscure all three regions.

If source video is unavailable locally, do not fabricate a playback claim. Obtain an authorized source video or clearly use a documented representative fixture; the final screenshot and caption must match what was actually exercised.

The committed captures use a temporary, identity-preserving fixture containing 21 videos and 3,807 canonical frames selected from the full one-frame-per-second corpus. The capture backend loaded the corresponding original SigLIP2 vectors and caption/OCR/object evidence with visual dense retrieval enabled. Since the original `L30_V060` source video was unavailable locally, the inspector capture uses a 148-second playback proxy assembled from that video's canonical 1-FPS frames solely to exercise the real range-streaming and timeline path. The appendix discloses these constraints; full-corpus counts come from the repository artifacts, not this fixture.
