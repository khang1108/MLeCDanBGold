# Unified Temporal Alignment Migration Gate

**Date:** 2026-08-30
**Status:** STRUCTURAL MIGRATION USER-AUTHORIZED; COMPETITION CUT-OVER BLOCKED PENDING EVALUATION

## Authority review

The local organizer document, [`Thông tin vòng Sơ tuyển AIC 2026`](../Thông tin vòng Sơ tuyển AIC 2026.md), is the current competition authority available in this repository.

- **SOURCE:** Textual KIS receives one complete natural-language event
  description and requires one frame within the accepted interval of the
  correct video. The document does not describe a progressive request/session
  protocol.
- **SOURCE:** TRAKE requires one retrieved video and one semantic frame for
  every stage in its event sequence. A wrong video scores zero; otherwise each
  event frame is judged against its accepted interval.
- **SOURCE:** The organizer examples are chronologically ordered events, but
  the available document does not explicitly state whether a submitted frame
  may be reused for multiple events. The strict no-reuse behavior of the
  current monotonic DP is therefore **PROPOSED**, not an organizer requirement.
- **SOURCE:** The local runtime preserves competition `frame_idx` separately
  from internal `frame_id`. The organizer document calls the submitted
  coordinate `frame_id`; the repository's canonical identity rule remains the
  implementation authority until an updated scorer resolves this terminology.

## Frozen-baseline availability

No versioned development-query manifest, ground-truth intervals, or prior
KIS/TRAKE prediction report is present under the tracked repository paths.
The local `artifacts/` directory contains frame/enrichment/index bundles only;
it is intentionally ignored and contains no `evaluation/` report. A measured
old-versus-new comparison cannot be produced from the current workspace.

## Required evaluation record before cut-over

Create the following local, ignored run directory when judgments are available:

```text
artifacts/evaluation/temporal_migration/
├── manifest.json          # query-set ID, source checksum, query IDs only
├── current_baseline.json  # current KIS/TRAKE outputs and metrics
├── unified_baseline.json  # proposed outputs and metrics
└── summary.md             # decision, failures, versions, and latency
```

Each report must record:

- dataset/query-set version and ground-truth source checksum;
- model checkpoint, index/artifact version, config checksum, and code revision;
- the official R-Score/Final Score when labels are available, or an explicitly
  labelled proxy when they are not;
- canonical `video_id`, `frame_id`, `frame_idx`, and `timestamp_ms` outputs;
- mean candidate-video count, aligned-path span, and P50/P95 latency;
- representative failures, including events that would require frame reuse.

## Decision

Tasks 1–5 added characterization, task-agnostic contracts, deterministic
planning, filter-aware scoring, and a duplicate pure DP module without
changing KIS execution. The user subsequently explicitly authorized Tasks
6–12, including the structural deletion of the prior progressive/scene path.
That authorization permits repository cleanup; it is not evidence of a
measured competition cut-over. Do not describe the new KIS/TRAKE behavior as
release-accepted or improved until the evaluation record exists and the
resulting trade-off is explicitly accepted.
