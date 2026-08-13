# Shared temporal evidence core

The core processes cumulative KIS/VQA hint snapshots transactionally:

```text
snapshot delta
  -> global and candidate-local retrieval
  -> explicit single-video backfill for UNKNOWN pairs
  -> multi-hint video scoring and bounded candidate state
  -> bounded scene assembly
  -> normalized scene scoring
```

Important invariants:

- absence from a pooled Top-K result remains `UNKNOWN`;
- only a dedicated single-video search may produce evaluated-no-match;
- rescued videos are backfilled before candidate pruning;
- canonical frames are deduplicated before applying the Top-M budget;
- committed evidence is restricted to the active candidate pool;
- scenes obey both `scene_max_gap_ms` and `scene_max_span_ms`;
- relation components are omitted when they cannot be evaluated;
- task type, base filters, and VQA question fingerprint are frozen per
  progressive session.

`progressive.architecture` explicitly selects `temporal` or `legacy` at
application composition time. The temporal path is the configured default.
