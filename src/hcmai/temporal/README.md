1

# Shared temporal evidence and alignment facade

All task adapters create a validated `TemporalQueryPlan` and select one
explicit alignment mode:

```text
progressive_scene
  -> ProgressiveEvidenceProvider
  -> ProgressiveSceneAligner
  -> SceneCandidate[] for KIS/VQA

ordered_path
  -> DenseOrderedEvidenceProvider
  -> MonotonicOrderedPathAligner
  -> OrderedPathCandidate[] for TRAKE
```

The ports keep sparse progressive evidence separate from dense event/frame
matrices. TRAKE remains stateless and keeps the existing monotonic DP; the
shared adapter resolves its output through canonical `FrameRecord` values.

For KIS/VQA, the facade processes cumulative hint snapshots transactionally:

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
application composition time for KIS/VQA. The temporal path is the configured
default. Dense provider failure never falls back to unordered scene alignment.
