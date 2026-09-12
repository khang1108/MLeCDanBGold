# Retrieval Baselines

This directory contains research-only command-line controls for multi-event video retrieval. The commands reuse the configured HCMAI indexes and full-corpus temporal evidence scorer, but they do not alter the production API or search workflow.

## Readiness

| Method | Identifier | Status | Purpose |
|---|---|---:|---|
| Global query | `global_query` | Runnable | One representation for the complete query; max frame score per video |
| Independent events | `independent_events` | Runnable | Per-event evidence without temporal order or distinct-frame constraints |
| Strict unary monotonic DP | `dante_style_unary_dp` | Runnable | Full-frame ordered unary alignment using the repository's current decoder |
| Unary candidate lattice | `unary_candidate_lattice` | Runnable | Strict ordered alignment over a sparse top-$K$ frame lattice |
| Unary bounded order | `unary_bounded_order` | Runnable | Critical control for order relaxation without transition features |
| Transition-aware strict order | `transition_aware_strict` | Deferred proposed method | Requires a defined transition scorer and labeled train/validation data |
| Transition-aware bounded order | `transition_aware_bounded_order` | Deferred proposed method | Requires the same transition scorer plus a validated order prior |
| Modern external model | Not selected | Deferred reproduction | Requires an exact repository, immutable checkpoint, license, preprocessing contract, and compatible evaluation protocol |

The DANTE-style control reproduces the **unary monotonic objective**, not DANTE's complete system, preprocessing, encoder stack, or reported result.

## Scientific controls

Every runnable method follows the same experiment rules:

- score every video in the configured corpus;
- return no more than one ranked result per video;
- never read ground truth during retrieval or insert a target into candidates;
- use deterministic query parsing and no live query rewriting;
- default to Dense evidence enabled and BM25 disabled;
- use the same unary evidence scorer for decoder comparisons;
- serialize event-based frame assignments in original query-event order;
- record inferred chronological event order in a separate nullable field;
- write the method name, numerical options, index metadata, and stage timings into a versioned JSON envelope.

BM25 remains available as an explicit ablation through `--use-bm25`, but it is off by default because the current title index has a documented lexical-leak failure mode.

## Objectives

Let $U_i^v(t)$ be the fixed unary score for event $i$ at frame $t$ of video $v$.

### Global query

The complete input text is encoded as one query $q$:

$$
S(v)=\max_t U_q^v(t).
$$

The maximizing frame is returned. For a multi-event TRAKE query, this method evaluates video retrieval only; one frame cannot constitute an event-level path.

### Independent event aggregation

Each event selects its strongest frame independently:

$$
z_i^*=\arg\max_t U_i^v(t),
\qquad
S(v)=\frac{1}{M}\sum_{i=1}^{M}U_i^v(z_i^*).
$$

Frames may be reused and their timestamps may be unordered. Mean and sum induce the same within-query ranking because every video is scored against the same $M$ events; the mean keeps scores comparable across different event counts.

### DANTE-style strict unary DP

The full-frame control calls `hcmai.temporal.dp.align_video` with one path per video:

$$
S(v)=\max_{z_1<\cdots<z_M}
\left[
\sum_{i=1}^{M}\phi\!\left(U_i^v(z_i)\right)
-\lambda\sum_{i=2}^{M}(t_{z_i}-t_{z_{i-1}})
\right],
$$

where $\phi(s)=s$ when `event_power=1`, otherwise $\phi(s)=\max(s,0)^{\text{event_power}}$. `cluster_delta` can additionally prevent successive events from occupying one near-identical score region.

### Unary candidate lattice

For each event, retain its top `candidates_per_event` frames, optionally applying timestamp non-maximum suppression. Exact dynamic programming over those sparse layers optimizes the same strict unary objective as above. When every frame is admitted and clustering is disabled, it is an objective-equivalent sparse implementation of the full decoder.

### Unary bounded order

Enumerate the identity order and deterministic permutations reachable within `max_adjacent_swaps`. For each order $\pi$, decode frames chronologically over the unary lattice:

$$
S(v)=\max_{\pi,z_1<\cdots<z_M}
\left[
\sum_i \phi\!\left(U_{\pi_i}^v(z_i)\right)
-\lambda\sum_{i=2}^{M}(t_{z_i}-t_{z_{i-1}})
-\eta\,\operatorname{inv}(\pi)
\right].
$$

This method has no pairwise semantic transition term. It is included so a future transition-aware method cannot attribute gains from order relaxation to transition modeling.

## Inputs and runtime

Run commands from the repository root with `PYTHONPATH=.:src`. The default configuration expects at least:

```text
artifacts/frame_store/frames.parquet
artifacts/indexes/visual/
configs/baseline.yaml
llm/config.yaml
```

Configured Context, ASR, and BM25 artifacts are loaded when available. Query files are discovered under `<query-root>/<split>/` using the existing suffixes:

- `*-kis.txt`
- `*-qa.txt`
- `*-trake.txt`

KIS and QA use `plan_query_events`. TRAKE uses only explicit `E<n>:` or `Cảnh <n>:` lines. Actual scoring requires either the configured embedding service or compatible local embedding dependencies and checkpoints. The runner preserves every query-level failure in the output envelope and returns exit status 1 if any selected query fails, so incomplete experiment runs cannot appear successful.

## Run commands

### 1. Global query

```bash
PYTHONPATH=.:src aic/bin/python -m baseline.run_global_query \
  --query-root artifacts/query \
  --splits 002 \
  --kinds kis qa trake \
  --top-k 100 \
  --output artifacts/baselines/global_query.json
```

### 2. Independent events

```bash
PYTHONPATH=.:src aic/bin/python -m baseline.run_independent_events \
  --query-root artifacts/query \
  --splits 002 \
  --kinds kis qa trake \
  --top-k 100 \
  --output artifacts/baselines/independent_events.json
```

### 3. DANTE-style strict unary DP

```bash
PYTHONPATH=.:src aic/bin/python -m baseline.run_dante_style \
  --query-root artifacts/query \
  --splits 002 \
  --kinds kis qa trake \
  --top-k 100 \
  --lambda-gap 0.00001 \
  --event-power 1.0 \
  --cluster-delta 0.0 \
  --output artifacts/baselines/dante_style_unary_dp.json
```

### 4. Unary candidate lattice

```bash
PYTHONPATH=.:src aic/bin/python -m baseline.run_unary_lattice \
  --query-root artifacts/query \
  --splits 002 \
  --kinds kis qa trake \
  --top-k 100 \
  --candidates-per-event 32 \
  --candidate-min-separation-ms 0 \
  --output artifacts/baselines/unary_candidate_lattice.json
```

### 5. Unary bounded order

```bash
PYTHONPATH=.:src aic/bin/python -m baseline.run_unary_bounded_order \
  --query-root artifacts/query \
  --splits 002 \
  --kinds kis qa trake \
  --top-k 100 \
  --candidates-per-event 32 \
  --max-adjacent-swaps 1 \
  --max-order-candidates 64 \
  --swap-penalty 0.0 \
  --output artifacts/baselines/unary_bounded_order.json
```

Omit DP parameters to inherit `search.alignment` values from `configs/baseline.yaml`. Use `--use-bm25` to enable BM25 or `--no-use-dense --use-bm25` for its isolated ablation.

## Fast smoke test

Limit any method to one selected query before a full run. The CLI still writes the run envelope when individual queries fail, but exits nonzero if any response failed so incomplete experiments cannot appear successful:

```bash
PYTHONPATH=.:src aic/bin/python -m baseline.run_global_query \
  --query-root artifacts/query \
  --splits 002 \
  --kinds kis \
  --max-queries 1 \
  --top-k 10 \
  --output artifacts/baselines/smoke-global.json
```

## Evaluation

The evaluator accepts both historical response lists and the versioned baseline envelope, supporting either the authoritative fixture via `--test-set` or raw query directories via `--query-root`:

```bash
# Evaluate with frozen test-set fixture (recommended, supports KIS, QA retrieval-only, TRAKE)
PYTHONPATH=.:src aic/bin/python -m scripts.evaluation.evaluate_benchmark_v2 \
  --responses artifacts/baselines/dante_style_unary_dp.json \
  --test-set artifacts/evaluation/query_002.json \
  --metadata artifacts/frame_store/frames.parquet \
  --tolerance-seconds 5 \
  --output artifacts/baselines/dante_style_unary_dp.metrics.json

# Legacy directory evaluation fallback
PYTHONPATH=.:src aic/bin/python -m scripts.evaluation.evaluate_benchmark_v2 \
  --responses artifacts/baselines/dante_style_unary_dp.json \
  --query-root artifacts/query \
  --metadata artifacts/frame_store/frames.parquet \
  --tolerance-seconds 5 \
  --output artifacts/baselines/dante_style_unary_dp.metrics.json
```

For a global TRAKE result, video recall is valid, while event recall and AllHit remain false because the method emits only one frame. Compare event grounding only among methods that emit one assignment for every original event.

## Compatibility Notes & Contract Invariants

During and after the query-to-path contract cleanup (Task 0+):

1. **Canonical Identity Invariants:**
   - Evaluator and runner pipelines must preserve canonical `video_id`, `frame_id`, `frame_idx`, and `timestamp_ms`.
   - Modality fusion, lattice construction, and DP decoders must never mutate, invent, or drop canonical IDs.
   - Decoded results maintain exact equality with the pre-cleanup result when replayed from the same unary score matrices:
     ```python
     assert after.video_id == before.video_id
     assert after.frame_ids == before.frame_ids
     assert after.frame_idxs == before.frame_idxs
     assert after.timestamps_ms == before.timestamps_ms
     assert after.score == before.score
     ```

2. **Score and Metric Integrity:**
   - Given byte-identical unary score matrices, numerical kernels and decoder weights remain unchanged, so decoder scores must compare exactly; tolerance must not be loosened.
   - Run `PYTHONPATH=.:src uv run pytest tests/baseline/test_method_replay.py -q` to replay all five methods against fixed matrices with exact identity, ordering, and score assertions.
   - A real-corpus rerun is same-matrix evidence only when query embeddings and numerical-runtime provenance are also fixed. Report upstream float drift separately from decoder regressions.
   - In research extensions, additional verifier scores must be recorded in sidecars or distinct fields (`decoder_score` vs `verifier_score`) without overwriting baseline `score`.

3. **Query Event Planning:**
   - Raw KIS and QA queries use `plan_query_events` to fold attribute sentences, drop trailing questions, and restore chronological order.
   - Explicit TRAKE event lines (`E<n>:` or explicit event arrays) are preserved exactly without splitting or reordering.

4. **Memory and Resource Guardrails:**
   - Avoid loading full parquet tables, massive image datasets, or FAISS indices simultaneously into memory during automated tests or local benchmarks.
   - Use targeted test paths (`tests/architecture/test_query_path_contract.py`, `tests/temporal`, `tests/orchestration`, `tests/api`) with synthetic fixtures to prevent out-of-memory (OOM) conditions.

5. **Frozen Output Boundaries:**
   - HTTP KIS and TRAKE keep their established response schemas; baseline runs keep the `hcmai-baseline-run-v1` evaluator envelope.
   - Research output must use a sidecar keyed by run, query, video, and ordered frame IDs. It must not overwrite existing `.metrics.json` artifacts or baseline `score` values.
   - Requested and effective source readiness plus fusion mode belong to experiment metadata, not production HTTP fields.
