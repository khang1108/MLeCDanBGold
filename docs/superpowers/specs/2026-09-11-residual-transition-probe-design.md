# Residual Transition Probe Design

## Goal

Determine whether the existing event-to-frame score matrix contains a useful directional transition signal before changing the production temporal decoder. The probe must require no training, no new model, no embedding rebuild, and no handcrafted duration prior.

## Core identity

For adjacent query events `i` and `i+1`, and chronological frame candidates `a < b` from the same video, define the raw score-space residual:

```text
R(i,a,b) = scores[i+1,b] - scores[i+1,a]
           - scores[i,b] + scores[i,a]
```

This equals `(q[i+1] - q[i]) dot (z[b] - z[a])` when `scores = q dot z`. It therefore needs only `VideoEventScores.scores`; raw query and frame vectors are unnecessary.

The valid-edge contract is strict: candidates must already belong to one `VideoEventScores` object and `timestamps_ms[b] > timestamps_ms[a]`. No gap or expected-duration penalty is used. The `reverse` negative is a directionality diagnostic and is deliberately evaluated outside this valid-forward-edge contract; it can never enter DP.

## Variants under test

1. `raw`: the four-score residual above.
2. `robust_z`: normalize raw residuals within each event-pair/video lattice using median and MAD.
3. `multiscale_median`: replace each endpoint score by neighborhood means at radii `(0, 1, 2)` keyframes and take the median residual across scales.
4. `multiscale_robust_z`: apply lattice-wise median/MAD normalization to variant 3.

The probe must report all variants. It must not tune a variant on the final benchmark metrics.

## Probe pairs

An explicit JSONL manifest is the evaluation boundary. It stores canonical competition-facing `frame_idx` values, never NumPy column positions. Each row identifies a query, adjacent event index, video, `positive_frame_idx_a`, `positive_frame_idx_b`, `negative_frame_idx_a`, `negative_frame_idx_b`, and a negative type. The evaluator resolves each `frame_idx` to exactly one `VideoEventScores.frame_idx` column and rejects missing or duplicate identities. Required negative types are:

- `reverse`: swap the positive endpoints;
- `wrong_next`: retain the positive first endpoint and replace the second with a high-unary wrong candidate from the same video;
- `wrong_previous`: replace the first endpoint while retaining the positive second endpoint;
- `same_video_unrelated`: use a chronological pair from the same video but outside the annotated narrative transition.

Ground-truth paths are used only by `scripts/evaluation/build_residual_probe_manifest.py`, never by retrieval or ranking code. The builder reads `event_windows`, retains only adjacent events whose representative frames are strictly chronological, and selects negatives from the top-32 per-event candidate pool in the same video. Split and query identifiers must be retained in the report. Unavailable negative types must be reported rather than fabricated.

The supplied window fixtures currently provide approximately 13 chronological adjacent transitions across five TRAKE queries. These support a pilot implementation check only. The research gate requires at least 30 transitions from independent query/video groups; multiple tolerance paths for one query do not count as independent transitions.

## Metrics and decision gate

The primary metric is pairwise accuracy `P(R_positive > R_negative)`, with ties scored as 0.5. Report per-negative-type and query-macro pairwise accuracy. ROC-AUC is computed inside each query/event group with both classes present and then macro-averaged; do not pool raw residuals across queries/videos. Report a query-level paired bootstrap 95% confidence interval with a fixed seed.

For `robust_z`, median and MAD are fit over all strict chronological edges between the top-32 candidates for event `i` and the top-32 candidates for event `i+1` in the same video. Gold labels are not used to fit normalization and the full-video `N^2` edge set is never materialized.

Runs with fewer than 30 independent transitions are labeled `PILOT_INCONCLUSIVE` regardless of their scores. Residual signal passes Phase 1 only if one predeclared variant satisfies all of:

- at least 30 positive transitions;
- overall pairwise accuracy >= 0.65;
- reverse accuracy >= 0.70;
- at least three negative types have accuracy >= 0.60 when four are available;
- lower bootstrap bound for overall accuracy > 0.50.

If no variant passes, stop: do not add Residual-DP. Preserve the negative result and inspect score distributions or another representation.

## Phase 2 decoder experiment

Only after Phase 1 passes, add a separate method `residual_candidate_lattice`; do not mutate unary controls. Its recurrence is:

```text
DP[i,b] = U[i,b] + max_a(DP[i-1,a] + beta * R(i-1,a,b))
```

subject to strict chronological positions within one video. Evaluate fixed `beta` values `(0.1, 0.25, 0.5, 1.0)` and include `beta=0` as the implementation-equivalence control. Do not add skip states, adjacent swaps, gap penalties, or duration priors in this probe.

Phase 2 is promising only if at least one fixed nonzero beta improves exact-path or path-MRR over unary candidate lattice, does not reduce video R@1 by more than one query, and adds less than 2x alignment latency. This is an experiment gate, not a production rollout.

## Artifacts

- Machine-readable pair report JSON.
- Edge-level CSV with positive score, negative score, margin, variant, and negative type.
- End-to-end baseline JSON and metrics JSON for each beta, when Phase 1 passes.
- Markdown summary recording dataset size, gates, failures, and the chosen next decision.
- A `KNOWLEDGE.md` entry marked `PROPOSED`, `VERIFIED`, `REJECTED`, or `INCONCLUSIVE`, as required by the repository research-memory convention.

## Non-goals

- Training a temporal model.
- Rebuilding visual embeddings.
- Claiming residual geometry is valid before the gate passes.
- Introducing a fixed action duration or maximum gap.
- Changing production search behavior during Phase 1.
