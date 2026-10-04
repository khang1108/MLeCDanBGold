# From Events to Transitions: Motion-Aware Graph Decoding for Multi-Event Video Retrieval

## Implementation & Experiment Plan for SOICT 2026

**Paper title:**  
**From Events to Transitions: Motion-Aware Graph Decoding for Multi-Event Video Retrieval**

**Target venue:** SOICT 2026 — Lifelogging, Event Retrieval, and Personal Data Analytics track  
**Submission deadline:** 5 October 2026  
**Implementation baseline:** `src_v2.4.2`

---

# 1. Goal

The final architecture should follow:

```text
Query
  ↓
E1 → E2 → ... → EM
  ↓
Existing multimodal retrieval
  ↓
Unary event-frame scores U_i(t)
  ↓
Top-K candidates per event
  ↓
Query-conditioned transition scores ψ_i(a,b)
  ↓
Sparse candidate graph
  ↓
Transition-aware DP
  ↓
EventTrail / ranked videos
```

The decoding objective is:

\[
S(z_1,\ldots,z_M)
=
\sum_i U_i(z_i)
+
\beta\sum_{i=1}^{M-1}\psi_i(z_i,z_{i+1})
-
\lambda\sum_{i=1}^{M-1}\Delta t_i
\]

subject to:

\[
z_i\in C_i,\qquad t(z_i)<t(z_{i+1})
\]

where:

- \(U_i(t)\): existing multimodal unary relevance score for event \(E_i\) at frame \(t\)
- \(C_i\): Top-K candidate frames for event \(E_i\)
- \(\psi_i(a,b)\): query-conditioned transition compatibility between two candidate moments
- \(\beta\): transition weight
- \(\lambda\): existing temporal gap penalty

---

# 2. Phase 0 — Freeze the Baseline

## Task 0.1 — Freeze Existing Temporal DP

**File**

```text
src/hcmai/temporal/dp.py
```

Do not modify the current recurrence, normalization, gap penalty, or ranking semantics.

The current method becomes the frozen experimental control:

```text
B2 = Static Unary Temporal DP
```

### Done condition

- Existing tests pass.
- `dp.py` receives no algorithmic modifications.
- Any future proposed method is implemented beside it, not inside it.

---

# 3. Phase 1 — Candidate Graph Primitives

## Task 1.1 — Extract Candidate Selection from the Decoder

Create a minimal representation:

```python
@dataclass(frozen=True, slots=True)
class EventCandidateLayer:
    event_index: int
    frame_indices: np.ndarray
    scores: np.ndarray
```

Implement:

```python
def select_event_candidates(
    video: VideoEventScores,
    *,
    candidate_k: int,
) -> tuple[EventCandidateLayer, ...]:
```

### Event 0

The first event does not need a temporal predecessor:

```python
valid = np.isfinite(scores[0])
```

Do not filter frame 0 using predecessor reachability.

### Events i > 0

Use the current temporal feasibility rules when constructing candidate layers.

### Selection

Keep the highest unary-scoring frames for each event.

### Validation

Require:

```python
candidate_k > 0
```

### Tests

#### First-frame regression

```text
E1 scores = [10, 1, 0]
E2 scores = [0, 9, 8]
```

Expected:

```text
E1 candidate set contains frame 0
```

#### Invalid K

```text
candidate_k = 0
```

must raise `ValueError`.

#### Large K

```text
candidate_k > number_of_frames
```

must not crash.

### Done condition

Candidate selection is independent from graph decoding and can be reused by both the transition scorer and decoder.

---

# 4. Phase 2 — Sparse Transition Graph Decoder

## Task 2.1 — Make the Decoder Consume External Candidate Layers

Target API:

```python
def decode_candidate_lattice(
    video: VideoEventScores,
    *,
    candidates: Sequence[EventCandidateLayer],
    transition_scores: Sequence[np.ndarray] | None = None,
    transition_weight: float = 0.0,
    ...
):
    ...
```

Do not generate Top-K candidates inside this function.

Each transition matrix must have shape:

\[
\Psi_i \in \mathbb{R}^{K_i\times K_{i+1}}
\]

### Recurrence

For candidate \(b\in C_i\):

\[
D_i(b)
=
U_i(b)
+
\max_{a\in C_{i-1},\,t_a<t_b}
[
D_{i-1}(a)
-\lambda(t_b-t_a)
+\beta\psi_i(a,b)
]
\]

Store parent pointers for backtracking.

### Do not add

- NetworkX
- generic graph frameworks
- arbitrary DAG support
- generic graph node registries

The graph is a fixed layered chain:

```text
C1 → C2 → ... → CM
```

### Complexity target

\[
O(MK^2)
\]

for approximately uniform candidate count \(K\).

---

## Task 2.2 — Baseline Equivalence Test

When:

```text
candidate_k = number_of_frames
transition_weight = 0
```

the proposed decoder must reproduce the frozen baseline:

\[
GraphDP = BaselineDP
\]

Check:

- same selected path
- same path score
- same ranking

Run randomized synthetic tests for:

```text
M = 2, 3, 4
multiple random frame counts
100+ random cases
```

### Done condition

No randomized mismatch between the full-candidate graph decoder and baseline DP.

---

# 5. Phase 3 — Transition Representation

## Task 3.1 — Expose Visual Embeddings for Candidate Frames

Use the existing offline visual index.

Implement an accessor similar to:

```python
def get_frame_embeddings(
    video_id: str,
    frame_indices: np.ndarray,
) -> np.ndarray:
    ...
```

Output:

\[
V\in\mathbb{R}^{K\times D}
\]

### Requirements

- Do not decode and re-encode the source video.
- Use the embeddings already stored in the visual index.
- Preserve canonical frame indexing.
- Verify whether vectors are already L2-normalized.
- Normalize at the scorer boundary if required.

### Done condition

The same candidate frame always resolves to the same stored visual vector.

---

## Task 3.2 — Encode Query Events in the Same Vision-Language Space

For query events:

```text
E1
E2
...
EM
```

encode their text using the **SigLIP text encoder**, not BGE-M3.

Output:

\[
Q=[q_1,\ldots,q_M]\in\mathbb{R}^{M\times D}
\]

### Requirement

Encode all event texts once per query.

### Reason

Visual transition vectors and textual transition vectors must live in the same shared SigLIP space.

---

# 6. Phase 4 — Query-Conditioned Transition Scorer

## Task 4.1 — Implement `EmbeddingDeltaTransitionScorer`

Do not use:

\[
\cos(v_a,v_b)
\]

as the proposed motion score.

That measures appearance continuity rather than event transition.

Use:

\[
\Delta v_{a,b}=v_b-v_a
\]

and:

\[
\Delta q_i=q_{i+1}-q_i
\]

Then:

\[
\boxed{
\psi_i(a,b)
=
\frac14
(v_b-v_a)^T(q_{i+1}-q_i)
}
\]

### Efficient matrix computation

For:

```text
A = source embeddings [Ka, D]
B = target embeddings [Kb, D]
dQ = q_next - q_prev
```

compute:

\[
s_A=A\,dQ
\]

\[
s_B=B\,dQ
\]

Then:

\[
\Psi=
\frac14
\left(
s_B[None,:]-s_A[:,None]
\right)
\]

Do not materialize a:

```text
Ka × Kb × D
```

tensor.

### Complexity

\[
O((K_a+K_b)D+K_aK_b)
\]

---

## Task 4.2 — Preserve Signed Transition Scores

Do not clamp scores with:

```python
np.maximum(score, 0)
```

Negative values are useful because a reversed semantic transition should be penalized.

Example:

```text
sit → stand
```

should score higher than:

```text
stand → sit
```

for the same query edge.

---

## Task 4.3 — Transition Scorer Unit Tests

### Forward transition

```text
q1 = [1, 0]
q2 = [0, 1]

vA = [1, 0]
vB = [0, 1]
```

Expected:

\[
\psi(A,B)>0
\]

### Reverse transition

```text
vA = [0, 1]
vB = [1, 0]
```

Expected:

\[
\psi_{\text{reverse}}
<
\psi_{\text{forward}}
\]

Preferably:

\[
\psi_{\text{reverse}}<0
\]

### Static pair

If:

\[
v_A\approx v_B
\]

the transition signal should remain small.

---

# 7. Phase 5 — End-to-End Motion Graph Decoder

## Task 5.1 — Create a Small Orchestration Function

Implement:

```python
def decode_motion_graph_video(
    video_scores,
    *,
    frame_embeddings,
    event_embeddings,
    candidate_k,
    transition_weight,
    ...
):
    ...
```

Flow:

```text
select candidates
       ↓
fetch candidate embeddings
       ↓
score E1→E2
score E2→E3
...
       ↓
decode candidate lattice
```

Do not introduce a plugin framework or registry.

### Done condition

One function can decode one video end to end using unary and transition evidence.

---

# 8. Phase 6 — Runtime Integration

## Task 6.1 — Keep the Current Runtime as Baseline

Keep the existing:

```python
search_plan(...)
```

as the frozen static baseline.

Add a parallel path such as:

```python
search_plan_motion_graph(...)
```

or an equivalent explicit experimental API.

Do not silently replace production behavior.

---

## Task 6.2 — Minimal Configuration

Use only:

```yaml
temporal:
  decoder: static
```

or:

```yaml
temporal:
  decoder: motion_graph
  candidate_k: 32
  transition_weight: 0.25
```

Reuse the current:

- gap penalty
- event power
- video ranking behavior
- existing multimodal fusion

Do not add configuration for unsupported future features.

---

# 9. Phase 7 — Correctness Gates

Do not start large-scale experiments before every gate below passes.

## Gate 1 — Zero-Transition Equivalence

For:

\[
\beta=0
\]

and:

\[
K=F
\]

require:

\[
MotionGraph = BaselineDP
\]

for path and score.

---

## Gate 2 — Full-Candidate Equivalence

When every frame is retained, sparse graph representation must reproduce the dense objective.

---

## Gate 3 — Transition Signal Can Change the Optimal Path

Construct a synthetic example where unary evidence prefers one path but transition evidence strongly favors another.

The graph decoder must choose the transition-compatible path when \(\beta\) is sufficiently high.

---

## Gate 4 — Forward vs Reverse

Require:

\[
\psi_{\text{correct}}
>
\psi_{\text{reverse}}
\]

---

## Gate 5 — Shuffled Embeddings

Shuffle target-frame embeddings.

Expected:

\[
\psi_{\text{true}}
>
\psi_{\text{shuffled}}
\]

on average.

---

## Gate 6 — Finite Numerics

Require finite values for:

- unary scores
- transition matrices
- DP states
- final path scores

No NaNs or silent broadcasting.

---

# 10. Phase 8 — Candidate Recall Study

This is the first real-data experiment.

For each annotated event, calculate:

\[
CandidateRecall@K
\]

for:

\[
K\in\{4,8,16,32,64\}
\]

Recommended output:

| K | Candidate Recall | Avg. edges/video | Latency |
|---:|---:|---:|---:|
| 4 | | | |
| 8 | | | |
| 16 | | | |
| 32 | | | |
| 64 | | | |

### Selection rule

Choose the smallest K near the coverage elbow.

A useful target is approximately:

\[
Recall_K \ge 95\%
\]

if achievable.

### Interpretation

If ground-truth frames never enter the candidate set, the transition decoder cannot recover them.

---

# 11. Phase 9 — Transition Diagnostic Dataset

Create:

```text
data/eval/transition_diagnostic.json
```

Example schema:

```json
{
  "query_id": "Q001",
  "video_id": "V001",
  "query": "A man sits and then stands up.",
  "events": [
    {
      "id": "E1",
      "text": "a man is sitting",
      "start_ms": 1000,
      "end_ms": 4000
    },
    {
      "id": "E2",
      "text": "the man stands up",
      "start_ms": 5000,
      "end_ms": 7000
    }
  ]
}
```

Prioritize queries with 2–4 events.

### Categories

- state transition
- object manipulation
- direction change
- object-state change
- multi-step interaction
- same event set with different order

Examples:

```text
sitting → standing
holding cup → placing cup
approaching → leaving
closed → open
pick → carry → place
enter → sit vs sit → enter
```

---

# 12. Phase 10 — Counterfactual Evaluation

For each multi-event query:

```text
E1 → E2 → E3
```

generate controlled variants.

## Reverse

```text
E3 → E2 → E1
```

## Adjacent swap

```text
E1 → E3 → E2
```

## Shuffled transition evidence

Keep unary nodes fixed but shuffle edge matrices.

## No-edge control

Set:

\[
\beta=0
\]

These counterfactuals test whether the proposed edge signal actually captures event order and transition direction.

---

# 13. Phase 11 — Experimental Baselines

Keep the experimental matrix focused.

## B0 — Full-Query Retrieval

Use the complete natural-language query with the existing retrieval pipeline.

## B1 — Independent Event Retrieval

Decompose into events but ignore temporal order.

Example:

\[
S(V)=\sum_i\max_t U_i(t)
\]

## B2 — Static Temporal DP

Current frozen `dp.py`.

## B3 — Top-K Candidate Graph without Transition

Use the proposed sparse graph with:

\[
\beta=0
\]

This controls for candidate pruning.

## B4 — Appearance Continuity

Use:

\[
\psi(a,b)=\cos(v_a,v_b)
\]

This demonstrates the difference between appearance persistence and semantic transition.

## Ours — Query-Conditioned Signed Transition

Use:

\[
\boxed{
\psi_i(a,b)
=
\frac14(v_b-v_a)^T(q_{i+1}-q_i)
}
\]

---

# 14. Phase 12 — Hyperparameter Sweep

Do not run a large grid search.

## Candidate count

Test:

```text
K ∈ {8, 16, 32, 64}
```

## Transition weight

Test:

```text
β ∈ {0, 0.05, 0.10, 0.20, 0.40}
```

Keep the existing temporal gap penalty \(\lambda\) fixed initially.

This preserves a fair comparison to the baseline.

---

# 15. Phase 13 — Metrics

## Video Retrieval

Report:

- R@1
- R@5
- R@10
- MRR

## Event Grounding

Define:

\[
EventHit@\delta
\]

for a temporal tolerance appropriate to the sampling rate.

## Full Path Accuracy

Use:

\[
\boxed{AllHit@\delta}
\]

where all events in the decoded path must be correctly grounded.

Also report mean per-event hit rate.

## Transition Diagnostics

Report:

- forward-vs-reverse accuracy
- original-vs-shuffled margin

For example:

\[
Margin_{\text{reverse}}
=
S(Q,V)-S(Q^{rev},V)
\]

## Efficiency

Report:

- candidate K
- number of evaluated edges
- alignment latency

---

# 16. Phase 14 — Required Ablation Tables

## Main Table

| Method | R@1 | R@5 | MRR | EventHit | AllHit | Reverse Acc |
|---|---:|---:|---:|---:|---:|---:|
| B1 Independent | | | | | | |
| B2 Static DP | | | | | | |
| B3 Top-K, β=0 | | | | | | |
| B4 Visual continuity | | | | | | |
| **Ours** | | | | | | |

## Candidate / Efficiency Table

| K | β | Candidate Recall | AllHit | Latency |
|---:|---:|---:|---:|---:|
| | | | | |

---

# 17. Phase 15 — Experiment Runner

Create:

```text
scripts/evaluation/run_motion_graph_ablation.py
```

Example:

```bash
python scripts/evaluation/run_motion_graph_ablation.py \
    --queries data/eval/transition_diagnostic.json \
    --candidate-k 32 \
    --transition-weight 0.2
```

Store machine-readable results:

```text
results/
  baseline_dp.json
  topk_no_edges.json
  visual_continuity.json
  transition_delta.json
  reverse.json
  shuffled.json
```

Add:

```text
scripts/evaluation/summarize_motion_graph_results.py
```

to generate CSV and LaTeX table rows.

Do not manually copy experimental numbers into the paper.

---

# 18. Phase 16 — Figure 1: Motivation

Target visual:

```text
Query:
person sitting → person standing

Static DP
─────────
[sitting]          [standing]
   ✓ node             ✓ node

Selected states are individually relevant,
but the transition may be wrong.


Motion-aware graph
──────────────────
[sitting] ─────────> [standing]
   ✓ node    ✓edge      ✓ node
```

Core message:

> Correct states do not necessarily imply a correct transition.

---

# 19. Phase 17 — Figure 2: Architecture

```text
Narrative Query
      ↓
Event Graph
E1 → E2 → E3
 │    │    │
 ↓    ↓    ↓
Multimodal Unary Retrieval
 │    │    │
 C1   C2   C3
  ╲   │   ╱
Query-conditioned transition edges
      ↓
Sparse Graph Decoder
      ↓
EventTrail
```

---

# 20. Phase 18 — Paper Writing Order

Do not begin with the Introduction.

Recommended order:

1. **Method**
2. **Experimental Setup**
3. **Results and Ablations**
4. **Introduction**
5. **Related Work**
6. **Abstract**
7. **Conclusion**
8. **Limitations**

This ensures the paper claims follow the actual experimental evidence.

---

# 21. Superpowers Task Decomposition

Give Superpowers one task at a time.

```text
SP-01: Fix and extract candidate selection
SP-02: Refactor transition decoder to consume external candidate layers
SP-03: Add baseline-equivalence randomized tests
SP-04: Implement frame embedding accessor
SP-05: Implement query-event SigLIP embedding helper
SP-06: Implement EmbeddingDeltaTransitionScorer
SP-07: Add forward/reverse/shuffle tests
SP-08: Build decode_motion_graph_video()
SP-09: Integrate parallel runtime path
SP-10: Add minimal experiment config
SP-11: Implement Candidate Recall@K evaluator
SP-12: Add Transition Diagnostic Dataset schema and loader
SP-13: Build baseline/Ours experiment runner
SP-14: Build reverse/shuffle counterfactual runner
SP-15: Build result summarizer and LaTeX table output
```

### Rule

Do not give `SP-01` through `SP-15` to an agent at once.

For each task:

```text
implement
↓
review diff
↓
run tests
↓
commit
↓
next task
```

---

# 22. Suggested Commit Strategy

```text
feat/transition-candidates
feat/transition-decoder
test/transition-equivalence
feat/semantic-transition-score
feat/motion-graph-runtime
eval/candidate-recall
eval/transition-diagnostics
eval/motion-graph-ablation
```

This keeps the static baseline easy to recover and isolates experimental changes.

---

# 23. Stop Conditions

## Stop Condition 1 — Candidate Bottleneck

If:

\[
CandidateRecall@32 < 80\%
\]

do not immediately tune the graph decoder.

The unary retrieval stage is likely the bottleneck.

---

## Stop Condition 2 — Transition Hypothesis Fails

If ground-truth transitions do not generally satisfy:

\[
\psi_{\text{correct}}
>
\psi_{\text{reverse}}
\]

then the embedding-delta hypothesis is weak.

Only then consider:

- temporal-window visual embeddings
- patch-level temporal differences
- stronger video representations

---

## Stop Condition 3 — No Causal Edge Contribution

If:

```text
Ours ≈ visual-continuity baseline
```

and shuffled transition edges do not reduce performance, then the edge signal is not contributing meaningfully.

Do not claim motion-aware transition reasoning in that case.

---

# 24. Critical Path Before Submission

## Engineering Critical Path

```text
SP-01
↓
SP-02
↓
SP-03
↓
SP-04
↓
SP-05
↓
SP-06
↓
SP-07
↓
SP-08
↓
SP-09
```

## Experimental Critical Path

```text
Candidate Recall@K
↓
Transition Diagnostic Set
↓
B2 Static DP
↓
B3 Top-K No Edge
↓
B4 Visual Continuity
↓
Ours
↓
Reverse / Shuffle Controls
```

---

# 25. Minimum Paper-Ready Evidence

The implementation is not paper-ready merely because it runs.

The minimum empirical package is:

1. Baseline equivalence test
2. Candidate Recall@K analysis
3. Forward-vs-reverse transition diagnostic
4. Static DP vs proposed retrieval results
5. Shuffled-edge ablation
6. Alignment latency / efficiency table

The empirical core should therefore contain:

\[
\boxed{
B2\ StaticDP
\quad vs\quad
B3\ TopK
\quad vs\quad
B4\ Continuity
\quad vs\quad
Ours
}
\]

plus:

\[
\boxed{
Reverse
+
Shuffle
+
CandidateRecall@K
}
\]

---

# 26. Research Claim to Preserve

Do not claim that the paper introduces motion modeling in general.

The intended claim is:

> Existing structured multi-event retrieval can select individually relevant moments under temporal ordering, but temporal order alone does not verify that consecutive selected moments undergo the transition described by the query. We therefore introduce query-conditioned transition edges into a sparse event-to-video candidate graph and decode the best path using both node evidence and directional transition evidence.

The method should be described as modeling:

> **semantic motion / directional visual transition in a shared vision-language embedding space**

rather than optical motion.

---

# 27. Final Paper Contributions

Keep the contribution list to three points.

1. **Sparse event-to-video candidate graph.**  
   Formulate structured multi-event retrieval using event nodes and candidate video moments while preserving the existing multimodal unary evidence.

2. **Query-conditioned signed transition scoring.**  
   Model directional visual semantic change between consecutive candidate moments and align it with the semantic transition implied by consecutive textual events.

3. **Transition-aware structured decoding and diagnostics.**  
   Decode the best event trail using node and edge evidence and evaluate the mechanism using reverse, shuffled, and candidate-recall diagnostics in addition to retrieval and grounding metrics.

---

# 28. Scope Guard

Before the SOICT deadline, do **not** add:

- optical flow
- learned GNN
- MoE
- large video encoder replacement
- entity tracking
- generic DAG support
- patch-level motion unless the embedding-delta hypothesis fails
- major EventTrail/UI refactors
- additional retrieval fusion changes

The paper should remain focused on:

\[
\boxed{
\text{Unary semantic evidence}
+
\text{Query-conditioned transition evidence}
+
\text{Sparse structured decoding}
}
\]
