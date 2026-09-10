# Temporal Alignment as a Publication Direction: Assessment and Recommendations

**Date:** 2026-09-09<br>
**Status:** RESEARCH RECOMMENDATION — CONDITIONAL GO<br>
**Scope:** Multi-event video retrieval and temporal grounding for HCMAI 2026 and a possible SOICT submission

---

## 1. Executive decision

Temporal alignment is the right research area to investigate, but the current strict monotonic dynamic program should be treated as a **baseline**, not as the paper's principal contribution.

The recommended paper direction is:

> **Transition-Aware and Order-Robust Alignment for Multi-Event Video Retrieval**

The central claim should not be that dynamic programming can align ordered events. Closely related methods, particularly DANTE, already occupy that space. A more defensible claim is that existing methods mainly combine independent event-to-frame evidence under an ordering constraint, whereas the proposed method explicitly scores **whether two selected moments form a plausible event transition** and treats the order written by the user as uncertain evidence rather than an absolute rule.

The project should proceed only if a frozen benchmark and controlled baseline can be established immediately. The existing five-query study is valuable for generating hypotheses, but it is not publication-grade evidence.

---

## 2. Assessment of the current temporal decoder

The active implementation in [`src/hcmai/temporal/dp.py`](../../src/hcmai/temporal/dp.py) performs strict chronological event-to-frame alignment. In simplified form, its objective is

$$
\max_{z_1 < \cdots < z_M}
\left[
\sum_{i=1}^{M} U_i(z_i)
-
\lambda \sum_{i=2}^{M}\left(t_{z_i}-t_{z_{i-1}}\right)
\right],
$$

where $U_i(z_i)$ is the unary compatibility between event $i$ and frame $z_i$.

This is a useful and efficient baseline, but its gap term telescopes:

$$
\sum_{i=2}^{M}\left(t_{z_i}-t_{z_{i-1}}\right)
=
t_{z_M}-t_{z_1}.
$$

Consequently, the current gap penalty controls only the total path span. It does **not** distinguish:

- plausible and implausible intermediate pacing;
- event-specific durations;
- continuation of the same person or object versus an entity switch;
- valid state changes such as *holding an ingredient* followed by *placing it on a plate*;
- a missing or unsupported event;
- chronological narration from rhetorical or recall order.

The decoder contributes unary semantic matching, same-video continuity, strict order, optional score-cluster separation, and path compactness. It does not yet perform genuine transition reasoning.

The implementation is also close to DANTE's monotonic recurrence and prefix-maximum optimization. Therefore, presenting strict DP itself as novel would be difficult to defend.

---

## 3. Evidence from the repository

The local failure analysis in [`2026-09-04-temporal-quality-failure-analysis.md`](2026-09-04-temporal-quality-failure-analysis.md) identifies several important mechanisms:

1. Real narrative queries can list events in a different order from the video.
2. Mandatory matching of every event can punish the true video when one event has weak evidence.
3. A linear gap penalty can force paths into an unrealistically compact span.
4. Fixed multimodal weights can dilute strong visual evidence.
5. Lexical title matches can dominate frame-level evidence.

These observations support temporal research, but they do not by themselves validate the previously explored soft-order decoder. The five-query experiment contains confounds:

- the soft-order stage evaluated the top-50 first-stage candidates plus a forcibly inserted target;
- first-stage and second-stage runs did not always use identical modalities;
- one apparent recovery involved a single-event query and therefore cannot demonstrate order handling;
- the corrected historical soft-order implementation was not evaluated for quality;
- its measured worst-case latency was approximately 4.26 seconds for a single 2,735-frame video.

These results should be described as **diagnostic observations**, not as final comparative evidence.

---

## 4. Recommended research gap

A narrow and defensible gap is:

> Current multi-event retrieval systems can learn strong event representations or enforce event order, but they generally do not explicitly model query-conditioned compatibility between consecutive selected moments during structured path inference. Strict order is additionally brittle when a user's narration follows rhetorical emphasis or recall order instead of video chronology.

This gap produces two linked research questions:

1. Does pairwise transition evidence improve retrieval beyond independent event scores and a generic time-gap penalty?
2. Can limited uncertainty over query order improve robustness without discarding useful chronological structure?

The first question should be the main contribution. The second should be a controlled extension, not an independent project.

---

## 5. Proposed method

### 5.1 Candidate lattice

For each query event, retain the top $K$ frame or segment candidates per video. Perform richer transition scoring only on this sparse lattice rather than all frame pairs.

This has two advantages:

- transition features can be more expressive than the current $O(MF)$ recurrence;
- the latency and memory cost can be controlled through $K$ and beam width.

Candidate generation must remain identical across the relevant ablations so that improvements can be attributed to the transition model.

### 5.2 Transition-aware objective

Use a structured objective of the form

$$
\max_{\pi,\,z_1<\cdots<z_M}
\left[
\sum_{i=1}^{M} U_{\pi_i}(z_i)
+
\sum_{i=2}^{M}
T_{\pi_{i-1},\pi_i}(z_{i-1},z_i)
-
\eta d(\pi,\mathrm{id})
\right].
$$

Here:

- $U_i(z)$ is the existing event-to-frame multimodal score;
- $T_{i,j}(a,b)$ scores whether moments $a$ and $b$ plausibly connect events $i$ and $j$;
- $\pi$ is a candidate event order;
- $d(\pi,\mathrm{id})$ penalizes departures from the order supplied by the user.

A practical transition score is

$$
T_{i,j}(a,b)
=
\alpha D_{i,j}(t_b-t_a)
+
\beta C_{i,j}(a,b),
$$

where $D$ is an event-conditioned duration or gap model and $C$ is semantic transition compatibility. The latter may use frozen text and visual embeddings with a small learned transition head.

Useful transition-training negatives include:

- the correct frame pair in reverse order;
- a correct first event followed by an unrelated moment from the same video;
- an entity-switched pair;
- an action- or state-replaced second event;
- a pair with the correct individual events but an implausible temporal separation.

### 5.3 Bounded order uncertainty

Do not initially support arbitrary event permutations or backward frame edges. Instead, evaluate a small order set containing:

- the identity order;
- one adjacent swap;
- optionally, two bounded adjacent swaps for longer queries.

For every candidate event order, frames are still decoded chronologically. This separates uncertainty about the **query's event order** from the physical direction of time in the video and should be easier to optimize than reverse temporal edges.

### 5.4 Features to defer

Unless they are already cheap to implement, the following should not be central to the first paper:

- unrestricted permutations;
- a full entity tracker;
- a large VLM verifier;
- a dedicated motion encoder;
- adaptive modality fusion as a second major contribution;
- elaborate null-state semantics.

A single optional-event/null transition may be included as an ablation if weak or missing query events remain a dominant measured failure mode.

---

## 6. Falsifiable hypotheses

The paper should test explicit hypotheses rather than reporting only aggregate retrieval gains.

### H1 — Transition contribution

With candidate generation, unary evidence, and score normalization fixed, adding pairwise transition compatibility improves video retrieval and event grounding over unary monotonic DP.

### H2 — Mechanism-specific gains

The improvement is larger on transition-sensitive cases: state change, entity continuity, action replacement, event inversion, and partial-event matching.

### H3 — Order robustness

Bounded latent event order outperforms both strict query order and fully order-invariant matching on naturally written recall queries, while preserving performance on truly chronological queries.

### H4 — Practical decoding

Candidate-lattice inference preserves most or all of the quality gain while meeting an interactive latency target.

A failed hypothesis is still informative. For example, if semantic transitions add no benefit beyond event-conditioned gap priors, the broader transition claim should be reduced accordingly.

---

## 7. Minimum defensible experimental program

### 7.1 Baselines

At minimum, compare:

1. one global query representation;
2. independent per-event aggregation without order;
3. the current strict unary monotonic DP, identified as a DANTE-style baseline;
4. a unary candidate-lattice or beam-search baseline;
5. transition-aware alignment with strict order;
6. transition-aware alignment with bounded order uncertainty;
7. one reproducible modern global event/compositional retrieval model, where code and data permit.

All internal causal comparisons should use the same encoder outputs and candidate pool.

### 7.2 Ablations

Required ablations are:

- no gap term;
- current linear gap term;
- event-conditioned duration only;
- semantic transition only;
- duration plus semantic transition;
- strict versus bounded order;
- candidate-lattice size $K$;
- each transition-negative category;
- optional null state, if included.

### 7.3 Metrics

Report separate metrics for separate claims:

- **Video retrieval:** R@1, R@5, R@10, and MRR.
- **Temporal grounding:** temporal error, hit within a fixed tolerance, and temporal IoU where intervals are available.
- **Whole-path quality:** mean event hit rate and AllHit@$\delta$.
- **Compositional robustness:** accuracy on reorder, action-replacement, segment-mismatch, entity-switch, and partial-match subsets.
- **Efficiency:** P50/P95 latency, memory, candidate count, and expensive model calls.
- **Uncertainty:** paired bootstrap confidence intervals and per-query paired testing.

Aggregate recall alone is insufficient: the evaluation must show that the proposed mechanism solves the failure category it claims to solve.

---

## 8. Benchmark requirements

The repository currently contains 55 public query text files, but the prior migration review found no frozen manifest containing answer videos, accepted intervals, and versioned predictions. This is the main research blocker.

A publication-grade protocol should include:

- target-video labels;
- event-level temporal intervals;
- the chronological order in the video;
- the user's natural written order;
- categorized hard negatives;
- video-disjoint development and test partitions;
- a frozen test split that is not inspected during tuning;
- full-corpus evaluation without target insertion.

A public dataset should provide the primary evidence if possible. Candidate sources include MELON, ActivityNet Captions, YouCook2, or a VideoComp-style compositional protocol, subject to data and license availability. The HCMAI/lifelog queries can provide a valuable in-domain secondary evaluation. If only the existing 55 queries can be annotated, they should be treated as a case study rather than the sole basis for broad generalization.

Every run should record:

- dataset and annotation version;
- model checkpoints and frozen encoder versions;
- index and corpus checksums;
- source revision and configuration;
- candidate-generation policy;
- latency environment;
- confidence intervals and failure examples.

---

## 9. Position relative to nearby work

The final related-work section must verify each paper in detail, but the present positioning is:

| Work | Nearby contribution | Consequence for this project |
|---|---|---|
| [DANTE](https://arxiv.org/abs/2512.13169) | Monotonic dynamic programming for multi-event alignment | Strict ordered DP and its prefix optimization are baselines, not novelty |
| [MADTempo](https://arxiv.org/abs/2512.12929) | Ordered multi-event search using boundary candidates, context, and beam search | Candidate/beam decoding alone is unlikely to be sufficient novelty |
| [EA-VTR](https://arxiv.org/abs/2407.07478) | Event-aware video-text representations | Distinguish global event representation learning from explicit path-transition inference |
| [Multi-event Video-Text Retrieval](https://arxiv.org/abs/2308.11551) | Earlier formulation of multi-event retrieval | Avoid claiming the task formulation itself |
| [VideoComp](https://arxiv.org/abs/2504.03970) | Compositional training and disruption-based hard negatives | Reuse or adapt strong compositional stress tests |
| [MELON](https://arxiv.org/abs/2609.01654) | Long-video multi-event retrieval and analysis of event order | Supports the need to test rather than assume strict textual order |
| [Lucifer-TRACE](https://link.springer.com/chapter/10.1007/978-981-92-2590-3_5) and VidAlign | Nearby structured/VLM-assisted retrieval terrain | A generic VLM reranker is not a strong primary contribution |

The intended distinction is therefore:

> **Explicit, query-conditioned pairwise transition modeling during structured path inference, combined with controlled uncertainty over the order in which events were described.**

---

## 10. Deadline and go/no-go gates

As of 2026-09-09, there is a deadline discrepancy that must be resolved before committing substantial effort:

- the public SOICT paper-submission page lists a general-paper deadline of **2026-09-16**;
- the special-track announcement available to the project reports **2026-10-05**.

If 2026-09-16 is the binding deadline, a new research paper based on this direction is a **no-go**. There is not enough time to freeze the benchmark, implement the method, run controlled experiments, and prepare a defensible manuscript.

If 2026-10-05 is confirmed for the intended track, the project is a **conditional go** with this schedule:

| Gate | Latest date | Required evidence |
|---|---:|---|
| Benchmark gate | 2026-09-13 | Frozen annotations, candidate protocol, strict baseline, and leakage checks |
| Mechanism gate | 2026-09-18 | Held-out improvement on at least one transition-sensitive subset with no major general regression |
| Experiment gate | 2026-09-23 | Main table, core ablations, confidence intervals, and latency |
| External-validity gate | 2026-09-27 | Result on at least one public benchmark or a clearly justified alternative |
| Writing gate | 2026-09-30 | Complete manuscript draft, figures, limitations, and reproducibility details |
| Submission buffer | 2026-10-01 to 2026-10-04 | Review, formatting, artifact checks, and final corrections |

A reasonable mechanism gate is a positive paired confidence interval on the targeted subset, accompanied by a practically meaningful gain rather than isolated query wins. If that evidence does not exist by approximately 2026-09-18 to 2026-09-20, the project should stop or be reframed as a system/diagnostic report instead of forcing a novelty claim.

---

## 11. Principal risks

### Novelty risk

Transition modeling may already exist under different terminology in recent work. Before fixing the contribution statement, conduct a focused literature review of pairwise event transitions, latent event order, temporal graph alignment, and long-video moment retrieval.

### Evaluation risk

The project may improve hand-selected narrative inversions without improving a frozen population. This is why a held-out benchmark and mechanism-labeled subsets are mandatory.

### Representation risk

A transition head cannot recover evidence absent from the underlying frame, caption, ASR, or motion representations. Poor unary evidence must be measured separately from a decoder failure.

### Scope risk

Combining transition learning, order uncertainty, optional events, adaptive fusion, VLM verification, and entity tracking would create an unfocused paper. The minimal contribution should remain transition-aware alignment, with bounded order uncertainty as one extension.

### Efficiency risk

Naive all-pairs transition scoring can become $O(MF^2)$. A top-$K$ candidate lattice, caching, and a bounded beam are not optional implementation details; they are part of making the method credible.

---

## 12. Final recommendation

Proceed with temporal alignment research under the following conditions:

1. Treat the current strict monotonic DP as the principal baseline.
2. Make pairwise, query-conditioned transition compatibility the core technical contribution.
3. Model query order as bounded uncertainty rather than allowing arbitrary reverse-time frame transitions.
4. Freeze a benchmark before tuning and evaluate every method on identical candidate sets.
5. Report mechanism-specific grounding and robustness results in addition to retrieval recall.
6. Keep the first paper narrow; defer unrelated fusion, VLM, and tracking contributions.
7. Confirm that 2026-10-05 is the applicable submission deadline.

The direction is promising because the repository already contains concrete evidence that unary strict-order alignment fails in realistic ways. It becomes publishable only if the proposed transition model produces repeatable, held-out improvements beyond DANTE-style DP while remaining fast enough for practical retrieval.

---

## References

- DANTE: <https://arxiv.org/abs/2512.13169>
- MADTempo: <https://arxiv.org/abs/2512.12929>
- EA-VTR: <https://arxiv.org/abs/2407.07478>
- Multi-event Video-Text Retrieval: <https://arxiv.org/abs/2308.11551>
- VideoComp: <https://arxiv.org/abs/2504.03970>
- MELON: <https://arxiv.org/abs/2609.01654>
- Lucifer-TRACE: <https://link.springer.com/chapter/10.1007/978-981-92-2590-3_5>
- SOICT paper submission: <https://soict.org/submission/paper-submission/>
