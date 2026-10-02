# HCMAI Research Memory

## Evidence for interaction and query inspection in the VBS introduction

**Date:** 2026-09-21
**Problem:** Ground the SHI introduction in observed VBS behavior without claiming
that all successful searches require interaction or that explanation is new to VBS.

### Sources

- Schall et al. (2024), [Interactive multimodal video search: an extended post-evaluation for the VBS 2022 competition](https://doi.org/10.1007/s13735-024-00325-9).
- Jäckl et al. (2026), [What Drove Success at the 15th Video Browser Showdown? A Comprehensive Interaction-Logging Analysis](https://doi.org/10.1145/3805622.3810635). Author institutional abstract: https://pure.itu.dk/en/publications/what-drove-success-at-the-15th-video-browser-showdown-a-comprehen/.
- Heller et al. (2021), [Towards Explainable Interactive Multi-Modal Video Retrieval with vitrivr](https://dbis.dmi.unibas.ch/publications/2021/VBS-2021/). Author abstract inspected; full paper not obtained in this review.
- Lokoč et al. (2023), [Video Search with CLIP and Interactive Text Query Reformulation](https://doi.org/10.1007/978-3-031-27077-2_50). Publisher abstract inspected; full paper not obtained in this review.

### Findings

**PAPER:** The VBS 2022 post-evaluation compares three systems on 57 visual KIS
tasks. Better initial text retrieval did not directly determine overall ranking;
browsing, image queries, and user differences also mattered. This is evidence in
that evaluation setting, not a universal claim that every search requires refinement.

**PAPER:** The 2026 logging study concerns two instrumented VBS systems, not all
participants. Its author abstract reports frequent textual reformulation and result
inspection as the prevalent strategy across most categories. It is an ICMR paper
about VBS, not a VBS system submission. The abstract does not establish that
explanations or editable constraints improve outcomes.

**PAPER:** vitrivr already investigated result explainability in 2021. The 2023
CLIP/query-reformulation paper describes class suggestions for intermediate results
to help users formulate queries. Avoid claiming that supporting interpretation or
reformulation itself is new to VBS.

### Relevance to HCMAI

**PROPOSED:** Position SHI around inspecting and editing structured query
constraints, if that describes the implemented interface. Distinguish query
representation from result explanation, label suggestions, and relevance feedback.
Novelty remains unverified. No implementation or effectiveness claim follows from
this writing review.

### Decision or experiment

Use scoped empirical statements in the introduction. Before claiming improved
search, compare a text-only reformulation interface, inspection without editing,
and inspection with editing, holding retrieval and candidate budgets fixed.
Measure task success, time, and correction actions. These are proposed ablations,
not an approved experiment plan or measured results.

The ten-paper editorial comparison is stored in
`paper/notes/introduction-style-benchmark.md`. The manuscript remains unchanged.

## Related-work positioning for Query Hypothesis and EventTrail

**Date:** 2026-09-21
**Problem:** Position SHI against interactive VBS systems, query reformulation,
result explanation, and temporal-query mechanisms without claiming that those
capabilities are new.

### Sources

- VISIONE system papers from VBS 2022--2024.
- Vibro at VBS 2022; PraK, diveXplore, vitrivr, and Exquisitor at VBS 2024.
- Exquisitor at VBS 2025 and 2026.
- Heller et al. (2021), explainable interactive retrieval with vitrivr.
- Heller et al. (2022), multimodal temporal queries with vitrivr.
- Loko{\v{c}} et al. (2023), interactive CLIP query reformulation.
- Schall et al. (2024) and J{\"a}ckl et al. (2026), VBS interaction analyses.

### Findings

**PAPER:** VBS systems already support multimodal retrieval, browsing, query
reformulation, relevance feedback, result explanation, and temporal queries.

**PAPER:** Exquisitor 2026 retrieves multi-event temporal sequences with a
sequence-chain representation and reciprocal-rank fusion. Temporal sequence
retrieval itself must therefore not be presented as new in SHI.

**PAPER:** vitrivr has exposed feature contributions and temporal context for
result explanation. Explainability in general must not be presented as new.

### Relevance to HCMAI

**PROPOSED:** Position Query Hypothesis as a source-grounded, revisioned event
structure that supports previewed structural edits before retrieval.

**PROPOSED:** Position EventTrail as interaction with a selected video's
complete event-to-occurrence path through event-conditioned alternatives and
anchor/exclusion constraints. Do not position it as a new temporal-query syntax
or a new decoding objective.

### Decision or Experiment

Every external or empirical claim in the manuscript must carry an adjacent
citation that supports that exact clause. SHI implementation descriptions and
explicitly labeled project hypotheses do not require external citations.
Effectiveness still requires controlled comparisons of text-only reformulation,
inspection without editing, and inspection with editing.

## Critical Path Simplification & Motion-Aware Graph Decoding

**Date:** 2026-10-02
**Problem:** The previous KIS orchestration path carried significant overengineering:
a 230-line `_resolve_operation` legacy state machine in `pipeline.py`, generic DAG edge validation
despite domain topology being strictly linear $E_1 \to E_2 \dots \to E_n$, duck-typed plan builders,
and temporal decoding lacking candidate lattice pruning and motion transition scoring.

### Sources
- Repository critical path trace: `src/hcmai/orchestration/pipeline.py`, `src/hcmai/kis/models.py`, `src/hcmai/retrieval/plan.py`.
- Dynamic programming temporal alignment: `src/hcmai/temporal/dp.py`, `src/hcmai/temporal/transition_decoder.py`.
- Paper draft: *From Events to Transitions: Motion-Aware Graph Decoding for Multi-Event Video Retrieval*.

### Findings
**SOURCE:** The active UI workflow operates via `QueryHypothesis` sessions executing
`search_only`, rendering legacy `initial_resolve`, `patch_events`, and `global_rewrite`
operations obsolete on the critical search path. Completely eliminating `_resolve_operation`
and routing `search_kis` exclusively through `query_hypothesis_session_id` or `base_intent` with `search_only`
shed over 1,250 lines of dead code and tests without breaking any frontend workflows.

**SOURCE:** The retrieval plan previously only retained `events: tuple[KISRetrievalEvent, ...]`,
dropping transition information. In the updated architecture, `KISRetrievalPlan.transitions` is a pure
derived `@property` returning `tuple[RetrievalTransition, ...]`, eliminating state synchronization bugs.

**VERIFIED:** Decoupled algorithmic decoding into two clean modules:
1. `src/hcmai/temporal/dp.py`: **[FROZEN BASELINE]** Static Temporal Baseline (Unary + temporal order + time gap penalty).
2. `src/hcmai/temporal/transition_decoder.py`: **[VERIFIED]** Motion-Aware Graph Decoding incorporating pairwise transition edge compatibility $\psi(s, t; E_{i-1}, E_i)$, strict matrix validation (finite values, shape checks), $O(M K^2)$ candidate lattice decoding (`decode_candidate_lattice`), and `MotionCosineTransitionScorer`.

### Status
VERIFIED (Full test suite passing: 555 backend unit tests, 382 frontend tests).

### Decision or Experiment
1. Keep `dp.py` frozen as the authoritative baseline for all paper ablations.
2. Use `decode_candidate_lattice` with `MotionCosineTransitionScorer` on multi-event KIS video benchmark queries, evaluating precision/recall gains when incorporating motion transition edge weights versus the unary baseline.


