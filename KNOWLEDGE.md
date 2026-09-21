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
