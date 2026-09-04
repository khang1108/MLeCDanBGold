# HCM-AI v10 P0 Temporal Evidence Preliminary Evaluation (Invalidated)

**Date:** 2026-09-04  
**Target Query:** `L26_V254_cooking`  
**Target Video:** `L26_V254` (313 frames)  
**Evaluation Script:** `scripts/evaluate_temporal_p0.py`  
**Results Artifact:** `artifacts/p0_ablation_results.jsonl`

**B-series rerun status:** Blocked on 2026-09-04 because the configured remote
embedding service was unavailable at `/v1/embeddings/text`. No B-series metric
was recorded, and the historical A-series artifact was left unchanged.

---

## 1. Executive Summary

This artifact preserves the first P0 run, but its rollout verdict is invalidated.
The conditions called A0-A6 were not strictly isolated: A1 changed the fusion
equation rather than merely exposing components, and the evaluator replaced the
loaded DP configuration with a fresh default `AlignmentConfig`. Runtime
experiments are now named B0-B6; exact componentized-legacy equivalence is a
regression test rather than a performance condition.

### Key Takeaways:

1. **v9 Legacy Baseline ($A0$) fails to retrieve the target video:** In $A0$, `L26_V254` is ranked $>300$ and does not appear in candidate alignment paths.
2. **Flat component fusion (historical $A1$) brings the target video to Rank 88**, but this is not an isolated componentization gain and frame 450 misses the event-2 plate region.
3. **Dense-only Adaptive (historical $A6$) has a narrow score gap ($0.062$), but localization at frames 3525-3600 is outside the known relevant region.**
4. **BM25 ASR Speech Boost Trade-off ($A3 \dots A5$):** Applying an aggressive static multiplier ($5.0\times$) to BM25 ASR on conversational queries elevates speech-heavy distractor videos in long corpora, causing target rank to degrade back towards $>300$.

---

## 2. Quantitative Ablation Matrix ($A0 \dots A6$)

| Run ID | Name | Target Rank | Target Score | Top Video | Top Score | Score Gap | Retrieval Latency | Aligned Target Frames |
|---|---|---|---|---|---|---|---|---|
| **$A0$** | `A0_legacy_v9` | $>300$ | N/A | `L26_V124` | 2.380 | N/A | 11.7 s | N/A |
| **$A1$** | `A1_components_fixed` | **88** | 2.527 | `L26_V427` | 2.715 | **0.188** | 9.4 s | `[425, 450, 475, 550]` |
| **$A2$** | `A2_asr_interval` | **108** | 2.480 | `L26_V427` | 2.696 | 0.216 | 5.5 s | `[425, 450, 475, 550]` |
| **$A3$** | `A3_robust_calibration` | 240 | 3.095 | `L26_V351` | 3.332 | 0.237 | 5.1 s | `[3525, 3550, 3575, 3600]` |
| **$A4$** | `A4_confidence_gating` | 295 | 3.270 | `L26_V376` | 3.611 | 0.341 | 4.7 s | `[3525, 3550, 3575, 3600]` |
| **$A5$** | `A5_adaptive_p0` | $>300$ | N/A | `L26_V351` | 3.715 | N/A | 4.4 s | N/A |
| **$A6$** | `A6_dense_only` | **116** | 3.908 | `L29_V010` | 3.970 | **0.062** | 3.7 s | `[3525, 3550, 3575, 3600]` |

---

## 3. Findings & Diagnostic Verification

### Alignment with Target Regions in $A1$ and $A2$:

Target regions defined for query `L26_V254_cooking`:

- `hold_two_X`: frames $[300, 475]$
- `plate_X`: frames $[500, 525]$
- `dialogue`: frames $[550, 950]$

The aligned path materialized under $A1$ and $A2$ is `[425, 450, 475, 550]`:
- Event 1: Frame 425 (within $[300, 475]$)
- Event 2: Frame 450 (within $[300, 475]$)
- Event 3: Frame 475 (boundary of $[300, 475]$)
- Event 4: Frame 550 (start of $[550, 950]$)

This path is monotonic, but it does not validate semantic localization. Event 2
should localize to the plate region $[500, 525]$ and instead lands at frame 450
inside the earlier holding region. The paths at frames 3525-3600 in A3, A4,
and A6 are clear localization failures despite their target-video ranks.

### Decision Gate Verdict:

- **NO-GO for P1.** Full adaptive P0 ranked the target below 300 and robust
  calibration coincided with a severe localization regression.
- Rerun B0-B6 with the loaded DP configuration held fixed after correcting
  sparse lexical calibration and half-open ASR interval coverage.
- Diagnose B3 calibration and B5 routing at the known shell, plate, and dialogue
  regions before claiming that DP is the remaining bottleneck.
- Keep legacy fusion as the verified rollback path.
