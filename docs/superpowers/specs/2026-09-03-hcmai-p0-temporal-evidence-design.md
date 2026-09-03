# HCM-AI P0 Temporal Evidence Design

## Scope

The query-preparation layer is frozen and externalized to the ChatGPT skill. P0 starts from an already prepared dynamic sequence of observable events `E1..EN` and improves only the evidence matrix supplied to the existing temporal decoder.

## Goal

Produce a cleaner, modality-aware emission matrix `S[event, frame]` from the existing Visual, Context, ASR-segment, and fielded BM25 artifacts, without regenerating corpus artifacts or changing DP recurrence semantics.

## Current Problems

1. Dense temporal scoring min-max normalizes Visual, Context, and ASR independently, then gives them fixed weights.
2. BM25 sums title/caption/OCR/ASR first, then the hybrid scorer min-max normalizes the sum and mixes it with Dense using a fixed 0.5/0.5 split.
3. Row-wise min-max can turn weak/noisy rows into apparent `[0,1]` confidence.
4. Dense temporal startup is all-or-nothing: missing Context or ASR disables the entire Dense temporal scorer even though Visual exists.
5. ASR segments are projected to a single canonical frame; uncovered frames are filled with the weakest covered ASR score, conflating "no ASR evidence" with "bad ASR match".
6. All events use the same modality weights even when the event is visibly visual-, OCR-, or ASR-dominant.

## Architecture

P0 introduces first-class evidence components, robust calibration, coverage/reliability gating, and event-adaptive fusion:

```text
prepared events E1..EN
        |
        +--> Visual Dense --------- raw component
        +--> Context Dense -------- raw component
        +--> ASR Dense ------------ raw component + temporal coverage
        +--> BM25 title ----------- raw component
        +--> BM25 caption --------- raw component
        +--> BM25 OCR ------------- raw component
        +--> BM25 ASR ------------- raw component
                    |
                    v
             robust calibration
                    |
             reliability scores
                    |
             event cue routing
                    |
       availability-aware normalized fusion
                    |
              S[event, frame]
                    |
            existing DP unchanged
```

## Hard Constraints

- Do not regenerate Caption/OCR/Object/ASR/Context artifacts.
- Do not rebuild FAISS/BM25 indexes.
- Do not change artifact paths, filenames, schemas, or manifests.
- Do not reintroduce runtime Qwen candidate generation.
- Do not change KIS/TRAKE public request/response contracts.
- Do not change `src/hcmai/temporal/dp.py` recurrence in P0.
- Preserve a `legacy` fusion mode that reproduces v9 numerical behavior when all artifacts are available.
- `N` is dynamic; nothing may assume four events.

## P0 Success Criteria

1. Legacy mode reproduces v9 scores within floating-point tolerance.
2. Adaptive mode never amplifies a constant/near-constant evidence row into a confident row.
3. Missing ASR does not disable Visual/Context; missing Context does not disable Visual/ASR.
4. ASR only contributes on canonical frames covered by the segment interval, with deterministic nearest-frame fallback when no canonical frame lies inside the interval.
5. Adaptive fusion renormalizes over evidence actually available at each `(event, frame)`.
6. Every component can be inspected independently for the known `L26_V254` diagnostic case.
7. DP receives the same `VideoEventScores` contract and remains unchanged.
