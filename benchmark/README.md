# benchmark

Ground truth for KIS/QA/TRAKE, built from our own contest submissions.

## Run

```bash
python3 benchmark/build_labels.py            # -> labels.jsonl + queries.yaml

PYTHONPATH=src:. aic/bin/python scripts/evaluate_temporal_p0.py \
  --queries-file benchmark/queries.yaml \
  --runs A0 --no-use-bm25 \
  --output-file artifacts/p0_ablation_results.jsonl

python3 benchmark/score.py                   # Recall@K + MRR per run
python3 benchmark/score.py --sources curated # only the 24 solid labels (TRAKE dropped)
```

The runner needs a live embedding endpoint (`common/config.py:449`).

## Two label sources, different quality

| `source` | Origin | Queries | How far to trust it |
| --- | --- | --- | --- |
| `curated` | `SOTUYEN1/submission_final` | 25 | Every row is correct: 1-6 rows, all one video |
| `top10` | `SOTUYEN2/submission_final` | 30 | First 10 rows of a 100-row shotgun submission |

`SOTUYEN2` was submitted shotgun-style: each file holds 99-100 rows spread over
6-23 videos. Taking the top 10 still leaves 19/30 queries with more than one
video in the label, so `top10` is looser than `curated`. Report the two groups
separately, never merge them silently.

## Join on `frame_idx`, never touch fps

`frame_idx` in `frames.parquet` is exactly the organizers' submission coordinate
(`common/utils/video.py:131` returns it directly, deliberately not derived from
`timestamp_ms × fps`). The corpus is sampled at 1 fps, so it only keeps multiples
of 25 or 30 — the `11276` from the question does not exist, `11275` does. Labels
therefore take the **nearest frame by `frame_idx` within the same video**;
`gap_frames` records the offset, measured max 15, mean 6.5 (< 0.5 s).

## Scoring

`score.py` computes two kinds of hit over the same ranking:

    V@K   the correct video is within the first K paths
    F@K   some frame falls within ±tolerance-ms of a gold candidate (default 2000)

MRR is computed on the video rank. TRAKE is skipped (n=3, not enough to decide
anything).

## Baseline A0, 2026-09-04

Dense-only (`--no-use-bm25`), n=52, tolerance 2000 ms:

| group | n | V@1 | V@10 | V@50 | F@1 | F@10 | F@50 | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 52 | 14 | 32 | 48 | 6 | 16 | 22 | 0.385 |
| curated | 24 | 5 | 16 | 23 | 2 | 9 | 11 | 0.355 |
| top10 | 28 | 9 | 16 | 25 | 4 | 7 | 11 | 0.411 |

Across the 48 queries that find the right video in the top 50, the timestamp
error is: median 6.5 s, p75 58 s, max 679 s. The bottleneck is localizing within
the video, not finding the video.

## P1, 2026-09-04

| run | BM25 | V@1 | V@5 | V@10 | V@50 | F@10 | F@50 | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | off | 14 | 27 | 32 | 48 | 16 | 22 | 0.385 |
| B1 | title/caption/ocr | 17 | 32 | 38 | 48 | 17 | 21 | **0.461** |
| B2 | + asr, weight 1.0 | 17 | 30 | 35 | 46 | 16 | 18 | 0.432 |

Lexical BM25 clearly lifts the video rank and barely touches frame localization.

**The conclusion about the `asr` field in the table above has been refuted, see
P4.** It was measured under the old planner (`split_query_events`); under the P2
planner, turning the ASR field off makes **both** modes worse. `bm25_fields.asr_weight`
stays at **1.0**.

## P2, 2026-09-04

A deterministic planner replaces `split_query_events` for KIS. Same B1 config.

| run | planner | V@5 | V@10 | V@20 | V@50 | F@20 | F@50 | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | sentence split | 32 | 38 | 43 | 48 | 18 | 21 | 0.461 |
| C1 | drop the question + merge attributes + reorder `Trước đó` | 33 | 38 | 45 | 47 | 19 | 21 | 0.474 |
| C2 | like C1, question kept as an attribute | 32 | 38 | 44 | 47 | 18 | 21 | 0.442 |
| C3 | like C1, attribute-merge rule removed | 34 | 39 | 46 | 48 | 18 | 19 | 0.474 |

C1 beats B1 by 0.013 MRR but 7 queries improve / 8 regress — **that is inside the
noise at n=52, do not read it as an improvement**. The reason to keep C1 is that
it fixes the right bug: 13 QA queries no longer force the DP to find a frame for
the question, and `p1-12` no longer produces a backwards-in-time path.

C2 says the question must be **dropped**, not kept as text: feeding it into the
BM25 query lowers both V@1 and MRR.

C3 isolates the attribute-merge rule. Merging costs an independent scoring row,
so the video rank drops (C3 wins V@5/10/20/50); in exchange the DP is not split
in two, so frame localization improves (F@50 21 vs 19). C1 is chosen because the
P0 bottleneck is F, not V.

To rerun the old version: `--legacy-planner`.

## P2-LLM, 2026-09-04 — negative, not used

Qwen3-8B splitting the query vs the rule-based planner, both under legacy A0.
The generator script and its `queries_llm.yaml` output were deleted after this
result; only `artifacts/p2_llm_results.jsonl` remains as evidence.

| planner | MRR | V@1 | V@5 | V@20 | F@20 | F@50 |
|---|---|---|---|---|---|---|
| rules | 0.474 | 17 | 33 | 45 | 19 | 21 |
| Qwen3-8B | 0.439 | 15 | 29 | 43 | 18 | 22 |

The LLM genuinely changed 37/55 queries — it was not blocked by the validator.
Paired test over 52 queries: MRR delta -0.035, bootstrap 95% CI [-0.098, +0.017],
sign test 11/11 p=1.00. The CI contains 0, so the correct conclusion is that the
LLM is **no better** than the rule planner, not that it is worse. Suspected
cause: the LLM splits more finely than the rules (1:23 2:11 3:8 4:8 5:4 6:1 vs
1:24 2:16 3:6 4:7 5:2), and every extra event is one more chance for the DP to
misalign. Do not build an LLM route on the server.

## P3 ladder, 2026-09-04 — adaptive_p0 beats legacy

| run | what it adds | MRR | V@1 | V@10 | F@1 | F@20 | F@50 | |
|---|---|---|---|---|---|---|---|---|
| A0 | legacy | 0.477 | 21 | 35 | 9 | 16 | 17 | |
| A1 | 7 flat components | 0.570 | 23 | 45 | 9 | 18 | 18 | ⚠ |
| A2 | + robust calibration | 0.554 | 23 | 42 | 10 | 19 | 19 | ⚠ |
| A3 | + confidence gating | 0.556 | 23 | 43 | 9 | 19 | 19 | ⚠ |
| A4 | + interval projection | 0.556 | 23 | 43 | 9 | 19 | 19 | |
| A5 | + event routing | 0.560 | 23 | 42 | 11 | 21 | 21 | ship |

⚠ = **measured with interval projection ON even though that rung is defined as
having it off.** `setup.py:206` never passed the flag, and `asr_projected.py`
built `segment_coverage_*` once in `__init__`, so the real value was always
`True`. Fixed into a property that rebuilds coverage. The consequence is
counter-intuitive: **A4/A5 are the two correct runs, A1-A3 are the contaminated
ones** — and the "A4 adds interval projection" column measured nothing, A4 was
byte-identical to A3 (0/55 queries differ). A clean ladder requires rerunning
A1 A2 A3.

A0 here is the 2026-09-04 remeasurement (`artifacts/p5_asr_restored.jsonl`) after
the patch. The old table recorded `MRR 0.474 / V@1 17 / V@10 38`: that was A0
running *with* interval projection. MRR barely moved, the distribution shifted a
lot.

A5 vs A0: 23 queries better, 10 worse, 19 tied. Big wins (`p2-1` 204→1, `p2-3`
156→29), small losses. MRR delta +0.083, sign test p=0.035 **clears the 95%
threshold**, but the bootstrap 95% CI [-0.022, +0.190] **still crosses 0** — the
sign is consistent enough, the magnitude is not pinned down at n=52.

Before fixing `q_high=0.95`, adaptive only reached MRR 0.171: the quantile clip
tied 23,500 top-of-table frames at the same score, and ranking collapsed back to
array position. That bug is the **entire** 0.171→0.560 gap, not the credit of
any adaptive rung.

Read it correctly: **F@50 is tied at 21/52**. Adaptive finds no additional query,
it pushes already-found ones higher. The real ceiling is the 31/52 queries that
never have a correct frame in the top 50.

## P4, 2026-09-04 — the harness ignored `baseline.yaml`

`ablation.py:to_hybrid_config` built a fresh `HybridTemporalConfig` from defaults,
so runs A0-A5 **never read** `configs/baseline.yaml`. Found by dropping
`bm25_asr` to 0 and getting byte-identical output. Fixed to
`base.model_copy(update=...)`, keeping the deployed weights.

Remeasured after the patch - turning the ASR field off hurts both modes:

| | MRR | |
|---|---|---|
| legacy `asr_weight` 1.0 → 0.0 | 0.477 → 0.444 | V@1 21→19 |
| adaptive `bm25_asr` 0.06 → 0.0 | 0.560 → 0.514 | V@1 23→21 |

Paired on A5: 15 queries better, 9 worse, 28 tied. So the P1 conclusion ("the asr
field is noise, drop it to 0") is **refuted** - it was measured under the old
planner. Keep `asr_weight = 1.0` and `base_component_weights.bm25_asr = 0.06`.

`configs/baseline.yaml` now matches the code defaults value for value, and A5
reproduces 0.560 exactly across three independent runs via three different config
paths.
