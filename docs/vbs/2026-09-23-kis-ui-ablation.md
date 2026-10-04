# KIS UI baseline setup (exploratory)

## Conditions

Both apps use the same backend, indexes, model, retrieval controls, and ten
frozen KIS queries from `configs/evaluation/ui_ablation_10_kis.json`.

| Condition | Frontend | Local port | Interaction |
| --- | --- | --- | --- |
| A | `frontend_baseline/` | 3001 | KIS search, result grid, frame inspector; no Query Hypothesis or EventTrail UI/session calls |
| B | `frontend/` | 3000 | Full Query Hypothesis and EventTrail UI |

The baseline uses the existing stateless KIS `initial_resolve` operation on the
first query, then regular KIS operations. The full UI opens a Query Hypothesis
session first. This difference is part of the tested workflow; record request
latency separately before attributing any difference to the UI alone.

## Run both apps

Use two terminals from the repository root:

```bash
cd frontend && PORT=3000 npm start
cd frontend_baseline && PORT=3001 npm start
```

Both `.env` files must point to the **same** `REACT_APP_API_BASE_URL` and
`REACT_APP_STREAM_API_BASE_URL`. `frontend_baseline/node_modules` is a local
symlink to `frontend/node_modules`; for a standalone install, run `npm ci` in
`frontend_baseline` instead. The two ports give the apps separate browser
storage. If cookies or DRES sessions are used, check that both point to the
same evaluation task before an attempt.

## Manual run sheet

Freeze a time budget (suggested: 300 seconds) and the operator/query order
before opening the ground truth. Read each query from
`artifacts/query/002/<query_id>.txt`. For each attempt, record one row:

```text
run_id,participant_id,query_id,condition,order_group,seen_before,time_budget_ms,answer_video_id,answer_timestamp_ms,success,completion_ms,stop_reason,interaction_count,retrieval_calls,notes
```

Start the timer when the query becomes visible. Stop when the operator chooses
one final frame or the time budget expires. Keep failures and timeouts in the
denominator. Compare the final `video_id` and timestamp to
`artifacts/evaluation/query_002_windows.json` after the attempt. Record exact
and ±5-second window hits separately in analysis; do not expose labels during
the run. Count explicit clicks/edits as interactions and count KIS searches as
retrieval calls. Save raw rows before computing summaries.

If one operator sees the same target in both conditions, mark `seen_before=true`
for the second attempt and treat the comparison as exploratory. For a cleaner
comparison, assign different people or disjoint, difficulty-matched query sets
to each condition and counterbalance condition order.

Report completion counts and denominators, censored completion time, and
interaction counts per condition. This setup does not itself establish an
accuracy or usability improvement.
