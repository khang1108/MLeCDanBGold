# Scoped temporal feedback: preregistered evaluation and claim boundaries

**Status:** protocol to freeze before inspecting condition C results. Current
repository evidence is functional only; it is not a user study or an accuracy,
latency, usability, or novelty result.

## 1. Research question and conditions

The question is whether retained, event–interval feedback reduces the effort
needed to locate the required moments, compared with the same local controls
where the user must re-enter conditions. The primary contrast is **B–C**.

| Condition | Definition | Role |
| --- | --- | --- |
| A | Existing retriever, player, and temporal context. | Contextual baseline only if time permits. |
| B | A plus local video/interval search and the full constraint expressivity available to C; conditions are re-entered when needed. | Fair control for local search. |
| C | B plus retained event–interval feedback and undo. | Proposed mechanism. |

B must not be restricted to one constraint or otherwise weakened. Keep
retriever, scoring, data, model availability, and human time budget identical.
C winning A alone cannot isolate retained feedback from local search.

## 2. Preregistration and dataset controls

Before collecting outcome data, freeze the dataset/query assignment, query
wording, condition order, scoring revision, and analysis script revision. Do
not choose queries after seeing C outcomes. Record baseline metadata exactly as
currently available: **30 queries total: 19 KIS, 9 QA, 2 TRAKE**. The main
claim target is KIS. Include successes and failures, not only the four known
KIS moment-localization cases (`query-p2-14-kis.txt`, `query-p2-18-kis.txt`,
`query-p2-3-kis.txt`, `query-p2-6-kis.txt`). Those four are a failure-analysis
subset, not an interaction benchmark.

Known-answer queries used for development or demo must be marked separately
and excluded from confirmatory participant claims. Self-created temporal
queries are **synthetic development queries**. TRAKE is an internal task, not
a VBS task; never infer HCMAI benchmark accuracy as VBS accuracy, and never
infer HCMAI accuracy as V3C accuracy.

Choose one equal human time budget before collection and write its integer
value into every run. This protocol intentionally does not prescribe a VBS
rule value. A failure remains in the denominator. Report completion rate and
censored completion times; do not report success-only mean time as the primary
effort result.

## 3. Assignment and difficulty

Do not have one participant solve the same known target in B and C as an
unbiased paired effect. Use different, difficulty-matched queries across
conditions; counterbalance condition order and query blocks where feasible.
Each condition must receive easy and hard examples. Freeze assignment before
looking at C results.

Difficulty is assigned from pre-study baseline metadata only:

1. baseline top-1 video hit/rank; and
2. score gap between the best and runner-up candidate.

Store the rule and thresholds used to make easy/hard bins. Do not tune them on
participant outcomes. A developer-only or otherwise small sample is
exploratory: make no statistical-significance or general-usability claim.

## 4. Run record and operational definitions

One row represents one participant–query–condition attempt. Preserve failed and
timed-out attempts. Use the following exact columns, with no silent aliases:

```text
run_id, participant_id, query_id, condition, order_group,
seen_before, dataset_revision, scoring_revision, time_budget_ms,
success, completion_ms, stop_reason,
condition_reentries, confirmed_evidence_revisits,
retrieval_calls, scoring_ms, decode_ms, interaction_count
```

Operational definitions:

- `success`: preregistered answer/moment criterion is met before the deadline;
  define the criterion before collection and apply it identically to B and C.
- `completion_ms`: elapsed wall-clock time from task start to success or
  stopping; includes waiting and interaction time. For failures, record the
  censor/deadline value and `stop_reason`, never a fabricated completion.
- `condition_reentries`: count of re-entering an event–interval–polarity that
  was already expressed earlier in that task. A new condition is not a
  re-entry.
- `confirmed_evidence_revisits`: count only explicit reopening of an interval
  after it was confirmed. Do not infer revisits from autoplay frames, passive
  playback, or total clicks. If logging cannot support this definition,
  drop this metric rather than estimate it.
- `retrieval_calls`: calls to retrieval/scoring services; report separately
  from decoder work.
- `scoring_ms` and `decode_ms`: measured stage times, not theoretical
  complexity. Report branch-open separately from warm feedback mutation.
- `interaction_count`: preregistered explicit user actions; define the event
  vocabulary in the logger before collection and use it for all conditions.
- `seen_before`: whether the participant had previously seen the query/answer
  material; do not mix such rows into an unlabelled primary analysis.

Primary outcomes are completion rate and censored completion time. Effort
secondary outcomes are re-entries and confirmed-evidence revisits when
measurable; clicks alone are not a sufficient effort measure because waiting
time can dominate.

## 5. Technical evidence gates

Before any human session, record pass/fail evidence for each gate:

1. Task 5 replay passes end to end, including confirm → reject → re-solve →
   path diff → undo.
2. Feedback mutations do not invoke the scorer when the cached score matrix is
   valid; scorer count is exactly zero for those mutations.
3. Branch-open latency is reported separately from warm feedback latency.
4. One no-feedback baseline regression run passes before and after the
   integration change.
5. Canonical IDs/timestamps, event scope, revision guards, unknown-vs-negative
   semantics, and no-path status distinctions remain intact.

These gates support mechanism readiness only. Do not run a model sweep or use
GPU time to hide a semantics defect. A failed gate blocks human interpretation
until fixed and rerun.

## 6. Analysis and claim table

Use counts and denominators for every condition; report KIS separately from
QA and internal TRAKE. No claim below may exceed its evidence row.

| Available evidence | Allowed claim | Forbidden claim |
| --- | --- | --- |
| Functional tests or a scripted demo only | Mechanism supports scoped constraints, revisioned undo, and the demonstrated state transitions. | No proof of effort reduction, user benefit, latency, accuracy, significance, or general usability. |
| Small favorable human B–C comparison | Exploratory evidence for the stated sample, query set, assignment, and setting. | No broad VBS/general-user claim or unqualified improvement claim. |
| C beats A, without B–C | The complete local workflow may help relative to A. | No isolated retained-feedback effect. |
| B–C shows no improvement | Report the null/limitation and inspect controllability, re-entry, and undo costs. | No fabricated gain or selective success-only explanation. |
| Simulated feedback policy only | Decoder/constraint effectiveness under that simulated policy. | No human-study or behavioral claim. |

The existing baseline report supports contextual retrieval/decoder facts only:
30-query composition, FF-MDP KIS figures, and known sampling/latency limits.
It does not measure B–C interaction. Do not call `dante_style_unary_dp` a full
DANTE reproduction, and do not treat one-pass latency as a general speed
claim.

## 7. Paper/demo package

Package only artifacts that actually exist and label each as functional,
benchmark, or human-study evidence:

- screenshot or annotated sequence showing event selection, interval
  constraint, revised path, undo, and candidate handoff;
- one pipeline diagram: query → retrieval → local branch → scoped feedback →
  full constrained decode → candidate handoff;
- benchmark table with task, query count (`KIS 19 / QA 9 / TRAKE 2`), split,
  scoring revision, and metric definitions;
- B–C table only if human data was actually collected, with n, assignment,
  failures, censoring, and the frozen time budget;
- limitations: small data, developer participants (if applicable), frame
  sampling, one local branch, no UI/transport completion unless separately
  verified, and no novelty proof.

Do not put agent or multiagent behavior in the title or contribution unless it
is implemented and evaluated. Existing design research supports a hypothesis;
it does not establish “first,” novelty, or VBS improvement.

## 8. Sources and current boundary

This protocol cites the repository design and plan: sections 3–4 and 14–16 of
`docs/superpowers/specs/2026-09-13-vbs-scoped-temporal-feedback-design.md`,
`docs/superpowers/plans/2026-09-13-vbs-scoped-temporal-feedback-plan.md`,
`docs/vbs/scoped-feedback-handoff.md`, and
`artifacts/baselines/baseline_report.md`. No new literature survey is needed.

As of this checkout, the evidence boundary is functional core/replay material
and existing offline baseline artifacts. There is no verified B–C human result,
participant sample, VBS-rule time budget, completed UI/transport rehearsal, or
new benchmark artifact in this document.
