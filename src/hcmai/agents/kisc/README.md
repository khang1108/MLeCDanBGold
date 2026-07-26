# KISC agent — AI2-04 bounded conversation resolver

This package owns the pure provider-independent resolver in
[`resolver.py`](resolver.py). Branch `feat/ai2-conversation-resolver` also
introduces the canonical interpreted-state contract while leaving raw session
management in `hcmai.kisc`. The literal AI2-04 resolver core and baseline are
complete, but real provider quality and KISC/API integration are not ready.

## 1. Task identity

| Field | Value |
|---|---|
| Task ID | AI2-04 |
| Owner | Khầy |
| Workstream | KISC resolution |
| Priority | P0 |
| Task-board status | Complete |
| Task | Implement bounded KISC conversation resolver |
| Branch | `feat/ai2-conversation-resolver` |
| Base | `main@47ebe06492a917749d7c16523b484df5be5a568f` |
| Implementation commit documented | `e9c8222f67a6e01ab1082a3200d7737b5d48504e` |
| Canonical resolver | [`src/hcmai/agents/kisc/resolver.py`](resolver.py) |
| Canonical schemas | [`src/hcmai/common/schemas/conversation.py`](../../common/schemas/conversation.py) |
| Test | [`tests/test_conversation_resolver.py`](../../../../tests/test_conversation_resolver.py) |

The task board requires ordered history/current message/feedback/optional prior
state to become a standalone query plus positive, negative, and uncertain
constraints. It requires pronoun/correction resolution, one structured model
call, deterministic parse fallback, 20 fixtures, selected/rejected behavior,
empty history, and no retrieval inside the resolver.

The stale task-board path `src/aic/agents/kisc/resolver.py` maps directly to
`src/hcmai/agents/kisc/resolver.py`. The task-board fallback wording conflicts
with a later Tech Lead decision: provider/parse failure raises a bounded typed
error and returns no fabricated state. The Task Board fallback criterion is
**OVERRIDDEN BY TECH LEAD**; the implemented behavior is a typed error with no
fabricated `ConversationState`.

## 2. Branch purpose

This branch owns a provider-independent, bounded interpretation boundary. A
caller supplies ordered context, the current message, current feedback, and
optional previous `ConversationState`; one injected structured call returns a
complete validated state.

It does not own server-side session CRUD, memory selection, provider routing,
vLLM/LiteLLM adapters, retrieval, `SearchRequest`, `SearchEngine`, API routes,
or KISC application orchestration.

## 3. Implemented

- [`ConversationState`](../../common/schemas/conversation.py)
  holds a standalone query, ordered positive/negative/uncertain constraints,
  and disjoint accepted/rejected frame IDs. Lists deduplicate while preserving
  order.
- `ConversationTurn` remains one raw ordered event and
  `ConversationSession` remains raw server-side session state.
- [`ConversationResolver`](resolver.py) accepts
  all bounded context explicitly and invokes one injected callable exactly
  once.
- The structured request asks for a complete state, not a delta; it defines
  accumulation, newest-wins corrections, polarity, uncertainty, and feedback.
- Current feedback removes a newly accepted frame from rejected state and a
  newly rejected frame from accepted state.
- Malformed, incomplete, schema-invalid, or provider-failed output raises
  bounded `ConversationResolverError`; there is no retry or fallback state.
- The implementation imports no provider SDK, search, retriever, session
  manager, FAISS, tool loop, or ReAct framework.
- Public imports are exposed from `hcmai.agents.kisc`.

## 4. Not implemented or incomplete

- No real LiteLLM or vLLM adapter exists.
- No approved model/checkpoint/prompt has been evaluated semantically.
- The 21 fixture cases use fake complete structured outputs; they prove
  orchestration, not real pronoun/correction interpretation.
- A reproducible 21-case contract baseline exists locally under the required
  ignored `runs/kisc_resolver_baseline/` path.
- [`KiscSessionManager`](../../kisc.py) does not call this
  resolver.
- API/session/search integration is absent.
- The task board still says deterministic fallback, while the implemented
  Tech Lead policy is typed error/no state. Literal task-board completion needs
  an approved update.

## 5. Verification evidence

### Engineering evidence

| Evidence type | Command or artifact | Result | Proves | Does not prove |
|---|---|---|---|---|
| Offline/regression tests | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_conversation_resolver.py tests/test_schema.py tests/test_kisc.py` | 30 passed | Schema, one call, complete output, feedback, errors, no resolver retrieval, KISC regressions | Real LLM interpretation |
| Contract fixtures | `runs/kisc_resolver_baseline/` | 21/21 fake structured-call cases pass | Deterministic orchestration, validation, one call, and no retrieval | Provider/model quality |
| Source inspection | Resolver imports and request | No search/session/provider dependency | Ownership boundary | Integrated system behavior |

### Quality evidence

There is no real-provider output. Fake callers return fixture-specific expected
states, so semantic quality, prompt adherence, Vietnamese pronouns, and
correction reliability remain unverified.

### Integration evidence

The existing session manager and API continue their current paths and do not
instantiate or call `ConversationResolver`. No search request is constructed
by this branch.

## 6. Artifacts

Task-board-required local, Git-ignored output:

```text
runs/kisc_resolver_baseline/
├── config.json
├── metrics.json
├── per_case.json
└── README.md
```

It records the source commit, Python version, 21 fixture results, exactly-one-
call and no-retrieval checks, test command/exit code, and the Tech Lead
fallback override. Repository policy ignores `runs/`, so this generated
evidence is retained locally and is not available after a fresh clone. It is
contract evidence, not a real-provider quality benchmark.

## 7. Dependencies and cross-team contracts

| Dependency | Owner/task | Path or symbol | Use | Readiness | Blocking? | Modified here? |
|---|---|---|---|---|---|---|
| Conversation contracts | Tech Lead; minimal AI2-approved extension | [`ConversationTurn`, `FrameFeedback`, `ConversationState`](../../common/schemas/conversation.py) | Typed input/output | Available on branch | No for core | Minimal state addition only |
| Session manager | Tech Lead | [`KiscSessionManager`](../../kisc.py) | Future context/persistence caller | Exists; not integrated | Yes for system use | No |
| KISC protocol | Tech Lead | Shared one-call/no-ReAct decisions | Failure and ownership semantics | Partly frozen; task board conflicts | Yes for acceptance | No |
| KISC API | SWE | [`src/hcmai/app.py`](../../app.py) | Future request/provider wiring | Not integrated | Yes | No |
| Search boundary | Tech Lead | [`SearchEngine`](../../search.py) | Downstream consumer outside resolver | Exists; intentionally uncoupled | Yes for end-to-end KISC | No |
| Provider/model | Tech Lead / SWE | vLLM or LiteLLM not finalized | One structured call | Missing | Yes for quality | No |

Referenced team components are not modified by the resolver implementation.
The `ConversationState` addition is a separate, minimal canonical schema commit.

## 8. Current quality status

- AI2-04 literal core status: **COMPLETE**.
- Engineering: **PASS** — typed state, one-call orchestration, validation,
  bounded errors, feedback rules, no-retrieval boundary, and all 21 baseline
  cases are verified.
- Real-provider semantic quality: **NOT VERIFIED** — no real approved
  provider/model has been tested.
- KISCAgent/API integration: **OUTSIDE AI2-04 AND NOT COMPLETE**.

## 9. Merge readiness

| Field | Decision |
|---|---|
| Merge target | `main`, after protocol/provider and schema review |
| Current readiness | **READY FOR REVIEW — NOT READY TO MERGE** |
| Blocking conditions | Provider choice; real semantic suite; KISC/API integration contract |
| Required approvals | Tech Lead for schema/protocol/provider; SWE for API integration; AI2 for semantic evidence |
| Downstream usage | Standalone contract testing only |

Ready for review does not mean ready to merge.

## 10. Manual acceptance procedure

1. Tech Lead selects the real vLLM/LiteLLM provider, checkpoint, structured
   output mechanism, and bounded context policy.
2. Run at least 15 cases, including empty history, pronouns, accumulation,
   direct correction, positive-to-negative and negative-to-positive changes,
   uncertainty, contradiction, accepted frame, accepted-then-rejected,
   rejected-then-accepted, malformed output, provider failure, one-call proof,
   and no-retrieval proof.
3. Record:

   ```text
   case_id | history | current_message | feedback | previous_state
   expected_state | actual_state | one_call | no_retrieval
   stale_constraint | incorrect_polarity | hallucinated_constraint
   missing_frame_feedback | verdict | notes
   ```

4. Use complete expected states approved before running the model. For every
   successful case, verify standalone query, all three constraint lists, both
   feedback lists, and absence of stale contradictions.
5. For malformed/provider failures, verify one bounded
   `ConversationResolverError`, zero retry, unchanged previous state, and no
   search/session side effect.
6. PASS only with 100% schema/feedback/one-call/no-retrieval compliance, zero
   stale or hallucinated constraints, and at least 90% semantic case accuracy.
   Any retrieval call, second provider call, stale corrected constraint, or
   wrong accepted/rejected state is a blocking FAIL.
7. Preserve approved fixtures, provider/config revision, per-case output,
   call counts, failures, and signed verdicts under ignored
   `runs/kisc_resolver_baseline/`.

## 11. Known risks

- Fake structured outputs can make orchestration look semantically complete.
- Provider JSON/structured-output differences may alter failure behavior.
- Backend memory/history selection is undefined outside this pure component.
- The task-board fallback conflict can cause review disagreement.
- Integrating resolver and session state can accidentally duplicate memory or
  introduce hidden retrieval/tool calls.

## 12. Next actions

1. **Tech Lead:** update the task board for typed error/no-state failure and
   freeze provider/context protocol.
2. **Tech Lead / SWE:** define where session context is bounded and where the
   provider is constructed.
3. **AI2 owner:** implement the selected thin provider adapter in a separate
   reviewed scope and run the real semantic suite.
4. **SWE / Tech Lead:** integrate resolver output with KISC/search without
   moving retrieval into the resolver.
5. **AI2 + Tech Lead:** retain permanent run evidence and approve quality
   before merge.
