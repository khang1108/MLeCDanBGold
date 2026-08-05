# KISC conversation resolver

This package converts bounded conversation context into one complete
`ConversationState`, then optionally composes that resolver with the shared
`SearchService` for one stateless API turn.

## Ownership

- `resolver.py` owns prompt instructions, structured-output validation, and
  newest-wins feedback checks.
- `agent.py` owns the bounded resolve-once, search-once composition through
  `hcmai.orchestration.pipeline.SearchService`. Provider
  or validation failures abort the turn.
- `hcmai.common.schemas.ConversationState` is shared because the resolver
  produces it for search orchestration.
- `hcmai.agents.kisc.KiscSessionManager` can own bounded in-memory session
  turns and cumulative frame feedback in research experiments.

The provider is injected as one `structured_call`, which keeps model loading
outside the resolver and makes offline tests deterministic.

The KISC router exists as research code but is not mounted by `hcmai.app`.
There is therefore no public KISC HTTP endpoint in the current application;
the frontend owns conversation state. Mounting the router is an explicit future
integration decision, not backward-compatibility behavior.

## Verification

```bash
PYTHONPATH=src aic/bin/pytest tests/test_conversation_resolver.py
```
