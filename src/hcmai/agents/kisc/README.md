# KISC conversation resolver

This package converts bounded conversation context into one complete
`ConversationState`, then optionally composes that resolver with the shared
search engine for one stateless API turn.

## Ownership

- `resolver.py` owns prompt instructions, structured-output validation, and
  newest-wins feedback checks.
- `agent.py` owns the bounded resolve-once, search-once composition. Provider
  or validation failures abort the turn.
- `hcmai.common.schemas.ConversationState` is shared because the resolver
  produces it for search orchestration.
- `hcmai.agents.kisc.KiscSessionManager` continues to own session turns and cumulative
  frame feedback.

The provider is injected as one `structured_call`, which keeps model loading
outside the resolver and makes offline tests deterministic.

The public `POST /api/v1/kisc/search` contract keeps history and prior state in
the client. Legacy in-memory session endpoints remain available for backward
compatibility.

## Verification

```bash
PYTHONPATH=src aic/bin/pytest tests/test_conversation_resolver.py
```
