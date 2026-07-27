# KISC conversation resolver

This package converts bounded conversation context into one complete
`ConversationState`. It does not retrieve frames, call search tools, or manage
session persistence.

## Ownership

- `resolver.py` owns prompt instructions, structured-output validation, and
  newest-wins feedback checks.
- `hcmai.common.schemas.ConversationState` is shared because the resolver
  produces it for search orchestration.
- `hcmai.kisc.KiscSessionManager` continues to own session turns and cumulative
  frame feedback.

The provider is injected as one `structured_call`, which keeps model loading
outside the resolver and makes offline tests deterministic.

## Verification

```bash
PYTHONPATH=src aic/bin/pytest tests/test_conversation_resolver.py
```
