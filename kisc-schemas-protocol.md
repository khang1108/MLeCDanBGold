# KISC Schemas and Protocol

## Goal

Extend the existing conversation/search contracts and session manager into one
explicit, testable KISC protocol without adding a parallel schema layer.

## Tasks

- [x] Extend `conversation.py` with typed roles, turn timestamps/replies,
  persisted `problem_id`, and conflict-safe feedback. → Verify invalid roles
  and overlapping feedback raise `ValidationError`.
- [x] Extend `SearchRequest`/`SearchResponse` with KISC context validation and
  distinct user/assistant turn IDs. → Verify stateless feedback and partial
  conversational responses are rejected.
- [x] Refine `KiscSessionManager` to require an existing session, apply
  latest-decision-wins feedback, promote accepted results, reject unwanted
  results, and renumber ranks. → Verify a two-turn fixture.
- [x] Align FastAPI error handling and existing schema documentation with the
  protocol; do not add endpoints or schema files. → Verify unknown sessions
  return HTTP 404.
- [x] Add focused schema/KISC smoke tests and run compile, Pyright, and pytest.
  → Verify modified modules compile and focused tests pass without models.

## Done When

- [x] Session history, feedback transitions, ranked results, and turn
  correlation serialize through the existing public contracts.
