# Language-Free Multilingual KIS Design

**Date:** 2026-09-17
**Status:** Approved for planning

## Problem

KIS currently asks the LLM to return `language` and stores that value in every
semantic intent. Scoped updates fail when an OpenAI-compatible provider returns
valid event JSON but omits the redundant top-level field. The runtime also uses
the field to decide whether to translate event text before dense retrieval,
even though the active retrieval stack is multilingual.

The result is a fragile split contract: initial resolution already tolerates a
missing language, while scoped resolution rejects it before the server can
apply the update.

## Decision

KIS becomes language-free and passes canonical event text directly to both
multilingual dense retrieval and BM25.

Remove `language` from:

- `KISResolution`;
- `ScopedResolutionBatch`;
- `GlobalRewriteResolution`;
- `KISIntent`;
- all KIS LLM prompts and generated JSON schemas;
- KIS orchestration, readiness wiring, and model-authored validation.

`query_text` remains the canonical aggregate text. It is present whenever any
event contains text and absent only for image-only intents.

## Requirements

- **REQ-001:** New KIS intent and LLM response schemas must not expose or
  require a `language` field.
- **REQ-002:** A legacy serialized `KISIntent` containing `language` must still
  parse, while its new serialization must omit that key.
- **REQ-003:** Initial, scoped, and global semantic operations must build valid
  canonical intents from model output that contains no language metadata.
- **REQ-004:** KIS dense and BM25 retrieval views must use the canonical event
  text directly for every language.
- **REQ-005:** KIS runtime setup, search, and readiness must not construct,
  require, or invoke event translation.
- **REQ-006:** KIS latency responses must retain `translation_ms` for public
  compatibility and report it as `0.0`.
- **REQ-007:** Existing canonical identity, event authorization, entity binding,
  topology, and image-only invariants must remain unchanged.

## Runtime Flow

Initial, scoped, and global semantic operations return only semantic content.
The server continues to own revision, event IDs, temporal edges, image refs,
and canonical identity.

For retrieval, each text-bearing event is projected as:

```text
canonical_text = event.text
dense_text     = event.text when dense retrieval is enabled
bm25_text      = event.text when BM25 is enabled
```

No request-time translation is performed. Image-only events continue to carry
`None` for all text views.

## Compatibility

New API responses and stored snapshots do not emit `language`.

`KISIntent` accepts and discards a legacy input-only `language` key so replayed
browser state and historical request payloads do not fail with HTTP 422 during
the migration. This compatibility behavior does not expose a runtime field and
does not include `language` in new serialization or JSON Schema.

The shared `SearchLatency.translation_ms` response field remains temporarily
for public response compatibility and is always `0.0` for KIS. Removing that
shared latency field is outside this migration because it also affects non-KIS
contracts and historical clients.

The generic translation module and its focused unit tests remain in the
repository, but KIS setup no longer constructs or injects it and KIS search no
longer calls it. Deleting generic translation code and configuration is a
separate cleanup.

## Error Handling and Invariants

- Model output containing an unexpected `language` field is rejected by the
  existing `extra="forbid"` policy; prompts and schemas no longer request it.
- Text-bearing intents require non-blank `query_text`.
- Image-only intents require `query_text=None`.
- Scoped updates continue to validate authorized event IDs, entity bindings,
  contiguous event extension, and canonical temporal edges.
- Global rewrites continue to preserve topology and image-only event semantics.

## Test Strategy

Use a red/green migration:

1. Contract tests prove new model schemas do not contain `language`, new
   serialization omits it, and legacy `KISIntent` input still parses.
2. Resolver tests prove initial and scoped LLM responses without `language`
   produce valid intents.
3. Rewriter tests prove global rewrite responses no longer require or compare
   language metadata.
4. Orchestration tests prove Vietnamese and English text are both sent directly
   to dense/BM25 views and no translator is required.
5. Setup and health tests prove KIS runtime composition and readiness no longer
   expose translation as a required runtime capability.
6. Focused KIS, API, orchestration, and EventTrail suites run before the broad
   Python suite.

## Non-Goals

- Changing embedding checkpoints or index artifacts.
- Claiming a retrieval-quality improvement without an HCMAI experiment.
- Deleting the generic translation package or its configuration.
- Changing canonical frame/video identity or temporal alignment behavior.
