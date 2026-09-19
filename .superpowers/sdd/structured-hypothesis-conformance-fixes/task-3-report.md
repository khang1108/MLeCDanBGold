# Task 3 report — Backend provenance, mode intervals, and status mapping

## Implementation

- Merges now derive source provenance from the exact `intent.query_text[start_char:end_char]` slice, preserving punctuation and whitespace in merged text and provenance. If the child spans cannot be verified against the canonical query, the merged event is downgraded to `user_override` without provenance.
- Added origin/provenance checks for explicit source-grounded events and user-added events. User overrides may retain provenance. Source children produced by a split are downgraded when a valid child span cannot be reconstructed.
- Mode intervals now use chronological neighboring midpoint boundaries under inclusive mask semantics: the later interval starts one millisecond after the shared midpoint, while retaining nearest-competitor radius bounds, domain clipping, input ordering, and current-mode matching.
- `/api/v1/kis/search` maps `QUERY_HYPOTHESIS_NOT_FOUND` to 404 and `QUERY_HYPOTHESIS_EXPIRED` to 410.

## Files changed

- `src/hcmai/kis/hypothesis/mutations.py`
- `src/hcmai/kis/models.py`
- `src/hcmai/event_trail/decoding/decoder.py`
- `src/hcmai/api/routers/kis.py`
- `tests/kis/test_query_hypothesis_mutations.py`
- `tests/kis/test_models.py`
- `tests/event_trail/test_hypothesis_modes.py`
- `tests/api/test_kis_router.py`

Pre-existing deletions of `scripts/dres_control.py` and `scripts/import_evaluation_to_dres.py` were preserved.

## TDD evidence

### RED

Command:

```text
aic/bin/python -m pytest -q tests/kis/test_query_hypothesis_mutations.py tests/kis/test_models.py tests/event_trail/test_hypothesis_modes.py
```

Observed output: **6 failed, 31 passed**. The new failures covered whitespace-preserving merge provenance, unsafe source downgrade, invalid origin/provenance combinations, and inclusive midpoint overlap. Failures were assertion failures, not import or syntax errors.

API hang investigation:

```text
timeout 15s aic/bin/python -m pytest -q tests/api/test_kis_router.py
timeout 15s aic/bin/python -m pytest -q tests/api/test_query_hypothesis_api.py
```

Both exited **124** without completing the first POST. A minimal reproducer also stalls after `before`:

```text
timeout 10s aic/bin/python - <<'PY'
import asyncio
from unittest.mock import Mock
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from httpx import ASGITransport, AsyncClient
app = FastAPI(); service = Mock()
@app.post('/x')
async def x():
    print('before', flush=True)
    await run_in_threadpool(service.search_kis, {'a': 1})
    print('after', flush=True)
    return {'ok': True}
async def main():
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as c:
        print((await c.post('/x')).status_code, flush=True)
asyncio.run(main())
PY
```

This reproducer hangs outside the owned router logic, so no infrastructure change was made.

### GREEN

Focused backend command:

```text
aic/bin/python -m pytest -q tests/kis tests/event_trail/test_hypothesis_modes.py tests/api/test_kis_router.py::test_search_kis_maps_query_hypothesis_not_found_and_expired
```

Observed output: **64 passed in 1.41s**.

The API regression itself was also run independently:

```text
aic/bin/python -m pytest -q tests/api/test_kis_router.py::test_search_kis_maps_query_hypothesis_not_found_and_expired
```

Observed output: **1 passed in 1.25s**. The test replaces only the broken test-environment threadpool boundary with a direct async shim so the status mapping is exercised.

Additional verification: `git diff --check` passed.

## Concerns / limitations

- The two existing API modules remain blocked by the environment’s `anyio` worker-thread behavior; changing it would exceed Task 3 ownership.
- The model validator preserves compatibility for legacy events that omit an explicit `origin`; explicitly declared `origin="source"` text still requires provenance. This avoids breaking existing image/text fixtures and constructors outside Task 3’s ownership boundary.
- Duplicate peak timestamps cannot yield two disjoint inclusive intervals while both intervals contain the same representative timestamp; current alternative generation already separates peaks by its configured minimum separation.
