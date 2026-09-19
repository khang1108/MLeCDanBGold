# Task 4 backend cleanup report

## Scope

Removed the unreachable legacy feedback actions `EditIntentAction`,
`RestructureAction`, `AnchorAction`, and `RejectCandidateAction` from the
feedback models and package exports. Removed their unreachable service and
resolver branches. Canonical query changes remain proposal-only through
`QueryEditProposalAction`; refinement, repair, and clarification behavior are
unchanged.

Updated directly related feedback tests to use the active action union and
removed tests for actions that are no longer valid feedback actions.

## Verification

- `aic/bin/python -m pytest -q tests/test_kis_feedback_resolver.py tests/test_kis_feedback_execution.py tests/test_kis_feedback_state.py tests/kis/test_feedback_proposals.py`
  - **17 passed** in 0.62s.
- `tests/test_kis_feedback_api.py` was attempted with a 20-second timeout but
  did not complete or emit test output; no API-test result is claimed.

## Notes

The package `__init__.py` was updated because retaining imports of removed
models would make `hcmai.kis.feedback` fail during import.
