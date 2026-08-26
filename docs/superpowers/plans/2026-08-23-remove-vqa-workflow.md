# Retire VQA and VKIS, Keep KIS + TRAKE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove VQA answer generation and all VQA runtime surfaces while leaving a correct, testable KIS + TRAKE application.

**Architecture:** Keep the existing backend composition facade, KIS/TRAKE workflow locations, temporal facade, shared evidence stores, and shared inference gateway. Delete VQA-only contracts and providers, reduce the task registry to KIS/TRAKE, and scope frontend changes to removing the VQA endpoint client/docs and its direct callsite. Keep the current shared KIS/TRAKE workspace, including its existing layout and presentation code, in this migration.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, PyYAML, pytest, React, Jest/React Testing Library, LiteLLM pass-through configuration.

**Spec:** `docs/superpowers/specs/2026-08-23-kis-trake-only-runtime.md`

## Global Constraints

- The active supported task set is exactly `KIS` and `TRAKE`; remove `VQA` and `VKIS` from runtime contracts and capabilities.
- Remove `/api/v1/vqa`, `/v1/vqa`, and `/v1/vqa/multi`; do not leave a false-positive VQA capability or permanently unsupported VQA pipeline.
- Keep shared embeddings, captioning, OCR, ASR/diarization, retrieval, evidence stores, optional KIS reranking, and canonical frame materialization.
- Preserve `video_id`, `frame_id`, `frame_idx`, `timestamp_ms`, evidence provenance, aligned TRAKE frame arrays, and `official_frame_idx()` semantics.
- Keep KIS in `src/hcmai/orchestration/workflows/kis.py` and TRAKE in `src/hcmai/orchestration/workflows/trake.py` for this migration; do not combine this work with a broad package move.
- Treat current dirty working-tree changes as user-owned. In particular, do not restore or stage the deleted `thundercompute/launch.sh`, `delete.sh`, `deploy_cloudflared_private.sh.example`, or `.secrets/tnr_api_token.example` unless explicitly requested.
- Reconcile stale active docs/config that reference those already-deleted deployment files; do not undo the user’s manual-only deployment direction.
- Preserve historical VQA research/design documents unless they claim that VQA is an active runtime feature; update active READMEs and operator configuration.
- Do not update `KNOWLEDGE.md` for this source-code trace; no external research claim was introduced.

---

### Task 1: Reduce public task contracts and application configuration

**Files:**
- Modify: `src/hcmai/common/schemas/enum.py`
- Modify: `src/hcmai/common/schemas/search.py`
- Modify: `src/hcmai/common/schemas/task.py`
- Modify: `src/hcmai/common/schemas/__init__.py`
- Modify: `src/hcmai/common/config.py`
- Modify: `configs/baseline.yaml`
- Test: `tests/test_schema.py`
- Test: `tests/test_config.py`
- Test: `tests/unit/common/test_task_contracts.py`
- Test: `tests/unit/common/test_package_layout.py`

**Interfaces:**
- Produces `TaskType.KIS` and `TaskType.TRAKE` as the only task enum members.
- Produces `TaskRequest` as the discriminated union of `SearchRequest` and `TRAKERequest`.
- Produces `TaskResponse` as the discriminated union of `SearchResponse` and `TRAKEResponse`.
- `SearchRequest.query_type` accepts only `TaskType.KIS`; `TRAKERequest.query_type` remains literal `TaskType.TRAKE`.
- `FusionConfig.task_weights` still requires every remaining `TaskType` and every `RetrievalSource`.

- [ ] **Step 1: Rewrite contract tests for the two-task boundary.**

  Update the tests so they assert:

  ```python
  assert set(TaskType) == {TaskType.KIS, TaskType.TRAKE}
  assert TaskRequest.model_validate({"query": "person"}).query_type is TaskType.KIS
  assert TaskRequest.model_validate({
      "query_type": "trake",
      "query": "E1: walk\\nE2: sit",
      "events": ["walk", "sit"],
  }).query_type is TaskType.TRAKE
  ```

  Remove tests that construct `VQARequest`, `VQAResponse`, or `VKIS`.

- [ ] **Step 2: Run the focused contract tests and verify the expected failure.**

  Run:

  ```bash
  PYTHONPATH=.:src aic/bin/pytest -q \
    tests/test_schema.py \
    tests/test_config.py \
    tests/unit/common/test_task_contracts.py \
    tests/unit/common/test_package_layout.py
  ```

  Expected before implementation: collection fails because the existing task
  union and VQA exports still exist.

- [ ] **Step 3: Remove VQA/VKIS contract and config definitions.**

  Remove `TaskType.VQA`, `TaskType.VKIS`, the VQA imports/branches in
  `common/schemas/task.py`, all VQA re-exports, `VQAConfig`,
  `VQAProfileConfig`, `_default_vqa_profiles()`, `AppConfig.vqa`, and the
  `vqa:` YAML section. Remove `vkis` and `vqa` fusion weights so the YAML
  keys match the reduced enum exactly. Change the progressive config
  docstring from “KIS/VQA” to “KIS”.

- [ ] **Step 4: Enforce the KIS request boundary.**

  Change `SearchRequest.query_type` to `Literal[TaskType.KIS] = TaskType.KIS`
  and make `_task_discriminator()` route only `kis` to the search branch and
  `trake` to the TRAKE branch. Unknown values, including `vqa` and `vkis`,
  must fail validation instead of silently entering a shared pipeline.

- [ ] **Step 5: Run the focused contract tests and commit.**

  Run the command from Step 2 and expect all focused tests to pass. Commit
  only the Task 1 files with:

  ```bash
  git add src/hcmai/common/schemas/enum.py \
    src/hcmai/common/schemas/search.py \
    src/hcmai/common/schemas/task.py \
    src/hcmai/common/schemas/__init__.py \
    src/hcmai/common/config.py configs/baseline.yaml \
    tests/test_schema.py tests/test_config.py \
    tests/unit/common/test_task_contracts.py \
    tests/unit/common/test_package_layout.py
  git commit -m "refactor: reduce task contracts to kis and trake"
  ```

### Task 2: Remove the backend VQA pipeline and registry branch

**Files:**
- Delete: `src/hcmai/api/routers/vqa.py`
- Delete: `src/hcmai/pipelines/vqa/`
- Modify: `src/hcmai/app.py`
- Modify: `src/hcmai/api/routers/__init__.py`
- Modify: `src/hcmai/api/routers/search.py`
- Modify: `src/hcmai/api/routers/system.py`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Modify: `src/hcmai/orchestration/setup.py`
- Modify: `src/hcmai/orchestration/workflows/kis.py`
- Modify: `src/hcmai/temporal/core.py`
- Modify: `src/hcmai/temporal/providers/sparse.py`
- Modify: `src/hcmai/temporal/aligners/scene.py`
- Modify: `src/hcmai/temporal/state/state.py`
- Delete: `tests/integration/test_vqa_api.py`
- Delete: `tests/integration/test_vqa_pipeline.py`
- Delete: `tests/unit/vqa/`
- Modify: `tests/test_api.py`
- Modify: `tests/test_api_contracts.py`
- Modify: `tests/test_task_router.py`
- Modify: `tests/integration/test_progressive_temporal_core.py`
- Modify: `tests/unit/temporal/test_plan04_convergence.py`
- Modify: `tests/unit/temporal/test_query_evidence.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- `SearchService.__init__` no longer accepts `vqa_config`.
- `_default_registry()` registers exactly one `KISPipeline(TaskType.KIS)` and one `TRAKEPipeline(TaskType.TRAKE)`.
- `SearchService.health()` reports `capabilities.kis`, `capabilities.trake`, and `capabilities.search`, with `query_types` containing only `kis` and `trake`.
- `POST /api/v1/search` is KIS-only; `POST /api/v1/trake` remains TRAKE-only.
- No router is registered for `/api/v1/vqa`.

- [ ] **Step 1: Add KIS/TRAKE registry and health assertions.**

  Update `tests/test_task_router.py`, `tests/test_api_contracts.py`, and the
  uninitialized health response test to assert exactly `kis` and `trake`, no
  `vqa` or `vkis`, and a remote capability map without
  `multi_image_vqa`. Update API tests so a VQA payload is rejected by schema
  validation or returns the normal unknown-route response, never a VQA
  pipeline response.

- [ ] **Step 2: Run the backend routing tests and verify the expected failure.**

  Run:

  ```bash
  PYTHONPATH=.:src aic/bin/pytest -q \
    tests/test_api.py tests/test_api_contracts.py tests/test_task_router.py
  ```

  Expected before implementation: imports fail because `app.py`,
  `SearchService`, and router exports still require VQA.

- [ ] **Step 3: Remove VQA construction and router registration.**

  Remove the VQA router import/inclusion from `app.py`, the VQA export from
  `api/routers/__init__.py`, the special VQA branch from the search router,
  `VQAConfig`/`VQAPipeline` from `SearchService`, and the `settings.vqa`
  argument from `load_search_service()`. Keep the shared temporal facade,
  progressive KIS state, ordered TRAKE alignment, data service, retrieval,
  reranking, and degraded startup diagnostics intact.

- [ ] **Step 4: Delete the VQA pipeline and remove only VQA-specific temporal semantics.**

  Delete the VQA package and its tests. Change temporal docstrings and state
  descriptions from “KIS/VQA” to “KIS” where the implementation remains
  generic. Do not delete `TemporalEvidenceCore.localize()`,
  `ProgressiveEvidenceProvider`, `ProgressiveSceneAligner`, or
  `ProgressiveStateStore`; KIS still consumes them and TRAKE still consumes
  the same facade through `ordered_plan()`/`align_ordered()`.

- [ ] **Step 5: Run KIS/TRAKE backend tests and commit.**

  Run:

  ```bash
  PYTHONPATH=.:src aic/bin/pytest -q \
    tests/test_api.py tests/test_api_contracts.py tests/test_task_router.py \
    tests/integration/test_kis_golden_path.py \
    tests/integration/test_progressive_temporal_core.py \
    tests/unit/temporal/test_plan04_convergence.py
  ```

  Expect all KIS/TRAKE assertions to pass and no import of
  `hcmai.pipelines.vqa` to remain. Commit with:

  ```bash
  git add src/hcmai/app.py src/hcmai/api/routers/__init__.py \
    src/hcmai/api/routers/search.py src/hcmai/api/routers/system.py \
    src/hcmai/api/routers/vqa.py src/hcmai/orchestration/pipeline.py \
    src/hcmai/orchestration/setup.py src/hcmai/orchestration/workflows/kis.py \
    src/hcmai/temporal/core.py src/hcmai/temporal/providers/sparse.py \
    src/hcmai/temporal/aligners/scene.py src/hcmai/temporal/state/state.py \
    src/hcmai/pipelines/vqa \
    tests/integration/test_vqa_api.py tests/integration/test_vqa_pipeline.py \
    tests/unit/vqa tests/test_api.py tests/test_api_contracts.py \
    tests/test_task_router.py tests/integration/test_progressive_temporal_core.py \
    tests/unit/temporal/test_plan04_convergence.py \
    tests/unit/temporal/test_query_evidence.py tests/conftest.py
  git commit -m "refactor: remove backend vqa workflow"
  ```

### Task 3: Remove hosted VQA model and inference transport

**Files:**
- Delete: `thundercompute/adapters/vqa.py`
- Modify: `thundercompute/config.py`
- Modify: `thundercompute/config.yaml`
- Modify: `thundercompute/pipeline.py`
- Modify: `thundercompute/adapters/local.py`
- Modify: `thundercompute/adapters/http.py`
- Modify: `thundercompute/server/api.py`
- Modify: `src/hcmai/common/schemas/inference.py`
- Modify: `configs/litellm.yaml`
- Modify: `docker-compose.yml`
- Modify: `.env.example`
- Modify: `thundercompute/.env.example`
- Modify: `pyproject.toml`
- Delete: `tests/test_llm_vqa_model.py`
- Delete: `tests/unit/llm/test_multiframe_vqa.py`
- Modify: `tests/test_llm_api.py`
- Modify: `tests/unit/llm/test_inference_gateway.py`
- Modify: `tests/unit/llm/test_inference_pool.py`
- Modify: `tests/test_remote_preparation_adapters.py`

**Interfaces:**
- `LLMService` retains embedding, caption, OCR, transcript, and rerank methods; it has no `answer_vqa()` or `answer_vqa_multi()`.
- `LLMServiceConfig` has no `vqa_model` field.
- `LocalAdapter` has no `enable_vqa`, `vqa_model`, or VQA readiness branch.
- `InferenceClient` has no VQA methods; `/ready`, embeddings, captions, OCR, preprocessing, transcripts, and rerank remain unchanged.
- `InferenceCapabilities` has no `multi_image_vqa` field.
- The `vqa` optional dependency group is removed; `embedding`, `reranking`, `transcripts`, and `preprocessing` remain.

- [ ] **Step 1: Add shared inference regression assertions.**

  Update the gateway/API tests to assert the retained endpoints and readiness
  fields. Add an assertion that the model configuration loads without a VQA
  field and that a local adapter created with the retained flags has no VQA
  model attribute. Remove tests whose only purpose is grounded answer output.

- [ ] **Step 2: Run the inference tests and verify the expected failure.**

  Run:

  ```bash
  PYTHONPATH=.:src aic/bin/pytest -q \
    tests/test_llm_api.py tests/unit/llm/test_inference_gateway.py \
    tests/unit/llm/test_inference_pool.py tests/test_remote_preparation_adapters.py
  ```

  Expected before implementation: the tests still reference VQA response
  schemas, endpoints, and `multi_image_vqa`.

- [ ] **Step 3: Remove VQA model/config/client/server code.**

  Delete `GroundedVQAModel`; remove the hosted config field and YAML block;
  remove local adapter loading, environment override, methods, readiness
  status, and capability calculation; remove HTTP client methods and server
  handlers/routes; remove VQA schema imports. Keep the inference server
  lifespan because shared models still load there.

- [ ] **Step 4: Remove VQA gateway/deployment configuration and optional extra.**

  Remove `/v1/vqa` and `/v1/vqa/multi` pass-through entries and their
  `LITELLM_UPSTREAM_VQA*` variables from LiteLLM, Compose, and `.env.example`.
  Remove `HCMAI_VQA_MODEL` from `thundercompute/.env.example` while retaining
  any manual Cloudflare token placeholder needed by the user’s deployment
  flow. Remove only the `vqa` optional dependency group; retain
  `qwen-vl-utils` in `reranking` if the Qwen reranker imports it.

- [ ] **Step 5: Run retained inference tests and commit.**

  Run the command from Step 2 and a config parse check:

  ```bash
  PYTHONPATH=.:src aic/bin/python -c \
    'from thundercompute.config import LLMServiceConfig; LLMServiceConfig.from_yaml("thundercompute/config.yaml")'
  ```

  Commit only provider/config/dependency/test files with:

  ```bash
  git add thundercompute/adapters/vqa.py thundercompute/config.py \
    thundercompute/config.yaml thundercompute/pipeline.py \
    thundercompute/adapters/local.py thundercompute/adapters/http.py \
    thundercompute/server/api.py thundercompute/.env.example \
    src/hcmai/common/schemas/inference.py configs/litellm.yaml \
    docker-compose.yml .env.example pyproject.toml \
    tests/test_llm_api.py tests/unit/llm/test_inference_gateway.py \
    tests/unit/llm/test_inference_pool.py tests/test_remote_preparation_adapters.py \
    tests/test_llm_vqa_model.py tests/unit/llm/test_multiframe_vqa.py
  git commit -m "refactor: remove hosted vqa inference"
  ```

### Task 4: Remove only frontend VQA endpoint and entrypoint references

**Files:**
- Modify: `frontend/src/api/search.js`
- Modify: `frontend/src/api/search.test.js`
- Modify: `frontend/src/features/vqa/components/VqaSearchWorkspace.jsx`
- Modify: `frontend/src/features/vqa/components/VqaSearchWorkspace.test.jsx`
- Modify: `frontend/src/features/docs/components/ApiDocsModal.jsx`
- Modify: `frontend/src/features/docs/components/ApiDocsModal.test.jsx`

**Interfaces:**
- `search.js` exports `searchFrames`, `searchTrake`, frame URL helpers, and
  suggestions; it does not export or call `searchVqa`.
- The existing workspace continues to render KIS and TRAKE. Its VQA request
  branch is removed or guarded locally so it cannot issue a network request;
  no workspace rename, split, CSS rewrite, or result-component migration is
  part of this task.
- `App.jsx` is inspected for a dedicated VQA-only entrypoint. The current
  `VqaSearchWorkspace` import/render is retained because it is also the active
  KIS/TRAKE entrypoint; removing it would remove the supported UI.

- [ ] **Step 1: Identify direct endpoint and entrypoint references.**

  Inspect `App.jsx`, `VqaSearchWorkspace.jsx`, the API client, and API docs.
  Confirm whether a VQA-only App entrypoint exists. In the current trace,
  `VqaSearchWorkspace` is a shared KIS/TRAKE entrypoint, so it must remain.
  Only direct `/api/v1/vqa` client calls, imports, docs, and their tests are
  in scope.

- [ ] **Step 2: Add/update endpoint-boundary tests.**

  Run:

  ```bash
  cd frontend && CI=true npm test -- --watchAll=false --runInBand \
    src/api/search.test.js \
    src/features/docs/components/ApiDocsModal.test.jsx \
    src/features/vqa/components/VqaSearchWorkspace.test.jsx
  ```

  Remove or update only assertions/mocks that require the removed VQA
  endpoint. Preserve KIS/TRAKE parsing, retrieval, frame identity, and
  submission assertions.

- [ ] **Step 3: Remove the VQA client and guard its existing callsite.**

  Remove `searchVqa` and its `/api/v1/vqa` request from `api/search.js`.
  Remove the import and direct request from `VqaSearchWorkspace.jsx`; when a
  question is submitted, keep the existing UI stable but stop the request and
  surface a local disabled/error state. Keep all KIS/TRAKE branches and their
  state, rendering, styles, and canonical frame handling unchanged.

- [ ] **Step 4: Remove only stale frontend API documentation and tests.**

  Remove the `/api/v1/vqa` entry from `ApiDocsModal.jsx` and its direct test
  assertion. Remove VQA endpoint mocks/assertions from `search.test.js` and
  `VqaSearchWorkspace.test.jsx`, while retaining all KIS/TRAKE tests.
  Do not modify `VqaResults.jsx`, `AdHocSearchWorkspace`, `FrameCard`,
  `ImageModal`, `FrameMetadata`, CSS, or the App test merely to rename or
  reorganize the frontend.

- [ ] **Step 5: Verify the minimal frontend surface and commit.**

  Verify that no frontend code calls or documents `/api/v1/vqa`, while the
  existing KIS/TRAKE entrypoint still builds. Run:

  ```bash
  cd frontend && CI=true npm test -- --watchAll=false --runInBand \
    src/api/search.test.js \
    src/features/docs/components/ApiDocsModal.test.jsx \
    src/features/vqa/components/VqaSearchWorkspace.test.jsx
  cd frontend && npm run build
  ```

  Commit with:

  ```bash
  git add frontend/src/api/search.js frontend/src/api/search.test.js \
    frontend/src/features/vqa/components/VqaSearchWorkspace.jsx \
    frontend/src/features/vqa/components/VqaSearchWorkspace.test.jsx \
    frontend/src/features/docs/components/ApiDocsModal.jsx \
    frontend/src/features/docs/components/ApiDocsModal.test.jsx
  git commit -m "refactor: remove frontend vqa endpoint"
  ```

### Task 5: Reconcile active documentation and stale manual-deployment references

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `src/hcmai/README.md`
- Modify: `src/hcmai/common/schemas/README.md`
- Modify: `src/hcmai/orchestration/README.md`
- Modify: `thundercompute/README.md`
- Modify: `scripts/README.md`
- Modify: `.env.example` if any stale VQA/manual-helper block remains
- Modify: `.gitignore` and `.dockerignore` only where an ignored deleted path is stale
- Delete: `tests/unit/test_thunder_launcher.py` because its launcher files are already deleted and it is not a VQA/KIS/TRAKE runtime test

**Interfaces:**
- Active docs describe KIS and TRAKE as the supported task workflows.
- ThunderCompute docs describe shared inference endpoints only and the current manual deployment boundary; they do not instruct users to build/run a deleted launcher or VQA model.
- Historical research docs under `docs/research/` and historical implementation specs remain available and are not rewritten as if they were active runtime contracts.

- [ ] **Step 1: Search active documentation for stale runtime claims.**

  Run:

  ```bash
  rg -n -i "vqa|vkis|multi_image_vqa|/api/v1/vqa|/v1/vqa|HCMAI_VQA_MODEL|launch\.sh|delete\.sh|deploy_cloudflared_private\.sh\.example" \
    AGENTS.md README.md src/hcmai/README.md src/hcmai/common/schemas/README.md \
    src/hcmai/orchestration/README.md thundercompute/README.md scripts/README.md \
    .env.example thundercompute/.env.example .gitignore .dockerignore
  ```

  Classify each remaining hit as an active runtime claim or historical note.

- [ ] **Step 2: Update active architecture and operator docs.**

  Remove VQA/VKIS from active task tables, diagrams, endpoint lists, health
  examples, and folder descriptions. Document the two retained paths:

  ```text
  query -> KIS retrieval -> progressive scene localization -> frame submission
  events -> TRAKE retrieval -> ordered temporal alignment -> path submission
  ```

  Do not claim that the remote GPU hosts a VQA model. Keep historical research
  conclusions in their original documents unless they are presented as active
  deployment instructions.

- [ ] **Step 3: Remove stale deleted-launcher test/document references.**

  Delete the stale launcher test and remove manual-helper environment variables
  that no longer have a consumer. Keep only ignore rules that protect the
  user’s private deployment script and secret files.

- [ ] **Step 4: Run markdown/reference checks and commit.**

  Run:

  ```bash
  rg -n "hcmai\.pipelines\.vqa|thundercompute\.adapters\.vqa|/api/v1/vqa|/v1/vqa|HCMAI_VQA_MODEL|multi_image_vqa|searchVqa" \
    src thundercompute frontend/src configs .env.example docker-compose.yml pyproject.toml \
    --glob '!**/__pycache__/**'
  ```

  Expected output: no active backend/runtime/config/frontend endpoint or
  client-call references. The retained legacy
  `frontend/src/features/vqa/components/VqaSearchWorkspace.jsx` filename and
  deferred UI
  labels may remain by design; they must not call `searchVqa` or the VQA URL.
  Historical `docs/research/` and `docs/superpowers/` references may also
  remain by design.
  Commit with:

  ```bash
  git add AGENTS.md README.md src/hcmai/README.md \
    src/hcmai/common/schemas/README.md src/hcmai/orchestration/README.md \
    thundercompute/README.md scripts/README.md .env.example .gitignore \
    .dockerignore tests/unit/test_thunder_launcher.py
  git commit -m "docs: describe kis and trake runtime only"
  ```

### Task 6: Whole-system verification and SDD review gate

**Files:**
- Test: all retained backend and frontend tests
- Check: active source/config/docs reference inventory

- [ ] **Step 1: Run import and syntax checks.**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m compileall -q src/hcmai thundercompute
  PYTHONPATH=.:src aic/bin/python -c \
    'from hcmai.common.schemas import TaskType, TaskRequest, TaskResponse; assert set(TaskType) == {TaskType.KIS, TaskType.TRAKE}'
  docker compose config --quiet
  ```

- [ ] **Step 2: Run focused backend regression suites.**

  ```bash
  PYTHONPATH=.:src aic/bin/pytest -q \
    tests/test_api.py tests/test_api_contracts.py tests/test_schema.py \
    tests/test_config.py tests/test_task_router.py \
    tests/test_llm_api.py tests/unit/llm/test_inference_gateway.py \
    tests/integration/test_kis_golden_path.py \
    tests/integration/test_progressive_temporal_core.py \
    tests/retrieval/test_fast_track_retrieval_composition.py
  ```

- [ ] **Step 3: Run the repository release gate.**

  ```bash
  scripts/validate_repository.sh
  ```

  If the existing dirty deletion of `tests/unit/test_thunder_launcher.py`
  causes a stale-file assertion, record it as the pre-existing manual-only
  deployment cleanup rather than restoring the deleted launcher.

- [ ] **Step 4: Dispatch Superpowers reviews.**

  For each implementation task, dispatch one fresh implementer followed by a
  task reviewer. After Task 6, dispatch one whole-branch reviewer against the
  merge base. Use the SDD ledger to record task completion and any ruling. Do
  not merge or push without explicit user authorization.

## Final acceptance checklist

- [ ] `TaskType` contains exactly KIS and TRAKE.
- [ ] `SearchService` registry contains exactly KIS and TRAKE.
- [ ] KIS progressive search and TRAKE ordered alignment tests pass.
- [ ] No VQA route, pipeline, schema, model, adapter, prompt, or capability remains in active runtime.
- [ ] No VKIS route/config/test remains in active runtime.
- [ ] Shared inference still supports embedding, caption, OCR, ASR/diarization, and reranking.
- [ ] Frontend no longer defines/calls `searchVqa` or advertises
      `/api/v1/vqa`; the existing shared KIS/TRAKE workspace remains intact
      apart from the removed endpoint callsite.
- [ ] Active docs/configs match the two-task architecture.
- [ ] Canonical identity/provenance invariants remain covered by tests.
