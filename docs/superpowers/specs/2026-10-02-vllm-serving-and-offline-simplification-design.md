# Specification: Standalone vLLM Serving & Offline Simplification

- **Date:** 2026-10-02
- **Status:** Approved
- **Target:** Transition LLM & VLM inference to standalone vLLM server instances and simplify/prune the `offline/` and `llm/` folders.

---

## 1. Problem Statement & Motivation

1. **Slow and Fragile In-Process Model Loading:**
   Currently, offline captioning (`offline/enrichment/caption/adapters/qwen_vl.py`) and local text generation load HuggingFace `transformers` models (`AutoProcessor`, `Qwen3VLForConditionalGeneration`) directly into the Python process memory. This approach:
   - Suffers from Python GIL bottlenecks and slow sequential batching.
   - Prone to CUDA memory fragmentation and Out-Of-Memory (OOM) errors during large batch runs.
   - Misses out on state-of-the-art serving optimizations like continuous batching, chunked prefill, and PagedAttention.

2. **Excessive Complexity in `offline/` and `llm/` Directories:**
   - `offline/keyframes/keyframes_extraction/` contains an entire 18-file C++ CMake project. Under HCMAI rules, keyframes are canonically provided by the competition organizers (BTC); custom keyframe extraction is not used in the competition path.
   - `llm/local/text_generation.py` contains redundant local generation logic while the online search pipeline already standardizes on `hcmai.inference.LLMClient` communicating over HTTP.
   - The caption pipeline wraps layers of legacy adapters (`RemoteCaptionAdapter`, `LLMService.remote(...)`) rather than using a direct, clean OpenAI-compatible client.

---

## 2. Proposed Architecture

### 2.1 Dual Standalone vLLM Server Architecture

Two dedicated vLLM server instances run as independent services (locally on GPU or remotely on ThunderCompute / GPU nodes):

1. **Text LLM Server (Port 8000 by default):**
   - Model: e.g. `Qwen/Qwen2.5-7B-Instruct`
   - Command: `vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --dtype bfloat16 --max-model-len 8192`
   - Protocols: OpenAI-compatible `/v1/chat/completions` with JSON schema structured output.
   - Consumer: `hcmai.inference.LLMClient` for:
     - KIS initial intent decomposition (`KISIntentResolver`)
     - Scoped interactive query patching (`KISScopedResolver`)
     - Multi-lingual translation (`EventTranslator`)
     - Query hypothesis re-ranking / verification.

2. **Vision-Language (VLM) Server (Port 8001 by default):**
   - Model: e.g. `Qwen/Qwen2.5-VL-7B-Instruct`
   - Command: `vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001 --dtype bfloat16 --limit-mm-per-prompt image=1 --max-model-len 4096`
   - Protocols: OpenAI-compatible `/v1/chat/completions` with image payloads (`{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}`).
   - Consumer: `offline.enrichment.caption` pipeline via `VLLMCaptionClient`.

---

### 2.2 Offline Caption Pipeline Simplification

1. **Lightweight OpenAI Vision Client (`VLLMCaptionClient`):**
   - Replaces in-process HuggingFace `qwen_vl.py` and complex `RemoteCaptionAdapter`.
   - Sends batch requests to vLLM's `/v1/chat/completions` concurrently using `asyncio` / `httpx` or batched HTTP requests.
   - Uses the proven canonical prompt template (instructing factual single-sentence descriptions without transcribing text overlays).
   - Converts frames to base64 JPEG buffers on the fly.
   - Standardizes response handling with graceful error handling and retry mechanisms.

2. **Preserving Invariants & Downstream Contracts:**
   - Output format remains 100% identical: writes `CaptionArtifact` (`captions.parquet`, `manifest.json`) preserving `video_id`, `frame_idx`, and timestamp metadata.
   - Downstream consumers (Dense Context retrieval, BM25 scorers, Segment indexes) remain completely untouched and unaffected.

---

### 2.3 Dead Code Elimination & Directory Pruning

1. **Prune `offline/keyframes/keyframes_extraction/`:**
   - Remove the obsolete C++ CMake directory tree entirely.
   - Retain `offline/keyframes/__init__.py` and canonical frame loading readers.

2. **Prune In-Process HuggingFace Text Generation:**
   - Remove `llm/local/text_generation.py`.
   - Ensure `llm/` references point directly to standard HTTP endpoints.

3. **Clean Up `offline/enrichment/caption/adapters/`:**
   - Introduce `vllm.py` as the canonical adapter.
   - Keep a clean, minimal interface for test mocks (`mock_fn` or fake HTTP responses).
   - Deprecate / remove heavyweight PyTorch model loading in `qwen_vl.py`.

---

## 3. Configuration & Environment Variables

- `HCMAI_LLM_BASE_URL`: Base URL for Text LLM (default: `http://localhost:8000/v1`).
- `HCMAI_VLM_CAPTION_URL`: Base URL for VLM Captioning (default: `http://localhost:8001/v1`).
- `HCMAI_VLM_MODEL`: Model checkpoint name (default: `Qwen/Qwen2.5-VL-7B-Instruct`).
- `HCMAI_VLM_MAX_CONCURRENCY`: Concurrency limit for async caption requests (default: 16).
