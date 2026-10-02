# vLLM Serving & Offline Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transition LLM & VLM inference to standalone vLLM server endpoints (`/v1/chat/completions`) for high-throughput serving, and simplify the `offline/` and `llm/` folders by removing slow in-process HuggingFace loaders and obsolete C++ CMake extraction code.

**Architecture:**
- **Text LLM Serving (Port 8000):** Run standalone `vllm serve Qwen/Qwen2.5-7B-Instruct`. Existing `hcmai.inference.LLMClient` already conforms to OpenAI structured outputs and connects directly via `HCMAI_LLM_BASE_URL`.
- **VLM Caption Serving (Port 8001):** Run standalone `vllm serve Qwen/Qwen2.5-VL-7B-Instruct`. Implement `VLLMCaptionAdapter` in `offline/enrichment/caption/adapters/vllm.py` communicating with OpenAI Vision `/v1/chat/completions` (base64 image payload + canonical retrieval caption prompt).
- **Pruning:** Delete obsolete C++ CMake project in `offline/keyframes/keyframes_extraction/`. Prune in-process HuggingFace loaders (`llm/local/text_generation.py`, legacy PyTorch imports in `qwen_vl.py`).

**Tech Stack:** Python 3.12, vLLM (OpenAI-compatible HTTP API), PIL/Pillow, `httpx`/`requests`, Pytest.

**Spec:** `docs/superpowers/specs/2026-10-02-vllm-serving-and-offline-simplification-design.md`

## Global Constraints
- Preserve canonical identity (`video_id`, `frame_idx`, `frame_id`, `timestamp_ms`) across all caption artifacts.
- Downstream artifact schema (`captions.parquet`, `manifest.json`) must remain 100% compatible.
- All network calls must follow OpenAI-compatible standards (`/v1/chat/completions`, `/v1/models`).
- Do not break existing unit tests; maintain full test suite pass rate.

---

### Task 1: Implement `VLLMCaptionAdapter` for Offline Caption Enrichment

**Files:**
- Create: `offline/enrichment/caption/adapters/vllm.py`
- Test: `tests/offline/test_vllm_caption_adapter.py`

**Interfaces:**
- Consumes: `CaptionModelConfig` from `offline.enrichment.caption.models.contracts`, PIL `Image.Image`.
- Produces: `VLLMCaptionAdapter` conforming to `CaptionAdapter` protocol:
  - `resolve_revision() -> str`
  - `caption_batch(images: Sequence[Any]) -> list[str]`

- [ ] **Step 1: Write the failing test**

Create `tests/offline/test_vllm_caption_adapter.py`:

```python
"""Unit tests for VLLMCaptionAdapter with mocked OpenAI Vision endpoints."""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image
import pytest

from offline.enrichment.caption.adapters.vllm import (
    VLLMCaptionAdapter,
    _encode_image_to_base64,
)


def _make_dummy_image(color: tuple[int, int, int] = (255, 0, 0)) -> Image.Image:
    """Create a 32x32 RGB image for testing."""
    return Image.new("RGB", (32, 32), color=color)


def test_REQ_001_encode_image_to_base64() -> None:
    """Encode PIL image to valid base64 JPEG string."""
    img = _make_dummy_image()
    b64_str = _encode_image_to_base64(img)
    assert isinstance(b64_str, str)
    decoded = base64.b64decode(b64_str)
    assert len(decoded) > 0


def test_REQ_002_adapter_uses_custom_batch_fn() -> None:
    """Support direct mock batch_fn for testing without HTTP."""
    config = SimpleNamespace(
        model_checkpoint="Qwen/Qwen2.5-VL-7B-Instruct",
        revision="test-rev",
    )
    adapter = VLLMCaptionAdapter(
        config=config,
        batch_fn=lambda imgs: [f"caption-{i}" for i in range(len(imgs))],
    )
    assert adapter.resolve_revision() == "test-rev"
    results = adapter.caption_batch([_make_dummy_image(), _make_dummy_image()])
    assert results == ["caption-0", "caption-1"]


def test_REQ_003_caption_batch_openai_request() -> None:
    """Format and send OpenAI vision chat completion requests."""
    config = SimpleNamespace(
        model_checkpoint="Qwen/Qwen2.5-VL-7B-Instruct",
        revision=None,
    )
    adapter = VLLMCaptionAdapter(
        config=config,
        base_url="http://mock-vllm:8001/v1",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "A high-angle view of a red surface in clear lighting."
                }
            }
        ]
    }

    with patch("requests.post", return_value=mock_response) as mock_post:
        captions = adapter.caption_batch([_make_dummy_image()])
        assert len(captions) == 1
        assert captions[0] == "A high-angle view of a red surface in clear lighting."
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert "http://mock-vllm:8001/v1/chat/completions" in mock_post.call_args[0]
        json_data = call_kwargs["json"]
        assert json_data["model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
        content_items = json_data["messages"][0]["content"]
        assert any(item.get("type") == "image_url" for item in content_items)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `aic/bin/pytest tests/offline/test_vllm_caption_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'offline.enrichment.caption.adapters.vllm'`

- [ ] **Step 3: Write minimal implementation**

Create `offline/enrichment/caption/adapters/vllm.py`:

```python
"""vLLM / OpenAI Vision API adapter for frame caption enrichment.

This module formats keyframes into OpenAI-compatible multimodal chat completion
requests for standalone vLLM serving, avoiding in-process PyTorch model loading.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
import io
import os
import re
from typing import Any

from PIL import Image
import requests

from offline.enrichment.caption.adapters.qwen_vl import _QWEN_VL_PROMPT, _clean_caption
from offline.enrichment.caption.models.contracts import CaptionModelConfig


def _encode_image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """Encode a PIL image to a base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format=format, quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class VLLMCaptionAdapter:
    """Adapt standalone vLLM OpenAI Vision serving to the enrichment batch contract."""

    def __init__(
        self,
        config: Any,
        base_url: str | None = None,
        batch_fn: Callable[[Sequence[Any]], Sequence[str]] | None = None,
        max_workers: int = 8,
        timeout: float = 30.0,
    ) -> None:
        self.config = config
        self.base_url = (
            base_url
            or os.getenv("HCMAI_VLM_CAPTION_URL")
            or "http://localhost:8001/v1"
        ).rstrip("/")
        self.batch_fn = batch_fn
        self.max_workers = max_workers
        self.timeout = timeout
        self.resolved_revision: str | None = getattr(config, "revision", None)

    def resolve_revision(self) -> str:
        """Resolve or query model revision from vLLM endpoint or config."""
        if self.resolved_revision:
            return self.resolved_revision

        if self.batch_fn is not None:
            self.resolved_revision = "mock-vllm-revision"
            return self.resolved_revision

        try:
            resp = requests.get(f"{self.base_url}/models", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                if models:
                    self.resolved_revision = str(models[0].get("id", "vllm-serving"))
                    return self.resolved_revision
        except Exception:
            pass

        self.resolved_revision = "vllm-serving"
        return self.resolved_revision

    def _caption_single_image(self, image: Image.Image) -> str:
        """Call vLLM chat completions with a single base64-encoded image."""
        b64 = _encode_image_to_base64(image)
        model = getattr(self.config, "model_checkpoint", "Qwen/Qwen2.5-VL-7B-Instruct")
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {"type": "text", "text": _QWEN_VL_PROMPT},
                    ],
                }
            ],
            "max_tokens": 128,
            "temperature": 0.0,
        }

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"vLLM caption request failed ({resp.status_code}): {resp.text}")

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"vLLM caption response has no choices: {data}")

        raw_text = choices[0].get("message", {}).get("content", "")
        return _clean_caption(raw_text)

    def caption_batch(self, images: Sequence[Any]) -> list[str]:
        """Generate captions for a batch of images concurrently."""
        if not images:
            return []

        if self.batch_fn is not None:
            return list(self.batch_fn(images))

        # Run concurrent calls against vLLM server to utilize continuous batching
        workers = min(len(images), self.max_workers)
        if workers <= 1:
            return [self._caption_single_image(img) for img in images]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(self._caption_single_image, images))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `aic/bin/pytest tests/offline/test_vllm_caption_adapter.py -v`
Expected: PASS (all 3 tests pass)

- [ ] **Step 5: Commit**

```bash
git add offline/enrichment/caption/adapters/vllm.py tests/offline/test_vllm_caption_adapter.py
git commit -m "feat(offline): add VLLMCaptionAdapter for standalone vLLM serving"
```

---

### Task 2: Integrate `VLLMCaptionAdapter` into Caption Generator & Config

**Files:**
- Modify: `offline/enrichment/caption/generator.py:41-75, 160-195`
- Modify: `offline/enrichment/caption/config.py`
- Test: `tests/offline/test_vllm_caption_generator.py`

**Interfaces:**
- Consumes: `VLLMCaptionAdapter` from Task 1.
- Produces: `generate_captions` supporting default vLLM execution backend while preserving exact artifact outputs.

- [ ] **Step 1: Write the failing test**

Create `tests/offline/test_vllm_caption_generator.py`:

```python
"""Integration test for caption generator with VLLMCaptionAdapter."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pyarrow.parquet as pq
import pytest

from offline.enrichment.caption.adapters.vllm import VLLMCaptionAdapter
from offline.enrichment.caption.config import CaptionConfig
from offline.enrichment.caption.generator import generate_captions


def test_REQ_004_generate_captions_with_vllm_adapter(tmp_path: Path) -> None:
    """Generate caption parquet and manifest artifacts using VLLMCaptionAdapter."""
    # Create dummy frame image
    img_path = tmp_path / "frame_001.jpg"
    Image.new("RGB", (32, 32), (0, 255, 0)).save(img_path)

    # Create dummy manifest frames parquet
    import pyarrow as pa
    frames_table = pa.Table.from_pydict({
        "frame_id": ["v001_f001"],
        "video_id": ["v001"],
        "frame_idx": [1],
        "timestamp_ms": [1000],
        "image_path": [str(img_path)],
    })
    frames_file = tmp_path / "frames.parquet"
    pq.write_table(frames_table, frames_file)

    output_dir = tmp_path / "output"
    config = CaptionConfig(
        model_checkpoint="Qwen/Qwen2.5-VL-7B-Instruct",
        batch_size=1,
    )
    adapter = VLLMCaptionAdapter(
        config=config,
        batch_fn=lambda imgs: ["A green square frame in clear focus."],
    )

    manifest = generate_captions(
        frames_path=frames_file,
        output_dir=output_dir,
        config=config,
        captioner=adapter,
        dataset_root=tmp_path,
    )

    assert manifest["completed_count"] == 1
    assert manifest["failed_count"] == 0

    captions_pq = output_dir / "captions.parquet"
    assert captions_pq.exists()
    table = pq.read_table(captions_pq)
    assert table.num_rows == 1
    assert table.column("caption")[0].as_py() == "A green square frame in clear focus."
    assert table.column("video_id")[0].as_py() == "v001"
    assert table.column("frame_idx")[0].as_py() == 1
```

- [ ] **Step 2: Run test to verify it passes/fails**

Run: `aic/bin/pytest tests/offline/test_vllm_caption_generator.py -v`
(Verify behavior and make sure dependencies import cleanly)

- [ ] **Step 3: Update `offline/enrichment/caption/generator.py`**

In `offline/enrichment/caption/generator.py`:
- Import `VLLMCaptionAdapter` lazily or alongside `QwenVLCaptionAdapter`.
- Set default fallback captioner to `VLLMCaptionAdapter(config)` if `captioner is None` and `execution_backend == "vllm"`.
- Support `--execution-backend vllm` in CLI parser and set default or recommended option.

- [ ] **Step 4: Run tests to verify they pass**

Run: `aic/bin/pytest tests/offline/test_vllm_caption_generator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add offline/enrichment/caption/generator.py tests/offline/test_vllm_caption_generator.py
git commit -m "feat(offline): integrate VLLMCaptionAdapter into caption generator"
```

---

### Task 3: Delete Obsolete C++ CMake Keyframe Extraction Project

**Files:**
- Delete: `offline/keyframes/keyframes_extraction/` (entire directory tree: CMakeLists.txt, include/, src/, tests/)

**Interfaces:**
- Consumes: Nothing; competition keyframes are canonically provided by BTC.
- Produces: Cleaned `offline/keyframes/` with only python modules.

- [ ] **Step 1: Check git status and verify no imports depend on keyframes_extraction**

Run: `git grep "keyframes_extraction"`
Expected: No active Python imports outside the deleted folder.

- [ ] **Step 2: Remove directory tree**

Run: `rm -rf offline/keyframes/keyframes_extraction`

- [ ] **Step 3: Run existing offline tests to verify zero regressions**

Run: `aic/bin/pytest tests/offline/ -q`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add -u offline/keyframes/
git commit -m "chore(offline): remove obsolete C++ keyframes_extraction cmake project"
```

---

### Task 4: Prune In-Process HuggingFace Text Generation & Cleanup `llm/`

**Files:**
- Delete: `llm/local/text_generation.py`
- Modify: `llm/local/adapter.py:49,61,95-98,122,324`
- Modify: `tests/llm/test_legacy_inference_removal.py`

**Interfaces:**
- Consumes: Standardized `hcmai.inference.LLMClient` via HTTP.
- Produces: Pruned `llm/local/` without dead HuggingFace text generation loading code.

- [ ] **Step 1: Write test verifying text generation is not loaded locally**

Update `tests/llm/test_legacy_inference_removal.py` to assert that `LocalAdapter` no longer contains or loads `text_generator`.

- [ ] **Step 2: Remove `llm/local/text_generation.py` and references in `llm/local/adapter.py`**

Delete `llm/local/text_generation.py`.
In `llm/local/adapter.py`, remove the `enable_text_generation` branch that imports `TextGenerationAdapter`.

- [ ] **Step 3: Run LLM tests**

Run: `aic/bin/pytest tests/llm/ -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git rm llm/local/text_generation.py
git add llm/local/adapter.py tests/llm/test_legacy_inference_removal.py
git commit -m "chore(llm): prune in-process HuggingFace text generation loader"
```

---

### Task 5: End-to-End Verification & Documentation Update

**Files:**
- Modify: `KNOWLEDGE.md`
- Run: Full test suite (`aic/bin/pytest tests/ -q`)
- Run: Frontend test suite (`npm test --prefix frontend`)

- [ ] **Step 1: Run full backend test suite**

Run: `aic/bin/pytest tests/ -q`
Expected: 100% PASS (~560 passed).

- [ ] **Step 2: Run frontend test suite**

Run: `npm test --prefix frontend`
Expected: 100% PASS.

- [ ] **Step 3: Update `KNOWLEDGE.md`**

Add an entry documenting the standalone vLLM serving architecture, dual ports (:8000 for Text LLM, :8001 for VLM), continuous batching benefits, and offline directory cleanup.

- [ ] **Step 4: Commit and tag/status update**

```bash
git add KNOWLEDGE.md
git commit -m "docs(research): record standalone vLLM serving and offline cleanup in KNOWLEDGE.md"
```
