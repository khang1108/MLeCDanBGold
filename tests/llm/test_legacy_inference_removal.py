"""Prevent retired query-preparation and text-embedding APIs from returning."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from llm.config import LLMServiceConfig
from llm.local.adapter import LocalAdapter
from llm.local.readiness import build_readiness
from llm.server.routers import ROUTERS


class LegacyInferenceRemovalTest(unittest.TestCase):
    """Verify that the private service owns only unreplaced capabilities."""

    def test_task_4_and_9_do_not_register_retired_text_routes(self) -> None:
        """Keep query preparation and text embedding on the shared clients."""

        paths = {
            route.path
            for router in ROUTERS
            for route in router.routes
        }

        self.assertNotIn("/query-preparation/translate", paths)
        self.assertNotIn("/query-preparation/candidates", paths)
        self.assertNotIn("/v1/embeddings/text", paths)
        self.assertNotIn("/v1/rerank", paths)
        self.assertIn("/v1/embeddings/images", paths)

    def test_task_4_and_9_remove_retired_local_model_methods(self) -> None:
        """Prevent local Qwen/text APIs from becoming a second code path."""

        self.assertFalse(hasattr(LocalAdapter, "embed_text"))
        self.assertFalse(hasattr(LocalAdapter, "translate_query_events"))
        self.assertFalse(hasattr(LocalAdapter, "generate_query_candidates"))
        self.assertFalse(hasattr(LocalAdapter, "rerank"))

    def test_readiness_still_works_without_retired_text_encoder(self) -> None:
        """Readiness must not depend on the removed caption text encoder."""

        adapter = SimpleNamespace(
            config=LLMServiceConfig(),
            visual_encoder=None,
            captioner=None,
            ocr_adapter=None,
            asr=None,
            diarization=None,
            transcript_config=None,
            enable_caption=False,
            enable_visual_embedding=True,
            enable_ocr=False,
            enable_asr=False,
            enable_diarization=False,
        )

        readiness = build_readiness(adapter)

        self.assertFalse(readiness.ready)
        self.assertFalse(readiness.capabilities.embedding)
        self.assertNotIn("query_preparation", readiness.capabilities.model_dump())
