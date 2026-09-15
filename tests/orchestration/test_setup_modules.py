"""Tests for focused startup loader boundaries."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from hcmai.common.config import AppConfig, EventTranslationConfig
from hcmai.orchestration.setup import _load_event_translator
from hcmai.orchestration.corpus_setup import load_corpus
from hcmai.orchestration.retrieval_setup import select_visual_retriever
from hcmai.retrieval.models import RetrievalSource
from hcmai.retrieval.translation.service import EventTranslator


class _Retrieval:
    """Minimal source registry used to test setup selection."""

    def __init__(self, visual: object) -> None:
        self.visual = visual
        self.requests: list[RetrievalSource] = []

    def source_retriever(self, source: RetrievalSource) -> object | None:
        """Record setup's source request and return the visual capability."""
        self.requests.append(source)
        return self.visual if source is RetrievalSource.VISUAL else None


class SetupModuleTest(unittest.TestCase):
    """Keep startup boundaries narrow and observable."""

    def test_corpus_loader_keeps_frame_metadata_as_a_required_input(self) -> None:
        with TemporaryDirectory() as directory:
            messages: list[str] = []
            missing_metadata = Path(directory) / "frames.parquet"

            with self.assertRaisesRegex(FileNotFoundError, "Metadata not available"):
                load_corpus(object(), missing_metadata, Path(directory), messages)

        self.assertEqual(messages, [])

    def test_visual_retriever_selection_stays_in_startup(self) -> None:
        visual = object()
        retrieval = _Retrieval(visual)

        selected = select_visual_retriever(retrieval)  # type: ignore[arg-type]

        self.assertIs(selected, visual)
        self.assertEqual(retrieval.requests, [RetrievalSource.VISUAL])

    def test_baseline_config_exposes_only_event_translation_settings(self) -> None:
        """The runtime baseline loads the reduced translation configuration."""
        settings = AppConfig.from_yaml("configs/baseline.yaml")

        self.assertEqual(
            settings.event_translation,
            EventTranslationConfig(
                prompt_version="event-translation-v1",
                cache_enabled=True,
                cache_ttl_seconds=3600,
                cache_max_entries=2048,
            ),
        )

    def test_event_translator_loader_uses_configured_translation_settings(self) -> None:
        """Startup composes translation from the shared LLM client and config."""
        settings = AppConfig(
            event_translation=EventTranslationConfig(prompt_version="test-prompt-v1")
        )
        llm = Mock(model="provider/model-a")
        messages: list[str] = []

        translator = _load_event_translator(settings, messages, llm=llm)

        self.assertIsInstance(translator, EventTranslator)
        self.assertIs(translator._llm, llm)
        self.assertEqual(translator._config, settings.event_translation)
        self.assertEqual(messages, [])

    def test_event_translator_loader_reports_missing_llm_client(self) -> None:
        """Startup reports when translation cannot use the shared LLM client."""
        messages: list[str] = []

        translator = _load_event_translator(AppConfig(), messages, llm=None)

        self.assertIsNone(translator)
        self.assertEqual(
            messages,
            ["Event translation unavailable: LLM client not configured"],
        )


if __name__ == "__main__":
    unittest.main()
