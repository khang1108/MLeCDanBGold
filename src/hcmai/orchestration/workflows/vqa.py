"""Compatibility import for the VQA task pipeline.

The task implementation now lives with its VQA domain modules under
``hcmai.pipelines.vqa.pipeline``. New code should import it from there.
"""

from hcmai.pipelines.vqa.pipeline import VQAPipeline

__all__ = ["VQAPipeline"]
