"""Compatibility module for enrichment routers.

Canonical routes now live in captions.py, ocr.py, and objects.py.
"""
from llm.server.routers.captions import router as captions_router
from llm.server.routers.ocr import router as ocr_router
from llm.server.routers.objects import router as objects_router

__all__ = ["captions_router", "ocr_router", "objects_router"]
