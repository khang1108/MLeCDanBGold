"""Capability-focused routers for the hosted inference API."""
from llm.server.routers.boundaries import router as boundaries_router
from llm.server.routers.captions import router as captions_router
from llm.server.routers.embeddings import router as embeddings_router
from llm.server.routers.generation import router as generation_router
from llm.server.routers.objects import router as objects_router
from llm.server.routers.ocr import router as ocr_router
from llm.server.routers.system import router as system_router
from llm.server.routers.transcripts import router as transcripts_router

ROUTERS = (
    system_router,
    embeddings_router,
    generation_router,
    captions_router,
    ocr_router,
    objects_router,
    transcripts_router,
    boundaries_router,
)
__all__ = ["ROUTERS"]
