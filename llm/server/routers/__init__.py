"""Capability-focused routers for the hosted inference API."""

from llm.server.routers.boundaries import router as boundaries_router
from llm.server.routers.embeddings import router as embeddings_router
from llm.server.routers.enrichment import router as enrichment_router
from llm.server.routers.query_preparation import router as query_preparation_router
from llm.server.routers.reranking import router as reranking_router
from llm.server.routers.system import router as system_router
from llm.server.routers.transcripts import router as transcripts_router

ROUTERS = (
    system_router,
    enrichment_router,
    embeddings_router,
    boundaries_router,
    transcripts_router,
    reranking_router,
    query_preparation_router,
)

__all__ = ["ROUTERS"]
