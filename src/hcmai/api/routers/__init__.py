"""FastAPI routers backed by the public SearchService facade."""

from hcmai.api.routers.frames import create_frames_router
from hcmai.api.routers.minichallenge import create_minichallenge_router
from hcmai.api.routers.query_suggestions import create_query_suggestion_router
from hcmai.api.routers.search import create_search_router
from hcmai.api.routers.system import create_system_router

__all__ = [
    "create_frames_router",
    "create_minichallenge_router",
    "create_query_suggestion_router",
    "create_search_router",
    "create_system_router",
]
