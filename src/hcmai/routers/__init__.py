"""FastAPI routers for the HCMAI HTTP boundary."""

from hcmai.routers.frames import create_frames_router
from hcmai.routers.query_suggestions import create_query_suggestion_router
from hcmai.routers.search import (
    StandaloneSearchDispatcher,
    create_search_router,
)
from hcmai.routers.system import create_system_router

__all__ = [
    "StandaloneSearchDispatcher",
    "create_frames_router",
    "create_query_suggestion_router",
    "create_search_router",
    "create_system_router",
]
