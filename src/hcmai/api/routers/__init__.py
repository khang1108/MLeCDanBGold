"""FastAPI routers backed by the public SearchService facade."""

from hcmai.api.routers.frames import create_frames_router
from hcmai.api.routers.search import create_search_router
from hcmai.api.routers.system import create_system_router
from hcmai.api.routers.trake import create_trake_router

__all__ = [
    "create_frames_router",
    "create_search_router",
    "create_system_router",
    "create_trake_router",
]
