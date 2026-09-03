"""FastAPI routers backed by HCMAI services and stores."""

from hcmai.api.routers.database import create_database_router
from hcmai.api.routers.frames import create_frames_router
from hcmai.api.routers.history import create_workspace_router
from hcmai.api.routers.query_candidates import create_query_candidates_router
from hcmai.api.routers.search import create_search_router
from hcmai.api.routers.system import create_system_router
from hcmai.api.routers.trake import create_trake_router
from hcmai.api.routers.videos import create_video_router


__all__ = [
    "create_frames_router",
    "create_database_router",
    "create_query_candidates_router",
    "create_search_router",
    "create_system_router",
    "create_trake_router",
    "create_video_router",
    "create_workspace_router",
]
