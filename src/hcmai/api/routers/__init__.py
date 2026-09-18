"""FastAPI routers backed by HCMAI services and stores."""

from hcmai.api.routers.avs import create_avs_router
from hcmai.api.routers.event_trail import create_event_trail_router
from hcmai.api.routers.feedback import create_feedback_router
from hcmai.api.routers.frames import create_frames_router
from hcmai.api.routers.kis import create_kis_router
from hcmai.api.routers.search import create_search_router
from hcmai.api.routers.system import create_system_router
from hcmai.api.routers.trake import create_trake_router
from hcmai.api.routers.videos import create_video_router
from hcmai.api.routers.vbs import create_vbs_router


__all__ = [
    "create_avs_router",
    "create_event_trail_router",
    "create_feedback_router",
    "create_frames_router",
    "create_kis_router",
    "create_search_router",
    "create_system_router",
    "create_trake_router",
    "create_video_router",
    "create_vbs_router",
]
