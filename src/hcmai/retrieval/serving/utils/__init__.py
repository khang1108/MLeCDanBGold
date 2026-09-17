"""Utility modules for HTTP retrieval serving."""

from hcmai.retrieval.serving.utils.config import RetrievalClientSettings
from hcmai.retrieval.serving.utils.errors import (
    RetrievalClientError,
    RetrievalInternalError,
    RetrievalInvalidRequestError,
    RetrievalNotFoundError,
    RetrievalProtocolError,
    RetrievalTooLargeError,
    RetrievalUnavailableError,
    map_http_error,
)
from hcmai.retrieval.serving.utils.serialization import (
    artifact_to_schema,
    path_to_schema,
    plan_to_schema,
    schema_to_artifact,
    schema_to_path,
    schema_to_plan,
    schema_to_video_scores,
    video_scores_to_schema,
)

__all__ = [
    "RetrievalClientError",
    "RetrievalClientSettings",
    "RetrievalInternalError",
    "RetrievalInvalidRequestError",
    "RetrievalNotFoundError",
    "RetrievalProtocolError",
    "RetrievalTooLargeError",
    "RetrievalUnavailableError",
    "artifact_to_schema",
    "map_http_error",
    "path_to_schema",
    "plan_to_schema",
    "schema_to_artifact",
    "schema_to_path",
    "schema_to_plan",
    "schema_to_video_scores",
    "video_scores_to_schema",
]
