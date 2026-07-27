"""Public KISC conversation interpretation API."""

from .resolver import ConversationResolver, ConversationResolverError

__all__ = [
    "ConversationResolver",
    "ConversationResolverError",
]
