"""Domain errors shared by orchestration workflows and transport adapters."""


class InvalidQueryInputError(ValueError):
    """Signal an invalid user-supplied query bundle before temporal execution."""


class RevisionConflictError(ValueError):
    """Signal a revision index conflict between client expected revision and input history."""


class SearchServiceGatewayError(RuntimeError):
    """Signal an unrecoverable protocol or internal error from an upstream gateway."""
