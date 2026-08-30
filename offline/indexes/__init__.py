"""Offline construction and publication of retrieval index bundles.

Runtime retrieval loads and searches the bundles created here.  This package
owns no query-serving APIs and deliberately keeps costly embedding and writes
outside the serving process.
"""

__all__: list[str] = []
