from __future__ import annotations

import importlib.util


def test_common_schema_package_is_absent() -> None:
    """Keep the retired generic schema package undiscoverable."""

    assert importlib.util.find_spec("hcmai.common.schemas") is None
