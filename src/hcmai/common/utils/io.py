"""Small helpers for reading and writing experiment data files."""

from __future__ import annotations

import json
import yaml
import pandas as pd

from os import PathLike
from pathlib import Path
from typing import Any


PathValue = str | PathLike[str]


def _prepare_output_path(path: PathValue) -> Path:
    """Return an output path and create its parent directory.

    Args:
        path (PathValue): Expected path to your output file

    Returns:
        A path to your output file.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def read_yaml(path: PathValue) -> Any:
    """Read a YAML document using PyYAML's safe loader.

    Args:
        path (PathValue): A path to your YAML file
    """

    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def write_yaml(
    data: Any,
    path: PathValue,
    *,
    sort_keys: bool = False,
) -> None:
    """Write one YAML document using PyYAML's safe dumper.

    Args:
    """

    output_path = _prepare_output_path(path)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            allow_unicode=True,
            sort_keys=sort_keys,
        )


def read_json(path: PathValue) -> Any:
    """Read and decode a JSON document."""

    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(
    data: Any,
    path: PathValue,
    *,
    indent: int | str | None = 2,
    ensure_ascii: bool = False,
) -> None:
    """Encode data as JSON and write it to ``path``."""

    output_path = _prepare_output_path(path)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=ensure_ascii,
            indent=indent,
        )
        file.write("\n")


def read_parquet(path: PathValue, **kwargs: Any) -> Any:
    """Read a Parquet file with pandas.

    Extra keyword arguments are passed to :func:`pandas.read_parquet`.
    """

    return pd.read_parquet(Path(path), **kwargs)


def write_parquet(
    data: Any,
    path: PathValue,
    **kwargs: Any,
) -> None:
    """Write a pandas-compatible table to Parquet.

    The object must provide a ``to_parquet`` method. Extra keyword arguments
    are passed to that method.
    """

    if not callable(getattr(data, "to_parquet", None)):
        raise TypeError("data must provide a callable to_parquet method")

    output_path = _prepare_output_path(path)
    data.to_parquet(output_path, **kwargs)


__all__ = [
    "read_json",
    "read_parquet",
    "read_yaml",
    "write_json",
    "write_parquet",
    "write_yaml",
]
