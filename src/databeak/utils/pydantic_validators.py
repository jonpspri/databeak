"""Shared Pydantic validators for DataBeak models."""

from __future__ import annotations

import json
from typing import Any, TypeVar

T = TypeVar("T")


def parse_json_string_to_dict(v: dict[str, Any] | str) -> dict[str, Any]:
    """Parse JSON string to dictionary for Claude Code compatibility.

    Args:
        v: Either a dictionary or JSON string that should parse to a dictionary

    Returns:
        Dictionary data

    Raises:
        ValueError: If JSON string is invalid or doesn't parse to dict
    """
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if not isinstance(parsed, dict):
                raise ValueError("JSON string must parse to dict")
            return parsed
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}") from e
    return v


def parse_json_string_to_dict_or_list(
    v: dict[str, Any] | list[Any] | str,
) -> dict[str, Any] | list[Any]:
    """Parse JSON string to dictionary or list for Claude Code compatibility.

    Args:
        v: Dictionary, list, or JSON string that should parse to dict or list

    Returns:
        Dictionary or list data

    Raises:
        ValueError: If JSON string is invalid or doesn't parse to dict/list
    """
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if not isinstance(parsed, dict | list):
                raise ValueError("JSON string must parse to dict or list")
            return parsed
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}") from e
    return v


def parse_json_string_to_list(v: list[Any] | str) -> list[Any]:
    """Parse JSON string to list for Claude Code compatibility.

    Args:
        v: Either a list or JSON string that should parse to a list

    Returns:
        List data

    Raises:
        ValueError: If JSON string is invalid or doesn't parse to list
    """
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if not isinstance(parsed, list):
                raise ValueError("JSON string must parse to list")
            return parsed
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}") from e
    return v
