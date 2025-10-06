"""Pydantic validators for JSON string parsing compatibility."""

from __future__ import annotations

import json
from typing import Any, TypeVar

T = TypeVar("T")


# Implementation: JSON string to dict parsing with error handling for Claude Code compatibility
def parse_json_string_to_dict(
    v: dict[str, Any] | str,
) -> dict[str, Any]:  # Any justified: JSON can contain arbitrary structure
    """Parse JSON string to dictionary with validation."""
    if isinstance(v, dict):
        return v

    parsed = json.loads(v)
    if not isinstance(parsed, dict):
        msg = "JSON string must parse to dict"
        raise TypeError(msg)

    return parsed


# Implementation: JSON string to dict or list parsing with type validation
def parse_json_string_to_dict_or_list(  # Any justified: JSON arbitrary structure
    v: dict[str, Any] | list[Any] | str,
) -> dict[str, Any] | list[Any]:
    """Parse JSON string to dictionary or list with validation."""
    if isinstance(v, dict | list):
        return v

    parsed = json.loads(v)
    if not isinstance(parsed, dict | list):
        msg = "JSON string must parse to dict or list"
        raise TypeError(msg)

    return parsed


# Implementation: JSON string to list parsing with type validation
def parse_json_string_to_list(
    v: list[Any] | str,
) -> list[Any]:  # Any justified: JSON arbitrary structure
    """Parse JSON string to list with validation."""
    if isinstance(v, list):
        return v

    parsed = json.loads(v)
    if not isinstance(parsed, list):
        msg = "JSON string must parse to list"
        raise TypeError(msg)

    return parsed
