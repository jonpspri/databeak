"""Pydantic validators for JSON string parsing compatibility."""

from __future__ import annotations

import json
from typing import Any, TypeVar

T = TypeVar("T")


# Implementation: JSON string to dict parsing with error handling for Claude Code compatibility
def parse_json_string_to_dict(v: dict[str, Any] | str) -> dict[str, Any]:
    """Parse JSON string to dictionary with validation."""
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if not isinstance(parsed, dict):
                raise ValueError("JSON string must parse to dict")
            return parsed
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}") from e
    return v


# Implementation: JSON string to dict or list parsing with type validation
def parse_json_string_to_dict_or_list(
    v: dict[str, Any] | list[Any] | str,
) -> dict[str, Any] | list[Any]:
    """Parse JSON string to dictionary or list with validation."""
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if not isinstance(parsed, dict | list):
                raise ValueError("JSON string must parse to dict or list")
            return parsed
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}") from e
    return v


# Implementation: JSON string to list parsing with type validation
def parse_json_string_to_list(v: list[Any] | str) -> list[Any]:
    """Parse JSON string to list with validation."""
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if not isinstance(parsed, list):
                raise ValueError("JSON string must parse to list")
            return parsed
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}") from e
    return v
