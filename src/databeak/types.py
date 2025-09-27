"""MCP helpful types and functions."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, NonNegativeInt

NonNegativeIntString = Annotated[str, Field(pattern=r"^(0|[1-9]\d*)$")]
PositiveIntString = Annotated[str, Field(pattern=r"^[1-9]\d*$")]

# Special type for handling "null" string that should be converted to None
NullableIntString = Annotated[str, Field(pattern=r"^(null|(0|[1-9]\d*))$")]


def parse_nullable_int_string(value: str) -> int | None:
    """Convert string representation to int or None.

    Args:
        value: String value ("null", "0", "123", etc.)

    Returns:
        Parsed integer or None for "null"

    Raises:
        ValueError: If string is not valid

    """
    if value == "null":
        return None
    try:
        return int(value)
    except ValueError as e:
        msg = f"Invalid integer string: {value}"
        raise ValueError(msg) from e


class AutoDetectHeader(BaseModel):
    """Auto-detect whether file has headers using pandas inference."""

    mode: Literal["auto"] = "auto"


class NoHeader(BaseModel):
    """File has no headers - generate default column names (Column_0, Column_1, etc.)."""

    mode: Literal["none"] = "none"


class ExplicitHeaderRow(BaseModel):
    """Use specific row number as header."""

    mode: Literal["row"] = "row"
    row_number: NonNegativeInt = Field(description="Row number to use as header (0-based)")


HeaderConfig = AutoDetectHeader | NoHeader | ExplicitHeaderRow


def resolve_header_param(config: HeaderConfig) -> int | None | Literal["infer"]:
    """Convert HeaderConfig to pandas read_csv header parameter.

    Args:
        config: Header configuration object

    Returns:
        Value for pandas read_csv header parameter

    """
    if isinstance(config, AutoDetectHeader):
        return "infer"  # Let pandas auto-detect headers
    if isinstance(config, NoHeader):
        return None  # No headers, generate default column names
    if isinstance(config, ExplicitHeaderRow):
        return config.row_number
    # This should never happen with proper discriminated union, but added for safety
    msg = f"Unknown header config type: {type(config)}"
    raise ValueError(msg)
