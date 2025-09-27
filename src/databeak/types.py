"""MCP helpful types and functions."""

from typing import Annotated

from pydantic import Field

NonNegativeIntString = Annotated[str, Field(pattern=r"^(0|[1-9]\d*)$")]
PositiveIntString = Annotated[str, Field(pattern=r"^[1-9]\d*$")]
