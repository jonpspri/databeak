"""FastMCP server for column-level data operations.

This server provides column selection, renaming, addition, removal, and type conversion.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

import pandas as pd
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field

from ..exceptions import (
    ColumnNotFoundError,
    InvalidParameterError,
    NoDataLoadedError,
    SessionNotFoundError,
)
from ..models import OperationType
from ..models.expression_models import SecureExpression
from ..models.tool_responses import BaseToolResponse, ColumnOperationResult
from ..utils.secure_evaluator import _get_secure_evaluator
from .server_utils import get_session_data

logger = logging.getLogger(__name__)


# Type aliases
CellValue = str | int | float | bool | None

# =============================================================================
# PYDANTIC MODELS FOR REQUEST PARAMETERS
# =============================================================================


class ColumnMapping(BaseModel):
    """Column rename mapping."""

    model_config = ConfigDict(extra="forbid")

    old_name: str = Field(description="Current column name")
    new_name: str = Field(description="New column name")


# Base class for update operations
class UpdateOperation(BaseModel):
    """Base class for update operations."""

    model_config = ConfigDict(extra="forbid")
    type: str = Field(description="Type of update operation")


class ReplaceOperation(UpdateOperation):
    """Replace operation specification."""

    type: Literal["replace"] = "replace"
    pattern: str = Field(description="Pattern to search for")
    replacement: str = Field(description="Replacement string")


class MapOperation(UpdateOperation):
    """Map operation specification."""

    type: Literal["map"] = "map"
    mapping: dict[str, Any] = Field(description="Value mapping dictionary")


class ApplyOperation(UpdateOperation):
    """Apply operation specification."""

    type: Literal["apply"] = "apply"
    expression: str = Field(description="Python expression to apply")


class FillNaOperation(UpdateOperation):
    """Fill NA operation specification."""

    type: Literal["fillna"] = "fillna"
    value: Any = Field(description="Value to fill NaN/null with")


# Discriminated union for update operations
UpdateOperationType = Annotated[
    ReplaceOperation | MapOperation | ApplyOperation | FillNaOperation, Field(discriminator="type")
]


class UpdateColumnRequest(BaseModel):
    """Request parameters for column update operations."""

    model_config = ConfigDict(extra="forbid")
    operation: Literal["replace", "map", "apply", "fillna"] = Field(
        description="Type of update operation"
    )
    value: Any | None = Field(
        None, description="Value for the operation (depends on operation type)"
    )
    pattern: str | None = Field(None, description="Pattern for replace operation")
    replacement: str | None = Field(None, description="Replacement for replace operation")


# =============================================================================
# RESPONSE MODELS (Server-specific)
# =============================================================================


class SelectColumnsResult(BaseToolResponse):
    """Result of selecting specific columns."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="Session identifier")
    selected_columns: list[str] = Field(description="List of selected column names")
    columns_before: int = Field(description="Number of columns before selection")
    columns_after: int = Field(description="Number of columns after selection")


class RenameColumnsResult(BaseToolResponse):
    """Result of renaming columns."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="Session identifier")
    renamed: dict[str, str] = Field(description="Mapping of old names to new names")
    columns: list[str] = Field(description="List of final column names")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


# Use shared implementation from server_utils
_get_session_data = get_session_data


# =============================================================================
# TOOL DEFINITIONS (Direct implementations)
# =============================================================================


# Implementation: validates column existence, reorders columns by selection order
# Creates new DataFrame copy with selected columns, records operation in session history
async def select_columns(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    columns: Annotated[list[str], Field(description="List of column names to select and keep")],
) -> SelectColumnsResult:
    """Select specific columns from dataframe, removing all others.

    Validates column existence and reorders by selection order. Returns selection details with
    before/after column counts.
    """
    try:
        # Get session_id from FastMCP context
        session_id = ctx.session_id
        session, df = _get_session_data(session_id)

        # Validate columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ColumnNotFoundError(missing_cols[0], df.columns.tolist())

        # Track counts before modification
        columns_before = len(df.columns)

        session.df = df[columns].copy()
        session.record_operation(
            OperationType.SELECT,
            {
                "columns": columns,
                "columns_before": df.columns.tolist(),
                "columns_after": columns,
            },
        )

        return SelectColumnsResult(
            session_id=session_id,
            selected_columns=columns,
            columns_before=columns_before,
            columns_after=len(columns),
        )

    except (SessionNotFoundError, NoDataLoadedError, ColumnNotFoundError) as e:
        logger.error(f"Select columns failed with {type(e).__name__}: {e.message}")
        raise ToolError(e.message) from e
    except Exception as e:
        logger.error(f"Unexpected error selecting columns: {e!s}")
        raise ToolError(f"Failed to select columns: {e}") from e


# Implementation: validates old column names exist in mapping keys, checks for naming conflicts
# Updates DataFrame columns in-place using pandas rename, records operation in session history
async def rename_columns(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    mapping: Annotated[
        dict[str, str], Field(description="Dictionary mapping old column names to new names")
    ],
) -> RenameColumnsResult:
    """Rename columns in the dataframe.

    Args:
        ctx: FastMCP context for session access
        mapping: Dict of old_name -> new_name

    Returns:
        Dict with rename details

    Examples:
        # Using dictionary mapping
        rename_columns(ctx, {"old_col1": "new_col1", "old_col2": "new_col2"})

        # Rename multiple columns
        rename_columns(ctx, {
            "FirstName": "first_name",
            "LastName": "last_name",
            "EmailAddress": "email"
        })
    """
    try:
        # Get session_id from FastMCP context
        session_id = ctx.session_id
        session, df = _get_session_data(session_id)

        # Validate columns exist
        missing_cols = [col for col in mapping if col not in df.columns]
        if missing_cols:
            raise ColumnNotFoundError(missing_cols[0], df.columns.tolist())

        # Apply renaming
        session.df = df.rename(columns=mapping)
        session.record_operation(
            OperationType.RENAME,
            {"mapping": mapping, "renamed_count": len(mapping)},
        )

        return RenameColumnsResult(
            session_id=session_id,
            renamed=mapping,
            columns=list(mapping.values()),
        )

    except (SessionNotFoundError, NoDataLoadedError, ColumnNotFoundError) as e:
        logger.error(f"Rename columns failed with {type(e).__name__}: {e.message}")
        raise ToolError(e.message) from e
    except Exception as e:
        logger.error(f"Unexpected error renaming columns: {e!s}")
        raise ToolError(f"Failed to rename columns: {e}") from e


# Implementation: validates column name doesn't exist, supports single value, list, or pandas eval formula
# Handles list length validation, formula evaluation with error handling, records operation
async def add_column(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    name: Annotated[str, Field(description="Name for the new column to add")],
    value: Annotated[
        CellValue | list[CellValue],
        Field(description="Single value for all rows or list of values (one per row)"),
    ] = None,
    formula: Annotated[
        SecureExpression | str | None,
        Field(
            description="Safe mathematical expression to compute column values (e.g., 'col1 + col2')"
        ),
    ] = None,
) -> ColumnOperationResult:
    """Add a new column to the dataframe.

    Args:
        ctx: FastMCP context for session access
        name: Name for the new column
        value: Single value for all rows, or list of values (one per row)
        formula: Python expression to compute values (e.g., "col1 + col2")

    Returns:
        ColumnOperationResult with operation details

    Examples:
        # Add column with constant value
        add_column(ctx, "status", "active")

        # Add column with list of values
        add_column(ctx, "scores", [85, 90, 78, 92, 88])

        # Add computed column
        add_column(ctx, "total", formula="price * quantity")

        # Add column with complex formula
        add_column(ctx, "full_name", formula="first_name + ' ' + last_name")
    """
    try:
        # Get session_id from FastMCP context
        session_id = ctx.session_id
        session, df = _get_session_data(session_id)

        if name in df.columns:
            raise InvalidParameterError("name", name, f"Column '{name}' already exists")

        if formula:
            try:
                # Convert string to SecureExpression if needed
                if isinstance(formula, str):
                    formula = SecureExpression(expression=formula)

                # Use secure evaluator instead of pandas.eval
                evaluator = _get_secure_evaluator()
                result = evaluator.evaluate_simple_formula(formula.expression, df)
                df[name] = result
            except Exception as e:
                raise InvalidParameterError("formula", formula, f"Invalid formula: {e}") from e
        elif isinstance(value, list):
            if len(value) != len(df):
                raise InvalidParameterError(
                    "value",
                    str(value),
                    f"List length ({len(value)}) must match row count ({len(df)})",
                )
            df[name] = value
        else:
            # Single value for all rows
            df[name] = value

        session.record_operation(
            OperationType.ADD_COLUMN,
            {
                "column": name,
                "value_type": "formula"
                if formula
                else "list"
                if isinstance(value, list)
                else "scalar",
            },
        )

        return ColumnOperationResult(
            session_id=session_id,
            operation="add",
            rows_affected=len(df),
            columns_affected=[name],
        )

    except (SessionNotFoundError, NoDataLoadedError, InvalidParameterError) as e:
        logger.error(f"Add column failed with {type(e).__name__}: {e.message}")
        raise ToolError(e.message) from e
    except Exception as e:
        logger.error(f"Unexpected error adding column: {e!s}")
        raise ToolError(f"Failed to add column: {e}") from e


# Implementation: validates columns exist before removal, prevents removing all columns
# Uses DataFrame drop with error handling, records operation in session history
async def remove_columns(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    columns: Annotated[
        list[str], Field(description="List of column names to remove from the dataframe")
    ],
) -> ColumnOperationResult:
    """Remove columns from the dataframe.

    Args:
        ctx: FastMCP context for session access
        columns: List of column names to remove

    Returns:
        ColumnOperationResult with removal details

    Examples:
        # Remove single column
        remove_columns(ctx, ["temp_column"])

        # Remove multiple columns
        remove_columns(ctx, ["col1", "col2", "col3"])

        # Clean up after analysis
        remove_columns(ctx, ["_temp", "_backup", "old_value"])
    """
    try:
        # Get session_id from FastMCP context
        session_id = ctx.session_id
        session, df = _get_session_data(session_id)

        # Validate columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ColumnNotFoundError(str(missing_cols[0]), df.columns.tolist())

        session.df = df.drop(columns=columns)
        session.record_operation(
            OperationType.REMOVE_COLUMN,
            {"columns": columns, "count": len(columns)},
        )

        return ColumnOperationResult(
            session_id=session_id,
            operation="remove",
            rows_affected=len(df),
            columns_affected=columns,
        )

    except (SessionNotFoundError, NoDataLoadedError, ColumnNotFoundError) as e:
        logger.error(f"Remove columns failed with {type(e).__name__}: {e.message}")
        raise ToolError(e.message) from e
    except Exception as e:
        logger.error(f"Unexpected error removing columns: {e!s}")
        raise ToolError(f"Failed to remove columns: {e}") from e


# Implementation: validates column exists, maps dtype to pandas types
# Uses pandas astype with error handling (raise/coerce), preserves original on failure
async def change_column_type(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    column: Annotated[str, Field(description="Column name to change data type for")],
    dtype: Annotated[
        Literal["int", "float", "str", "bool", "datetime"],
        Field(description="Target data type (int, float, str, bool, datetime)"),
    ],
    errors: Annotated[
        Literal["raise", "coerce"],
        Field(
            description="Error handling: 'raise' for errors, 'coerce' to replace invalid values with NaN"
        ),
    ] = "coerce",
) -> ColumnOperationResult:
    """Change the data type of a column.

    Args:
        ctx: FastMCP context for session access
        column: Column name to convert
        dtype: Target data type
        errors: How to handle conversion errors:
            - "raise": Raise an error if conversion fails
            - "coerce": Convert invalid values to NaN/None

    Returns:
        ColumnOperationResult with conversion details

    Examples:
        # Convert string numbers to integers
        change_column_type(ctx, "age", "int")

        # Convert to float, replacing errors with NaN
        change_column_type(ctx, "price", "float", errors="coerce")

        # Convert to datetime
        change_column_type(ctx, "date", "datetime")

        # Convert to boolean
        change_column_type(ctx, "is_active", "bool")
    """
    try:
        # Get session_id from FastMCP context
        session_id = ctx.session_id
        session, df = _get_session_data(session_id)

        if column not in df.columns:
            raise ColumnNotFoundError(column, df.columns.tolist())

        # Track before state
        original_dtype = str(df[column].dtype)
        null_count_before = df[column].isna().sum()

        # Map string dtype to pandas dtype
        type_map = {
            "int": "int64",
            "float": "float64",
            "str": "string",
            "bool": "bool",
            "datetime": "datetime64[ns]",
        }

        target_dtype = type_map.get(dtype)
        if not target_dtype:
            raise InvalidParameterError("dtype", dtype, f"Unsupported type: {dtype}")

        try:
            if dtype == "datetime":
                # Special handling for datetime conversion
                df[column] = pd.to_datetime(df[column], errors=errors)
            else:
                # General type conversion
                if errors == "coerce":
                    if dtype in ["int", "float"]:
                        df[column] = pd.to_numeric(df[column], errors="coerce")
                    else:
                        df[column] = df[column].astype(target_dtype)  # type: ignore[call-overload]
                else:
                    df[column] = df[column].astype(target_dtype)  # type: ignore[call-overload]

        except (ValueError, TypeError) as e:
            if errors == "raise":
                raise InvalidParameterError(
                    "column", column, f"Cannot convert to {dtype}: {e}"
                ) from e
            # If errors='coerce', the conversion has already handled invalid values

        # Track after state
        null_count_after = df[column].isna().sum()

        session.record_operation(
            OperationType.CHANGE_TYPE,
            {
                "column": column,
                "from_type": original_dtype,
                "to_type": dtype,
                "nulls_created": int(null_count_after - null_count_before),
            },
        )

        return ColumnOperationResult(
            session_id=session_id,
            operation=f"change_type_to_{dtype}",
            rows_affected=len(df),
            columns_affected=[column],
        )

    except (
        SessionNotFoundError,
        NoDataLoadedError,
        ColumnNotFoundError,
        InvalidParameterError,
    ) as e:
        logger.error(f"Change column type failed with {type(e).__name__}: {e.message}")
        raise ToolError(e.message) from e
    except Exception as e:
        logger.error(f"Unexpected error changing column type: {e!s}")
        raise ToolError(f"Failed to change column type: {e}") from e


async def update_column(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    column: Annotated[str, Field(description="Column name to update values in")],
    operation: Annotated[
        UpdateOperationType | UpdateColumnRequest | dict[str, Any],
        Field(description="Update operation specification (replace, map, apply, fillna)"),
    ],
) -> ColumnOperationResult:
    """Update values in a column using various operations with discriminated unions.

    Args:
        ctx: FastMCP context for session access
        column: Column name to update
        operation: Update operation specification (discriminated union or legacy dict)

    Returns:
        ColumnOperationResult with update details

    Examples:
        # Using discriminated union - Replace operation
        update_column(ctx, "status", {
            "type": "replace",
            "pattern": "N/A",
            "replacement": "Unknown"
        })

        # Using discriminated union - Map operation
        update_column(ctx, "code", {
            "type": "map",
            "mapping": {"A": "Alpha", "B": "Beta"}
        })

        # Using discriminated union - Fill operation
        update_column(ctx, "score", {
            "type": "fillna",
            "value": 0
        })

        # Legacy format still supported
        update_column(ctx, "score", {
            "operation": "fillna",
            "value": 0
        })
    """
    try:
        # Get session_id from FastMCP context
        session_id = ctx.session_id
        session, df = _get_session_data(session_id)

        if column not in df.columns:
            raise ColumnNotFoundError(column, df.columns.tolist())

        # Track initial state
        null_count_before = df[column].isna().sum()
        operation_type = "unknown"

        # Handle discriminated union operations
        if isinstance(
            operation, ReplaceOperation | MapOperation | ApplyOperation | FillNaOperation
        ):
            if isinstance(operation, ReplaceOperation):
                operation_type = "replace"
                df[column] = df[column].replace(operation.pattern, operation.replacement)
            elif isinstance(operation, MapOperation):
                operation_type = "map"
                df[column] = df[column].map(operation.mapping)
            elif isinstance(operation, ApplyOperation):
                operation_type = "apply"
                expr = operation.expression

                # Handle string operations that pandas.eval can't handle
                if (
                    ".upper()" in expr
                    or ".lower()" in expr
                    or ".strip()" in expr
                    or ".title()" in expr
                ):
                    # String operations
                    if expr == "x.upper()":
                        df[column] = df[column].str.upper()
                    elif expr == "x.lower()":
                        df[column] = df[column].str.lower()
                    elif expr == "x.strip()":
                        df[column] = df[column].str.strip()
                    elif expr == "x.title()":
                        df[column] = df[column].str.title()
                    else:
                        raise InvalidParameterError(
                            "expression",
                            operation.expression,
                            "For string operations, use exact expressions: 'x.upper()', 'x.lower()', 'x.strip()', 'x.title()'",
                        )
                else:
                    # Use secure evaluator for mathematical expressions
                    try:
                        # Create column context mapping for 'x' variable
                        column_context = {"x": column}
                        # Use secure evaluator instead of pandas.eval
                        evaluator = _get_secure_evaluator()
                        result = evaluator.evaluate_column_expression(expr, df, column_context)
                        df[column] = result
                    except Exception as e:
                        raise InvalidParameterError(
                            "expression",
                            operation.expression,
                            f"Invalid expression. Use 'x' to reference column values. Error: {e}",
                        ) from e
            elif isinstance(operation, FillNaOperation):
                operation_type = "fillna"
                df[column] = df[column].fillna(operation.value)

        else:
            # Handle legacy format or dict input
            if isinstance(operation, dict):
                if "type" in operation:
                    # Try to parse as discriminated union
                    try:
                        if operation["type"] == "replace":
                            replace_op = ReplaceOperation(**operation)
                            operation_type = "replace"
                            df[column] = df[column].replace(
                                replace_op.pattern, replace_op.replacement
                            )
                        elif operation["type"] == "map":
                            map_op = MapOperation(**operation)
                            operation_type = "map"
                            df[column] = df[column].map(map_op.mapping)
                        elif operation["type"] == "apply":
                            apply_op = ApplyOperation(**operation)
                            operation_type = "apply"
                            expr = apply_op.expression

                            # Handle string operations that pandas.eval can't handle
                            if (
                                ".upper()" in expr
                                or ".lower()" in expr
                                or ".strip()" in expr
                                or ".title()" in expr
                            ):
                                # String operations
                                if expr == "x.upper()":
                                    df[column] = df[column].str.upper()
                                elif expr == "x.lower()":
                                    df[column] = df[column].str.lower()
                                elif expr == "x.strip()":
                                    df[column] = df[column].str.strip()
                                elif expr == "x.title()":
                                    df[column] = df[column].str.title()
                                else:
                                    raise InvalidParameterError(
                                        "expression",
                                        apply_op.expression,
                                        "For string operations, use exact expressions: 'x.upper()', 'x.lower()', 'x.strip()', 'x.title()'",
                                    )
                            else:
                                # Use secure evaluator for mathematical expressions
                                try:
                                    # Create column context mapping for 'x' variable
                                    column_context = {"x": column}
                                    # Use secure evaluator instead of pandas.eval
                                    evaluator = _get_secure_evaluator()
                                    result = evaluator.evaluate_column_expression(
                                        expr, df, column_context
                                    )
                                    df[column] = result
                                except Exception as e:
                                    raise InvalidParameterError(
                                        "expression",
                                        apply_op.expression,
                                        f"Invalid expression. Use 'x' to reference column values. Error: {e}",
                                    ) from e
                        elif operation["type"] == "fillna":
                            fillna_op = FillNaOperation(**operation)
                            operation_type = "fillna"
                            df[column] = df[column].fillna(fillna_op.value)
                        else:
                            raise InvalidParameterError(
                                "type",
                                operation["type"],
                                "Supported types: replace, map, apply, fillna",
                            )
                    except Exception as e:
                        raise InvalidParameterError(
                            "operation", str(operation), f"Invalid operation specification: {e}"
                        ) from e
                else:
                    # Legacy format with "operation" field
                    update_request = UpdateColumnRequest(**operation)
                    operation_type = update_request.operation

                    if update_request.operation == "replace":
                        if update_request.pattern is None or update_request.replacement is None:
                            raise InvalidParameterError(
                                "pattern/replacement",
                                f"{update_request.pattern}/{update_request.replacement}",
                                "Both pattern and replacement required for replace operation",
                            )
                        df[column] = df[column].replace(
                            update_request.pattern, update_request.replacement
                        )
                    elif update_request.operation == "map":
                        if not isinstance(update_request.value, dict):
                            raise InvalidParameterError(
                                "value",
                                str(update_request.value),
                                "Dictionary mapping required for map operation",
                            )
                        df[column] = df[column].map(update_request.value)
                    elif update_request.operation == "apply":
                        if update_request.value is None:
                            raise InvalidParameterError(
                                "value",
                                str(update_request.value),
                                "Expression required for apply operation",
                            )
                        if isinstance(update_request.value, str):
                            expr = update_request.value

                            # Handle string operations that pandas.eval can't handle
                            if (
                                ".upper()" in expr
                                or ".lower()" in expr
                                or ".strip()" in expr
                                or ".title()" in expr
                            ):
                                # String operations
                                if expr == "x.upper()":
                                    df[column] = df[column].str.upper()
                                elif expr == "x.lower()":
                                    df[column] = df[column].str.lower()
                                elif expr == "x.strip()":
                                    df[column] = df[column].str.strip()
                                elif expr == "x.title()":
                                    df[column] = df[column].str.title()
                                else:
                                    raise InvalidParameterError(
                                        "value",
                                        update_request.value,
                                        "For string operations, use exact expressions: 'x.upper()', 'x.lower()', 'x.strip()', 'x.title()'",
                                    )
                            else:
                                # Use secure evaluator for mathematical expressions
                                try:
                                    # Create column context mapping for 'x' variable
                                    column_context = {"x": column}
                                    # Use secure evaluator instead of pandas.eval
                                    evaluator = _get_secure_evaluator()
                                    result = evaluator.evaluate_column_expression(
                                        expr, df, column_context
                                    )
                                    df[column] = result
                                except Exception as e:
                                    raise InvalidParameterError(
                                        "value",
                                        update_request.value,
                                        f"Invalid expression. Use 'x' to reference column values. Error: {e}",
                                    ) from e
                        else:
                            df[column] = df[column].apply(update_request.value)
                    elif update_request.operation == "fillna":
                        if update_request.value is None:
                            raise InvalidParameterError(
                                "value",
                                str(update_request.value),
                                "Fill value required for fillna operation",
                            )
                        df[column] = df[column].fillna(update_request.value)
                    else:
                        raise InvalidParameterError(
                            "operation",
                            update_request.operation,
                            "Supported operations: replace, map, apply, fillna",
                        )
            else:
                # Handle legacy UpdateColumnRequest object
                update_request = operation
                operation_type = update_request.operation
                # ... (same logic as above legacy handling)

        # Track changes
        null_count_after = df[column].isna().sum()

        session.record_operation(
            OperationType.UPDATE_COLUMN,
            {
                "column": column,
                "operation": operation_type,
                "nulls_changed": int(null_count_after - null_count_before),
            },
        )

        return ColumnOperationResult(
            session_id=session_id,
            operation=f"update_{operation_type}",
            rows_affected=len(df),
            columns_affected=[column],
        )

    except (
        SessionNotFoundError,
        NoDataLoadedError,
        ColumnNotFoundError,
        InvalidParameterError,
    ) as e:
        logger.error(f"Update column failed with {type(e).__name__}: {e.message}")
        raise ToolError(e.message) from e
    except Exception as e:
        logger.error(f"Unexpected error updating column: {e!s}")
        raise ToolError(f"Failed to update column: {e}") from e


# =============================================================================
# SERVER INITIALIZATION
# =============================================================================

column_server = FastMCP(
    "DataBeak Column Operations Server",
    instructions="Column-level operations server providing selection, renaming, addition, removal, and type conversion",
)

# Register the functions as MCP tools
column_server.tool(name="select_columns")(select_columns)
column_server.tool(name="rename_columns")(rename_columns)
column_server.tool(name="add_column")(add_column)
column_server.tool(name="remove_columns")(remove_columns)
column_server.tool(name="change_column_type")(change_column_type)
column_server.tool(name="update_column")(update_column)
