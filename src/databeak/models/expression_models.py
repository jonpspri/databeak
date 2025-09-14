"""Pydantic models for secure mathematical expression validation.

This module provides validated models for mathematical expressions used in
DataBeak column operations, ensuring only safe expressions are accepted.

Author: DataBeak Security Team
Issue: #46 - Address pandas.eval() code injection vulnerability
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..utils.secure_evaluator import validate_expression_safety


class SecureExpression(BaseModel):
    """Validated mathematical expression that can be safely evaluated.

    This model ensures that only safe mathematical operations are allowed
    in user-provided expressions, preventing code injection attacks.

    Examples of valid expressions:
        - "col1 + col2"
        - "abs(col1) * 2.5"
        - "np.sqrt(col1 + col2)"
        - "(col1 + col2) / 2"
        - "max(col1, 100)"

    Examples of blocked expressions:
        - "__import__('os').system('rm -rf /')"
        - "exec('malicious_code')"
        - "open('/etc/passwd').read()"
    """

    model_config = ConfigDict(extra="forbid")

    expression: str = Field(
        description="Safe mathematical expression using column names and mathematical functions",
        min_length=1,
        max_length=1000,  # Prevent extremely long expressions
    )

    @field_validator("expression")
    @classmethod
    def validate_expression_safety(cls, v: str) -> str:
        """Validate that expression contains only safe mathematical operations.

        Args:
            v: Expression string to validate

        Returns:
            The validated expression string

        Raises:
            ValueError: If expression contains unsafe operations
        """
        try:
            validate_expression_safety(v)
        except Exception as e:
            raise ValueError(f"Unsafe expression: {e}") from e
        return v

    def __str__(self) -> str:
        """Return the expression string."""
        return self.expression


class ColumnFormula(BaseModel):
    """Formula specification for column creation with validation.

    Used in add_column operations where a formula is provided to compute column values from existing
    columns.
    """

    model_config = ConfigDict(extra="forbid")

    formula: SecureExpression = Field(
        description="Mathematical formula referencing existing columns"
    )
    description: str | None = Field(
        default=None,
        description="Optional description of what the formula computes",
        max_length=200,
    )

    def __str__(self) -> str:
        """Return the formula expression."""
        return self.formula.expression


class ApplyExpression(BaseModel):
    """Expression specification for apply operations with column context.

    Used in update_column operations where 'x' represents the current column values being
    transformed.
    """

    model_config = ConfigDict(extra="forbid")

    expression: SecureExpression = Field(
        description="Mathematical expression where 'x' represents column values"
    )
    variable_name: str = Field(
        default="x",
        description="Variable name used in expression to represent column values",
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",  # Valid Python identifier
    )

    @field_validator("variable_name")
    @classmethod
    def validate_variable_name(cls, v: str) -> str:
        """Validate that variable name is safe."""
        if v in ("exec", "eval", "open", "import", "__import__"):
            raise ValueError(f"Variable name '{v}' is not allowed")
        return v

    def get_expression_with_column(self, column_name: str) -> str:
        """Get expression with variable substituted for column reference.

        Args:
            column_name: Name of the column to substitute for the variable

        Returns:
            Expression string with column reference
        """
        # Safely quote column name for pandas
        safe_column_ref = f"`{column_name.replace('`', '')}`"
        return self.expression.expression.replace(self.variable_name, safe_column_ref)

    def __str__(self) -> str:
        """Return the expression string."""
        return self.expression.expression


class ConditionalExpression(BaseModel):
    """Conditional expression for advanced column operations.

    Supports expressions that evaluate to boolean values for filtering or conditional operations.
    """

    model_config = ConfigDict(extra="forbid")

    condition: SecureExpression = Field(description="Boolean expression for conditional logic")
    true_value: Any = Field(description="Value to use when condition is True")
    false_value: Any = Field(description="Value to use when condition is False")

    def __str__(self) -> str:
        """Return the condition expression."""
        return self.condition.expression


# Type aliases for convenience
Formula = SecureExpression
Expression = SecureExpression
SafeFormula = ColumnFormula
